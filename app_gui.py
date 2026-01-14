import streamlit as st
import pandas as pd
import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
import os
import glob
import re
import matplotlib.pyplot as plt

# 设置页面宽屏模式
st.set_page_config(layout="wide", page_title="EMG 数据切分调试器")

# ================= 核心逻辑  =================
def parse_filename(filename):
    # 匹配 DFx.y 中的 x
    label_match = re.search(r'DF(\d+)\.', filename)
    ts_match = re.search(r'(\d{14})\.csv$', filename)
    label = int(label_match.group(1)) if label_match else None
    return label

def load_data(path):
    df = pd.read_csv(path)
    # 提取 CH1-CH5
    cols = [c for c in df.columns if 'CH' in c]
    return df[cols].values, df['Timestamp'].values if 'Timestamp' in df.columns else None

def process_signal(data, fs, low_cut, high_cut, smooth_ms, merge_ms, threshold_ratio):
    """
    为了可视化，我们需要返回中间过程变量
    """
    # 1. 滤波
    b, a = signal.butter(4, [low_cut, high_cut], btype='bandpass', fs=fs)
    filtered = signal.filtfilt(b, a, data, axis=0)
    
    # 2. 能量 (RMS)
    energy = np.sqrt(np.mean(filtered**2, axis=1))
    
    # 3. 平滑
    win_len = int((smooth_ms/1000) * fs)
    if win_len < 1: win_len = 1
    energy_smooth = np.convolve(energy, np.ones(win_len)/win_len, mode='same')
    
    # 4. 阈值计算
    noise_floor = np.percentile(energy_smooth, 10)
    peak_level = np.percentile(energy_smooth, 99)
    threshold = noise_floor + threshold_ratio * (peak_level - noise_floor)
    
    # 5. 掩码
    mask = energy_smooth > threshold
    
    # 6. 缝合
    gap_samples = int((merge_ms/1000) * fs)
    if gap_samples > 0:
        mask = ndimage.binary_closing(mask, structure=np.ones(gap_samples))
        
    return filtered, energy_smooth, threshold, mask

# ================= 侧边栏：动态数据筛选 =================
st.sidebar.header("📂 1. 数据筛选")

# 1. 固定根目录
data_root = "data"

if not os.path.exists(data_root):
    st.sidebar.error(f"未找到根目录: {data_root}")
    st.stop()

# 2. 选择测试者姓名 (Subject)
subjects = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
if not subjects:
    st.sidebar.warning("data 目录下没有文件夹")
    st.stop()
subject = st.sidebar.selectbox("选择测试者姓名 (Subject)", sorted(subjects))

# 3. 选择日期 (Date)
subject_path = os.path.join(data_root, subject)
dates = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]
if not dates:
    st.sidebar.warning(f"{subject} 目录下没有日期文件夹")
    st.stop()
date_str = st.sidebar.selectbox("选择日期 (Date)", sorted(dates))

# 4. 自动扫描该日期文件夹下的所有标签 (DF Label)
# 根据 preprocess.py 的文件路径规范搜索
search_path = os.path.join(data_root, subject, date_str, "RAW_EMG*.csv")
all_files_in_folder = glob.glob(search_path)

if not all_files_in_folder:
    st.sidebar.warning(f"该目录下未发现 RAW_EMG 文件")
    st.stop()

# 提取所有存在的标签
available_labels = set()
for f in all_files_in_folder:
    label = parse_filename(os.path.basename(f)) # 使用已有的解析函数
    if label is not None:
        available_labels.add(label)

if not available_labels:
    st.sidebar.error("无法从文件名中解析出动作标签")
    st.stop()

target_label = st.sidebar.selectbox("选择动作标签 (DF Label)", sorted(list(available_labels)))

st.sidebar.markdown("---")
st.sidebar.header("🎛️ 2. 算法参数微调")

# 采样率
fs = st.sidebar.number_input("采样率 (Hz)", value=1000)

# 滤波范围
band_range = st.sidebar.slider("带通滤波范围 (Hz)", 0, 500, (20, 450))

# VAD 参数
st.sidebar.subheader("VAD (活动检测) 参数")
thresh_ratio = st.sidebar.slider("阈值系数 (越小越灵敏)", 0.05, 0.50, 0.15, 0.01)
smooth_ms = st.sidebar.slider("能量平滑窗口 (ms)", 10, 500, 200, 10)
merge_ms = st.sidebar.slider("合并间隙 (ms)", 0, 1000, 300, 50)
st.sidebar.markdown("---")
st.sidebar.subheader("📅 节奏过滤参数")
use_rhythm_filter = st.sidebar.checkbox("开启等间距过滤", value=True)
interval_ratio = st.sidebar.slider("最小间距比例 (Interval Ratio)", 0.1, 0.9, 0.7, 0.05)

# ================= 主界面 =================
st.title("⚡ EMG 信号切分可视化")

# 1. 寻找文件
search_path = os.path.join(data_root, subject, date_str, "RAW_EMG*.csv")
files = glob.glob(search_path)

# 筛选符合 Label 的文件
matched_files = []
for f in files:
    l = parse_filename(os.path.basename(f))
    if l == target_label:
        matched_files.append(f)

if not matched_files:
    st.warning(f"❌ 未找到匹配文件！\n路径: `{search_path}`\n标签: `{target_label}`")
    st.info("请检查文件夹结构是否为 data/姓名/日期/RAW_EMG...DF{标签}...")
    st.stop()

# 2. 文件选择下拉框 (如果有多个)
selected_file = st.selectbox("找到以下文件 (选择一个查看):", matched_files, format_func=lambda x: os.path.basename(x))

# 3. 加载与处理
if selected_file:
    with st.spinner('正在处理信号...'):
        raw_data, _ = load_data(selected_file)
        
        # 运行算法 (获取基础 VAD 掩码)
        filtered_data, energy, threshold, mask = process_signal(
            raw_data, fs, band_range[0], band_range[1], 
            smooth_ms, merge_ms, thresh_ratio
        )
        
        # --- [NEW] 第一步：提取初始片段 ---
        labeled_mask, num_raw_segments = ndimage.label(mask)
        
        # 使用 find_objects 一次性获取所有片段的切片 (Slice)
        # slices[i] 对应 label i+1 的位置
        slices = ndimage.find_objects(labeled_mask)
        
        raw_segments = []
        for i, sl in enumerate(slices):
            if sl is None: continue
            
            # sl 是一个 tuple，对于 1D 数组，sl[0] 就是我们要的切片
            # start = sl[0].start, end = sl[0].stop
            start = sl[0].start
            end = sl[0].stop
            length = end - start
            
            # 简单的长度过滤 (防止极短的噪点)
            if length > 10: 
                raw_segments.append({
                    'id': i + 1,
                    'start': start,
                    'end': end,
                    'center': (start + end) / 2,
                    # 'indices': ... # [优化] 不再存储巨大的索引数组，省内存
                })
        
        # --- [优化版] 第二步：高能异常过滤 (去除翻腕) ---
        valid_segments_step1 = []
        if len(raw_segments) > 0:
            segment_energies = []
            for seg in raw_segments:
                # 利用切片直接访问，无需 fancy indexing
                seg_data = filtered_data[seg['start']:seg['end']]
                
                # 计算 RMS
                if len(seg_data) > 0:
                    rms = np.mean(np.sqrt(np.mean(seg_data**2, axis=0)))
                else:
                    rms = 0
                segment_energies.append(rms)
            
            # 计算基准
            median_energy = np.median(segment_energies) if segment_energies else 0
            energy_limit = median_energy * 5.0
            
            rejected_segments = []
            
            for i, seg in enumerate(raw_segments):
                if len(raw_segments) == 1 or segment_energies[i] < energy_limit:
                    valid_segments_step1.append(seg)
                else:
                    rejected_segments.append(seg)
        else:
            valid_segments_step1 = raw_segments

        # --- [优化版] 第三步：节奏过滤 ---
        final_segments = []
        if use_rhythm_filter and len(valid_segments_step1) > 1:
            centers = [s['center'] for s in valid_segments_step1]
            diffs = np.diff(centers)
            if len(diffs) > 0:
                median_interval = np.median(diffs)
                
                final_segments.append(valid_segments_step1[0])
                last_valid_center = valid_segments_step1[0]['center']
                
                for i in range(1, len(valid_segments_step1)):
                    curr_seg = valid_segments_step1[i]
                    if (curr_seg['center'] - last_valid_center) > median_interval * interval_ratio:
                        final_segments.append(curr_seg)
                        last_valid_center = curr_seg['center']
            else:
                 final_segments = valid_segments_step1
        else:
            final_segments = valid_segments_step1
            
        # --- [优化版] 第四步：重建最终 Mask ---
        final_mask = np.zeros_like(mask)
        for seg in final_segments:
            # [优化] 使用切片赋值，速度极快
            final_mask[seg['start']:seg['end']] = True
            
        mask = final_mask
        labeled_mask, num_segments = ndimage.label(mask)

    # ================= 绘图区域 =================
    
    # 图表 1: 信号全览
    st.subheader(f"📊 信号概览 (检测到 {num_segments} 个有效动作)")
    
    if len(raw_segments) > len(final_segments):
        st.caption(f"ℹ️ 已自动剔除 {len(raw_segments) - len(final_segments)} 个异常/干扰片段 (翻腕或非节奏动作)")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 时间轴
    t = np.arange(len(raw_data)) / fs
    
    # 子图1: 原始信号 vs 滤波信号 (只画 CH1 避免混乱)
    ax1.plot(t, raw_data[:, 0], color='lightgray', alpha=0.6, label='Raw CH1')
    ax1.plot(t, filtered_data[:, 0], color='#1f77b4', linewidth=1, label='Filtered CH1')
    ax1.set_ylabel('Amplitude')
    ax1.legend(loc='upper right')
    ax1.set_title('EMG Signal (Channel 1)')
    
    # 子图2: 能量与切分
    ax2.plot(t, energy, color='orange', label='Energy Envelope')
    ax2.axhline(threshold, color='red', linestyle='--', label='Threshold')
    
    # 画出切分区域 (Green Zones)
    # 使用 fill_between
    ax2.fill_between(t, 0, np.max(energy), where=mask, color='green', alpha=0.3, label='Detected Action')
    
    ax2.set_ylabel('Energy')
    ax2.set_xlabel('Time (s)')
    ax2.legend(loc='upper right')
    ax2.set_title(f'Activity Detection (Threshold Ratio: {thresh_ratio})')
    
    st.pyplot(fig)
    
    # ================= 详细切片展示 (修改部分) =================
    st.subheader("🔍 动作切片详情")

    if num_segments > 0:
        seg_id = st.slider("查看第几个动作片段?", 1, num_segments, 1)
        
        indices = np.where(labeled_mask == seg_id)[0]
        start, end = indices[0], indices[-1]
        
        # 提取选中的片段数据 (以 CH1 为例，或者让用户选通道)
        seg_data = filtered_data[start:end] 
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**时域波形 (Segment #{seg_id})**")
            fig_seg, ax_seg = plt.subplots(figsize=(6, 4))
            ax_seg.plot(seg_data) # 画出所有通道
            ax_seg.set_xlabel("Samples")
            ax_seg.set_ylabel("Amplitude")
            st.pyplot(fig_seg)

        with col2:
            st.markdown("时频分析 (STFT)")
            
            # --- 交互控制区 (改动：移至侧边栏) ---
            st.sidebar.markdown("---")
            st.sidebar.subheader("🔍 STFT 详细分析参数")
            
            # 1. 通道选择
            stft_ch_idx = st.sidebar.selectbox(
                "STFT 选择通道", 
                range(seg_data.shape[1]), 
                format_func=lambda x: f"Channel {x+1}"
            )
            
            # 2. 窗口大小 (nperseg)
            # 较小的值 = 时间分辨率高，频率分辨率低
            # 较大的值 = 时间分辨率低，频率分辨率高
            nperseg = st.sidebar.selectbox(
                "STFT 窗口大小 (nperseg)", 
                [32, 64, 128, 256], 
                index=1,
                help="小窗口提升时间分辨率，大窗口提升频率分辨率"
            )
            
            # 3. 重叠 (Overlap)
            # 通常取窗口的一半或更多，使图像更平滑
            noverlap = st.sidebar.slider("STFT 重叠点数", 0, nperseg-1, nperseg//2)
            
            # 4. 显示设置
            use_log_scale = st.sidebar.checkbox("STFT 使用对数刻度 (dB)", value=True, help="能更清晰地看到低能量的频率成分")
            max_freq_view = st.sidebar.slider("STFT 显示最大频率 (Hz)", 50, int(fs/2), 500)

            # --- 计算 STFT ---
            f_stft, t_stft, Zxx = signal.stft(
                seg_data[:, stft_ch_idx], 
                fs=fs, 
                nperseg=nperseg, 
                noverlap=noverlap
            )
            
            # 处理幅值
            magnitude = np.abs(Zxx)
            if use_log_scale:
                # 转换为 dB，加一个微小量防止 log(0)
                magnitude = 20 * np.log10(magnitude + 1e-6)
                cbar_label = 'Intensity (dB)'
            else:
                cbar_label = 'Intensity (Amplitude)'

            # --- 绘图 ---
            fig_stft, ax_stft = plt.subplots(figsize=(6, 4))
            
            # 使用 pcolormesh 绘制
            # shading='gouraud' 会让图像更平滑好看
            pcm = ax_stft.pcolormesh(t_stft, f_stft, magnitude, shading='gouraud', cmap='jet')
            
            ax_stft.set_ylabel('Frequency [Hz]')
            ax_stft.set_xlabel('Time [sec]')
            ax_stft.set_ylim(0, max_freq_view) # 动态限制频率范围
            ax_stft.set_title(f'Channel {stft_ch_idx+1} Spectrogram')
            
            # 颜色条
            fig_stft.colorbar(pcm, ax=ax_stft, label=cbar_label)
            st.pyplot(fig_stft)

    # ================= 新增：所有切片缩略图概览 =================
    st.markdown("---")
    st.subheader("🖼️ 所有切片缩略图概览 (Gallery Mode)")
    
    show_gallery = st.checkbox("展开查看所有动作切片", value=False)
    
    if show_gallery and num_segments > 0:
        cols_count = st.slider("每行显示数量", 3, 15, 5)
        
        # 这里的逻辑是：分块遍历，每次处理一行
        for i in range(1, num_segments + 1, cols_count):
            cols = st.columns(cols_count)
            
            # 在当前行的一组 columns 中填充内容
            for j in range(cols_count):
                current_seg_id = i + j
                
                if current_seg_id <= num_segments:
                    with cols[j]:
                        # 1. 提取当前片段数据
                        indices = np.where(labeled_mask == current_seg_id)[0]
                        if len(indices) > 0:
                            s, e = indices[0], indices[-1]
                            # 提取滤波后的数据用于展示
                            seg_thumb = filtered_data[s:e]
                            
                            # 2. 绘制微型图
                            # figsize设置得较小，去除多余元素
                            fig_thumb, ax_thumb = plt.subplots(figsize=(3, 2))
                            ax_thumb.plot(seg_thumb, linewidth=0.8)
                            ax_thumb.set_title(f"#{current_seg_id}", fontsize=10)
                            ax_thumb.axis('off') # 关闭坐标轴，让看起来更像缩略图
                            
                            st.pyplot(fig_thumb)
                            plt.close(fig_thumb) # 显式关闭，防止内存泄漏