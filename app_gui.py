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

# ================= 核心逻辑 (保持不变) =================
def parse_filename(filename):
    label_match = re.search(r'DF(\d+)\.', filename)
    label = int(label_match.group(1)) if label_match else None
    return label

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    cols = [c for c in df.columns if 'CH' in c]
    data = df[cols].values  # 将变量名存为 data 以便操作
    
    # === CH5 信号修正 ===
    if data.shape[1] >= 5:
        data[:, 4] = data[:, 4] * 2.5
    # =========================

    return data, df['Timestamp'].values if 'Timestamp' in df.columns else None

def refine_mask_logic(mask, fs):
    """
    业务逻辑优化掩码：
    1. > 5s: 视为粘连，尝试断开。
    2. 1s < len <= 5s: 丢弃。
    3. 500ms < len <= 1s: 截取中间 500ms。
    4. <= 500ms: 保留。
    """
    labeled, num = ndimage.label(mask)
    new_mask = np.zeros_like(mask, dtype=bool)
    samples_500ms = int(0.5 * fs)
    
    for i in range(1, num + 1):
        loc = np.where(labeled == i)[0]
        if len(loc) == 0: continue
        
        duration_ms = (len(loc) / fs) * 1000
        
        if duration_ms > 5000:
            seg_mask = np.zeros_like(mask)
            seg_mask[loc] = True
            structure = np.ones(int(0.2 * fs))
            opened_mask = ndimage.binary_opening(seg_mask, structure=structure)
            sub_labeled, sub_num = ndimage.label(opened_mask)
            for j in range(1, sub_num + 1):
                sub_loc = np.where(sub_labeled == j)[0]
                sub_dur = (len(sub_loc) / fs) * 1000
                if sub_dur <= 1000:
                    if 500 < sub_dur <= 1000:
                        center = int(np.mean(sub_loc))
                        half = samples_500ms // 2
                        s = max(0, center - half)
                        e = min(len(mask), center + half)
                        new_mask[s:e] = True
                    else:
                        new_mask[sub_loc] = True
        elif 1000 < duration_ms <= 5000:
            continue
        elif 500 < duration_ms <= 1000:
            center = int(np.mean(loc))
            half = samples_500ms // 2
            start = max(0, center - half)
            end = min(len(mask), center + half)
            new_mask[start:end] = True
        else:
            new_mask[loc] = True
            
    return new_mask

@st.cache_data
def process_signal(data, fs, low_cut, high_cut, smooth_ms, merge_ms, threshold_ratio, use_refine=True, use_notch=False, notch_freq=50):
    # --- 新增 (去除工频干扰) ---
    if use_notch:
        # Q值决定陷波的宽度，30 是一个比较通用的值
        b_notch, a_notch = signal.iirnotch(notch_freq, 30, fs)
        # 先进行陷波滤波
        data = signal.filtfilt(b_notch, a_notch, data, axis=0)
    
    # --- 原有：带通滤波 ---
    b, a = signal.butter(4, [low_cut, high_cut], btype='bandpass', fs=fs)
    filtered = signal.filtfilt(b, a, data, axis=0)
    
    energy = np.sqrt(np.mean(filtered**2, axis=1))
    
    win_len = int((smooth_ms/1000) * fs)
    if win_len < 1: win_len = 1
    energy_smooth = np.convolve(energy, np.ones(win_len)/win_len, mode='same')
    
    noise_floor = np.percentile(energy_smooth, 10)
    peak_level = np.percentile(energy_smooth, 99)
    threshold = noise_floor + threshold_ratio * (peak_level - noise_floor)
    
    raw_mask = energy_smooth > threshold
    
    gap_samples = int((merge_ms/1000) * fs)
    if gap_samples > 0:
        raw_mask = ndimage.binary_closing(raw_mask, structure=np.ones(gap_samples))
        
    if use_refine:
        final_mask = refine_mask_logic(raw_mask, fs)
    else:
        final_mask = raw_mask
        
    return filtered, energy, threshold, raw_mask, final_mask

# ================= 侧边栏：配置 =================
st.sidebar.header("📂 1. 数据与通道")

data_root = "data"
if not os.path.exists(data_root):
    st.sidebar.error(f"未找到根目录: {data_root}")
    st.stop()

subjects = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
subject = st.sidebar.selectbox("选择测试者", sorted(subjects)) if subjects else None

dates = []
if subject:
    subject_path = os.path.join(data_root, subject)
    dates = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]
date_str = st.sidebar.selectbox("选择日期", sorted(dates)) if dates else None

target_label = None
if subject and date_str:
    search_path = os.path.join(data_root, subject, date_str, "RAW_EMG*.csv")
    all_files = glob.glob(search_path)
    available_labels = sorted(list(set([parse_filename(os.path.basename(f)) for f in all_files if parse_filename(os.path.basename(f)) is not None])))
    if available_labels:
        target_label = st.sidebar.selectbox("选择动作标签", available_labels)

# --- 新增：通道选择 ---
st.sidebar.markdown("---")
view_ch = st.sidebar.number_input("👁️ 可视化通道 (CH)", 1, 8, 1) - 1 # 转为索引

st.sidebar.markdown("---")
st.sidebar.header("🎛️ 算法参数")

# 使用 form 表单包裹所有参数，避免拖动滑块时频繁刷新卡顿
with st.sidebar.form("analysis_config"):
    st.markdown("### 基础设置")
    fs = st.number_input("采样率 (Hz)", value=1000)
    band_range = st.slider("带通滤波 (Hz)", 0, 500, (20, 450))

    st.markdown("### 工频干扰去除")
    use_notch = st.checkbox("🔌 启用工频陷波 (Notch)", value=False, help="去除 50Hz/60Hz 电源噪声")
    notch_freq = st.selectbox("干扰频率 (Hz)", [50, 60], index=0)

    st.markdown("### VAD 检测")
    thresh_ratio = st.slider("阈值系数", 0.05, 0.50, 0.15, 0.01)
    smooth_ms = st.slider("平滑窗口 (ms)", 10, 500, 100, 10)
    merge_ms = st.slider("合并间隙 (ms)", 0, 1000, 200, 50)

    st.markdown("### 过滤逻辑")
    use_refine_logic = st.checkbox("启用时长门控 (1s/500ms)", value=True)
    use_rhythm = st.checkbox("启用 4s 节奏过滤", value=True)
    interval_ratio = st.slider("最小间距比例", 0.1, 1.0, 0.9)

    # 提交按钮
    submitted = st.form_submit_button("🚀 重新分析")

# STFT 参数移动到这里
st.sidebar.markdown("---")
with st.sidebar.expander("🔍 STFT 分析设置", expanded=False):
    nperseg = st.selectbox("窗口大小 (nperseg)", [32, 64, 128, 256], index=1)
    noverlap = st.slider("重叠点数", 0, nperseg-1, nperseg//2)
    stft_max_freq = st.slider("最大显示频率", 50, 500, 500)
    use_log_scale = st.checkbox("对数刻度 (dB)", value=True)

# ================= 主界面 =================
st.title("⚡ EMG 信号精细化切分 & 分析")

if not (subject and date_str and target_label is not None):
    st.info("请在左侧选择完整数据路径。")
    st.stop()

search_path = os.path.join(data_root, subject, date_str, "RAW_EMG*.csv")
files = [f for f in glob.glob(search_path) if parse_filename(os.path.basename(f)) == target_label]

if not files:
    st.warning("无匹配文件")
    st.stop()

selected_file = st.selectbox("当前文件", files, format_func=lambda x: os.path.basename(x))
if submitted or 'filtered' not in locals():
    if selected_file:
        with st.spinner('正在处理...'):
            raw_data, _ = load_data(selected_file)
            
            # 确保通道不越界
            if view_ch >= raw_data.shape[1]:
                st.error(f"所选通道 CH{view_ch+1} 超出数据范围 (Max {raw_data.shape[1]})")
                st.stop()
                
            filtered, energy, threshold, raw_mask, final_mask = process_signal(
                raw_data, fs, band_range[0], band_range[1], 
                smooth_ms, merge_ms, thresh_ratio, 
                use_refine=use_refine_logic,
                use_notch=use_notch,      
                notch_freq=notch_freq    
            )

            temp_labeled, temp_num = ndimage.label(final_mask)
            
            if temp_num > 0:
                seg_energies = []
                # 1. 计算所有段的能量
                for i in range(1, temp_num + 1):
                    loc = np.where(temp_labeled == i)[0]
                    # 注意：filtered 是 (Samples, Channels)
                    seg_slice = filtered[loc[0]:loc[-1]] 
                    # 计算 RMS：先对 Time(axis=0) 平方平均开根，再对 Channels 平均
                    rms = np.mean(np.sqrt(np.mean(seg_slice**2, axis=0)))
                    seg_energies.append(rms)
                
                # 2. 过滤异常
                median_E = np.median(seg_energies)
                upper_limit = median_E * 5.0
                
                for i in range(1, temp_num + 1):
                    # 对应的能量索引是 i-1
                    if seg_energies[i-1] > upper_limit:
                        # 在 final_mask 中抹除该段
                        loc = np.where(temp_labeled == i)[0]
                        final_mask[loc] = False
            
            # 节奏过滤
            labeled_mask, num_features = ndimage.label(final_mask)
            if use_rhythm and num_features > 1:
                centers = []
                for i in range(1, num_features + 1):
                    idx = np.where(labeled_mask == i)[0]
                    centers.append((idx[0] + idx[-1]) / 2)
                
                expected_interval = 4000 * (fs / 1000) 
                min_gap = expected_interval * interval_ratio
                
                valid_ids = [1]
                last_center = centers[0]
                for i in range(1, num_features):
                    if (centers[i] - last_center) > min_gap:
                        valid_ids.append(i + 1)
                        last_center = centers[i]
                
                rhythm_mask = np.zeros_like(final_mask)
                for vid in valid_ids:
                    rhythm_mask[labeled_mask == vid] = True
                
                display_mask = rhythm_mask
                labeled_mask, num_display = ndimage.label(display_mask)
            else:
                display_mask = final_mask
                num_display = num_features

        # --- 图表 1: 宏观概览 ---
        st.subheader(f"📊 信号概览 (CH{view_ch+1}, 动作数: {num_display})")
        
        # 【新增】定义降采样步长，每10个点取1个
        step = 10 
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        
        # 注意：x轴 (t) 和 数据 (raw_data) 都要切片 [::step]
        t = np.arange(len(raw_data)) / fs
        t_down = t[::step]
        
        # 绘图时全都加上 [::step]
        ax1.plot(t_down, raw_data[::step, view_ch], color='lightgray', alpha=0.5, label=f'Raw CH{view_ch+1}')
        ax1.plot(t_down, filtered[::step, view_ch], color='#1f77b4', linewidth=1, label='Filtered')
        
        ax2.plot(t_down, energy[::step], color='orange', label='Global Energy')
        # axhline 不需要切片，因为它是水平直线
        ax2.axhline(threshold, color='red', linestyle='--', alpha=0.5)
        
        # fill_between 需要特别处理，因为它是填充区域
        # 如果用降采样可能导致边缘锯齿，但为了速度可以接受，或者保持原样（fill_between通常比plot快一点）
        # 这里建议也降采样
        ax2.fill_between(t_down, 0, np.max(energy), where=raw_mask[::step], color='lightgreen', alpha=0.3, label='Discarded Candidates')
        ax2.fill_between(t_down, 0, np.max(energy), where=display_mask[::step], color='green', alpha=0.6, label='Accepted Segments')
        
        ax2.legend(loc='upper right')
        ax2.set_xlabel('Time (s)')
        st.pyplot(fig)
        
        # --- 详细交互区 (恢复 STFT 和 波形放大) ---
        st.markdown("---")
        st.subheader("🔍 动作切片详情")
        
        if num_display > 0:
            # 滑块选择特定动作
            seg_id = st.slider("选择要分析的动作片段 ID", 1, num_display, 1)
            
            # 提取数据
            indices = np.where(labeled_mask == seg_id)[0]
            start, end = indices[0], indices[-1]
            
            # 增加一点前后余量以便观察
            margin = int(0.05 * fs)
            plot_start = max(0, start - margin)
            plot_end = min(len(filtered), end + margin)
            
            seg_data = filtered[plot_start:plot_end, view_ch]
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown(f"**时域波形 (ID: {seg_id})**")
                fig_seg, ax_seg = plt.subplots(figsize=(6, 4))
                ax_seg.plot(np.arange(len(seg_data)), seg_data, color='#1f77b4')
                # 标出实际被选中的部分（去除余量）
                ax_seg.axvspan(start - plot_start, end - plot_start, color='green', alpha=0.2, label='Active Region')
                ax_seg.set_title(f"Segment #{seg_id} (CH{view_ch+1})")
                ax_seg.legend()
                st.pyplot(fig_seg)
                
            with c2:
                st.markdown(f"**时频图 (STFT)**")
                f_stft, t_stft, Zxx = signal.stft(
                    seg_data, 
                    fs=fs, 
                    nperseg=nperseg, 
                    noverlap=noverlap
                )
                magnitude = np.abs(Zxx)
                if use_log_scale:
                    magnitude = 20 * np.log10(magnitude + 1e-6)
                
                fig_stft, ax_stft = plt.subplots(figsize=(6, 4))
                pcm = ax_stft.pcolormesh(t_stft, f_stft, magnitude, shading='gouraud', cmap='jet')
                ax_stft.set_ylim(0, stft_max_freq)
                ax_stft.set_ylabel('Freq (Hz)')
                ax_stft.set_xlabel('Time (s)')
                fig_stft.colorbar(pcm, ax=ax_stft, label='dB' if use_log_scale else 'Amp')
                st.pyplot(fig_stft)
        
        # --- 画廊模式 (恢复) ---
        st.markdown("---")
        st.subheader("🖼️ 所有切片缩略图 (Gallery)")
        
        show_gallery = st.checkbox("展开查看所有切片", value=True)
        
        if show_gallery and num_display > 0:
            cols_count = st.slider("每行显示数量", 3, 10, 6)
            
            # 准备切片数据
            slices = []
            for i in range(1, num_display + 1):
                loc = np.where(labeled_mask == i)[0]
                slices.append(filtered[loc[0]:loc[-1], view_ch])
                
            for i in range(0, num_display, cols_count):
                cols = st.columns(cols_count)
                for j in range(cols_count):
                    idx = i + j
                    if idx < num_display:
                        with cols[j]:
                            fig_t, ax_t = plt.subplots(figsize=(3, 2))
                            ax_t.plot(slices[idx], lw=1, color='#1f77b4')
                            ax_t.set_title(f"#{idx+1}", fontsize=8)
                            ax_t.axis('off')
                            st.pyplot(fig_t)
                            plt.close(fig_t)