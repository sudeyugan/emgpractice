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

# ================= 侧边栏：控制面板 =================
st.sidebar.header("📂 1. 数据筛选")
data_root = st.sidebar.text_input("数据根目录", "data")
subject = st.sidebar.text_input("测试者姓名 (Subject)", "charles")
date_str = st.sidebar.text_input("日期 (Date)", "20250213")
target_label = st.sidebar.number_input("动作标签 (DF Label)", min_value=0, value=1, step=1)

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
        
        # 运行算法
        filtered_data, energy, threshold, mask = process_signal(
            raw_data, fs, band_range[0], band_range[1], 
            smooth_ms, merge_ms, thresh_ratio
        )
        
        # 计算切分统计
        labeled_mask, num_segments = ndimage.label(mask)
        
    # ================= 绘图区域 =================
    
    # 图表 1: 信号全览
    st.subheader(f"📊 信号概览 (检测到 {num_segments} 个动作)")
    
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
    
    # ================= 详细切片展示 =================
    st.subheader("🔍 动作切片详情")
    
    if num_segments > 0:
        # 让用户选择查看第几个切片
        seg_id = st.slider("查看第几个动作片段?", 1, num_segments, 1)
        
        indices = np.where(labeled_mask == seg_id)[0]
        start, end = indices[0], indices[-1]
        duration_ms = (end - start) / fs * 1000
        
        st.write(f"**片段 #{seg_id}**: 时间 {start/fs:.2f}s - {end/fs:.2f}s (持续 {duration_ms:.0f} ms)")
        
        # 画出这个具体的切片
        fig_seg, ax_seg = plt.subplots(figsize=(10, 4))
        seg_data = filtered_data[start:end]
        ax_seg.plot(seg_data)
        ax_seg.set_title(f"Segment #{seg_id} (All 5 Channels)")
        st.pyplot(fig_seg)
        
        if duration_ms < 300:
            st.error("⚠️ 警告：该片段过短，批量处理时可能会被丢弃！")
    else:
        st.error("未检测到任何动作！请尝试调低阈值系数。")