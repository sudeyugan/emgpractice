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

# ================= 核心逻辑 =================
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

def get_rhythm_mask(energy, fs, interval_ms=4000, window_ms=300, noise_cv_threshold=0.2):
    """
    [New] 4s 固定节奏峰值提取逻辑
    改进：使用相位投票 (Phase Voting) 确定锚点，抗干扰能力更强。
    """
    mask = np.zeros_like(energy, dtype=bool)
    
    # 1. 寻找所有候选峰
    # 稍微放宽一点限制，以便捕捉尽可能多的动作进行投票
    min_dist = int(2.0 * fs) 
    noise_floor = np.percentile(energy, 10)
    peaks, _ = signal.find_peaks(energy, distance=min_dist, height=noise_floor * 1.5)
    
    if len(peaks) == 0:
        return mask
    
    # 2. 确定锚点 (Anchor) - 智能相位投票
    # ---------------------------------------------------------
    # 目的：找到大多数峰值遵循的节奏相位，忽略偶尔的高能噪音(如手腕翻转)
    interval_samples = int((interval_ms / 1000) * fs)
    if interval_samples < 1: interval_samples = 1

    # (1) 计算所有峰相对于 4s 的相位偏移 (0 ~ interval)
    phases = peaks % interval_samples
    
    # (2) 直方图统计：看峰值主要集中在哪里
    # 设定 200ms 的宽容度 (bin_width)，足以容忍人手的轻微节奏误差
    bin_width = int(0.2 * fs) 
    bins = np.arange(0, interval_samples + bin_width, bin_width)
    counts, bin_edges = np.histogram(phases, bins=bins)
    
    # (3) 找到众数区间 (Most Common Phase)
    best_bin_idx = np.argmax(counts)
    phase_start = bin_edges[best_bin_idx]
    phase_end = bin_edges[best_bin_idx+1]
    
    # (4) 筛选出“合群”的峰 (On-beat peaks)
    # 处理循环边界情况 (如果相位刚好在 0 或 4s 附近)
    # 这里简单处理：直接取落在最佳 bin 里的峰
    candidates_mask = (phases >= phase_start) & (phases < phase_end)
    candidates = peaks[candidates_mask]
    
    # (5) 确定最终 Anchor
    if len(candidates) > 0:
        # 在符合节奏的峰里，选能量最大的作为锚点
        # 这样既避开了不符合节奏的超大噪音，又保证了对齐的信噪比
        best_sub_idx = np.argmax(energy[candidates])
        anchor_peak = candidates[best_sub_idx]
    else:
        # 降级：如果实在找不到规律，就回退到取第一个峰 (First Peak)
        # 用户之前建议的方法
        anchor_peak = peaks[0]
    # ---------------------------------------------------------

    half_win = int((window_ms / 1000) * fs) // 2
    search_radius = int(1.0 * fs)
    
    # 3. 生成网格并搜索 (逻辑保持不变，但 anchor_peak 更加可靠了)
    valid_centers = []
    max_len = len(energy)
    
    # 向后 (Forward)
    curr_grid = anchor_peak
    while curr_grid < max_len:
        s_start = max(0, curr_grid - search_radius)
        s_end = min(max_len, curr_grid + search_radius)
        region = energy[s_start:s_end]
        if len(region) > 0:
            local_max_idx = np.argmax(region)
            abs_center = s_start + local_max_idx
            if energy[abs_center] > noise_floor * 1.5:
                valid_centers.append(abs_center)
        curr_grid += interval_samples
        
    # 向前 (Backward)
    curr_grid = anchor_peak - interval_samples
    while curr_grid > -search_radius:
        s_start = max(0, curr_grid - search_radius)
        s_end = min(max_len, curr_grid + search_radius)
        region = energy[s_start:s_end]
        if len(region) > 0:
            local_max_idx = np.argmax(region)
            abs_center = s_start + local_max_idx
            if energy[abs_center] > noise_floor * 1.5:
                valid_centers.append(abs_center)
        curr_grid -= interval_samples

    valid_centers = sorted(list(set(valid_centers)))
    
    # 4. 生成 Mask (含 CV 噪音过滤)
    for c in valid_centers:
        s = max(0, c - half_win)
        e = min(max_len, c + half_win)
        
        seg_vals = energy[s:e]
        mean_e = np.mean(seg_vals)
        std_e = np.std(seg_vals)
        cv = std_e / (mean_e + 1e-6)
        
        ref_energy = energy[anchor_peak]
        # 如果能量很大但 CV 很小 (均匀噪音)，跳过
        if mean_e > ref_energy * 0.3 and cv < noise_cv_threshold:
             continue
             
        mask[s:e] = True
        
    return mask

def refine_mask_logic(mask, fs, energy=None):
    """(VAD 模式专用) 优化后的掩码逻辑"""
    labeled, num = ndimage.label(mask)
    new_mask = np.zeros_like(mask, dtype=bool)
    noise_ban_mask = np.zeros_like(mask, dtype=bool)
    
    samples_1s = int(1.0 * fs)
    samples_500ms = int(0.5 * fs)
    structure_len = int(0.4 * fs)
    
    for i in range(1, num + 1):
        loc = np.where(labeled == i)[0]
        if len(loc) == 0: continue
        
        duration_ms = (len(loc) / fs) * 1000
        
        if duration_ms > 5000:
            is_noise = False
            if energy is not None:
                seg_energy = energy[loc]
                mean_e = np.mean(seg_energy)
                std_e = np.std(seg_energy)
                cv = std_e / (mean_e + 1e-6)
                if cv < 0.2: 
                    is_noise = True
                    ban_start = max(0, loc[0] - samples_1s)
                    ban_end = min(len(mask), loc[-1] + samples_1s)
                    noise_ban_mask[ban_start:ban_end] = True
            
            if is_noise: continue

            seg_mask = np.zeros_like(mask)
            seg_mask[loc] = True
            structure = np.ones(structure_len) 
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
            
        elif 2000 < duration_ms <= 5000:
            continue
        elif 500 < duration_ms <= 2000:
            center = int(np.mean(loc))
            half = samples_500ms // 2
            start = max(0, center - half)
            end = min(len(mask), center + half)
            new_mask[start:end] = True
        else:
            new_mask[loc] = True
            
    new_mask[noise_ban_mask] = False
    return new_mask

def apply_rhythm_filter(mask, fs, interval_ms, ratio=0.8):
    """
    (VAD 模式专用) 节奏过滤：
    如果当前动作距离上一个动作小于 interval_ms * ratio，则视为干扰并剔除。
    """
    labeled, num = ndimage.label(mask)
    if num < 2: return mask
    
    # 1. 计算所有片段的中心点
    centers = []
    for i in range(1, num + 1):
        loc = np.where(labeled == i)[0]
        centers.append(np.mean(loc)) # 使用平均位置作为中心
    
    # 2. 最小允许间距 (例如 4s * 0.8 = 3.2s)
    min_gap_samples = (interval_ms / 1000) * fs * ratio
    
    # 3. 过滤逻辑
    valid_indices = [1] # 默认保留第一个动作
    last_center = centers[0]
    
    for i in range(1, num):
        curr_center = centers[i]
        # 只有当 距离 > 最小间距 时，才保留
        if (curr_center - last_center) > min_gap_samples:
            valid_indices.append(i + 1)
            last_center = curr_center
        # 否则跳过 (视为过密的干扰)
    
    # 4. 重建 Mask
    new_mask = np.zeros_like(mask)
    for idx in valid_indices:
        new_mask[labeled == idx] = True
        
    return new_mask

@st.cache_data
def process_signal(data, fs, low_cut, high_cut, 
                   mode='VAD',                  
                   # VAD params
                   smooth_ms=100, merge_ms=200, threshold_ratio=0.15, use_refine=True, 
                   # Peak params
                   rhythm_interval=4000, rhythm_window=300, noise_cv=0.2,
                   # Common
                   use_notch=True, notch_freq=50):
    
    # 1. 工频陷波
    if use_notch:
        b_notch, a_notch = signal.iirnotch(notch_freq, 30, fs)
        data = signal.filtfilt(b_notch, a_notch, data, axis=0)
    
    # 2. 带通滤波
    b, a = signal.butter(4, [low_cut, high_cut], btype='bandpass', fs=fs)
    filtered = signal.filtfilt(b, a, data, axis=0)
    
    # 3. 能量计算
    energy = np.sqrt(np.mean(filtered**2, axis=1))
    
    # 平滑
    win_len = int((smooth_ms/1000) * fs)
    if win_len < 1: win_len = 1
    energy_smooth = np.convolve(energy, np.ones(win_len)/win_len, mode='same')
    
    noise_floor = np.percentile(energy_smooth, 10)
    peak_level = np.percentile(energy_smooth, 99)
    threshold = noise_floor + threshold_ratio * (peak_level - noise_floor)
    
    raw_mask = np.zeros_like(energy, dtype=bool)
    final_mask = np.zeros_like(energy, dtype=bool)

    if mode == 'VAD':
        raw_mask = energy_smooth > threshold
        gap_samples = int((merge_ms/1000) * fs)
        if gap_samples > 0:
            raw_mask = ndimage.binary_closing(raw_mask, structure=np.ones(gap_samples))
        if use_refine:
            final_mask = refine_mask_logic(raw_mask, fs, energy=energy_smooth)
        else:
            final_mask = raw_mask

        if rhythm_interval > 0:
            final_mask = apply_rhythm_filter(final_mask, fs, rhythm_interval, ratio=0.9)
            
    elif mode == 'PEAK':
        final_mask = get_rhythm_mask(energy_smooth, fs, 
                                     interval_ms=rhythm_interval, 
                                     window_ms=rhythm_window,
                                     noise_cv_threshold=noise_cv)
        raw_mask = final_mask 
        
    return filtered, energy_smooth, threshold, raw_mask, final_mask

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

st.sidebar.markdown("---")
view_ch = st.sidebar.number_input("👁️ 可视化通道 (CH)", 1, 8, 1) - 1 

st.sidebar.markdown("---")
st.sidebar.header("🎛️ 算法参数")

with st.sidebar.form("analysis_config"):
    mode_choice = st.radio("分割模式 (Segmentation Mode)", 
                           ["能量阈值检测 (VAD)", "固定节奏峰值 (Peak 4s)"])
    
    st.markdown("### 基础设置")
    fs = st.number_input("采样率 (Hz)", value=1000)
    band_range = st.slider("带通滤波 (Hz)", 0, 500, (20, 450))
    use_notch = st.checkbox("🔌 启用工频陷波 (Notch)", value=True)
    
    if mode_choice == "能量阈值检测 (VAD)":
        st.markdown("### VAD 参数")
        thresh_ratio = st.slider("阈值系数", 0.05, 0.50, 0.15, 0.01)
        smooth_ms = st.slider("平滑窗口 (ms)", 10, 500, 100, 10)
        merge_ms = st.slider("合并间隙 (ms)", 0, 1000, 200, 50)
        use_refine_logic = st.checkbox("启用时长门控 (1s/500ms)", value=True)
        rhythm_int = 4000
        rhythm_win = 300
    else: 
        st.markdown("### 峰值提取参数")
        rhythm_int = st.number_input("动作间隔 (ms)", value=4000, step=100)
        rhythm_win = st.number_input("截取窗口 (ms)", value=300, step=50)
        st.caption("👇 平滑参数仍会影响峰值寻找")
        smooth_ms = st.slider("平滑窗口 (ms)", 10, 500, 100, 10)
        thresh_ratio = 0.15
        merge_ms = 0
        use_refine_logic = False

    submitted = st.form_submit_button("🚀 重新分析")

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
            
            if view_ch >= raw_data.shape[1]:
                st.error(f"所选通道 CH{view_ch+1} 超出数据范围")
                st.stop()
            
            mode_code = 'PEAK' if "Peak" in mode_choice else 'VAD'
            
            filtered, energy, threshold, raw_mask, final_mask = process_signal(
                raw_data, fs, band_range[0], band_range[1], 
                mode=mode_code,
                smooth_ms=smooth_ms, merge_ms=merge_ms, threshold_ratio=thresh_ratio, 
                use_refine=use_refine_logic,
                rhythm_interval=rhythm_int, rhythm_window=rhythm_win, noise_cv=0.2,
                use_notch=use_notch
            )

            labeled_mask, num_display = ndimage.label(final_mask)
            
            if mode_code == 'VAD' and num_display > 0:
                seg_energies = []
                for i in range(1, num_display + 1):
                    loc = np.where(labeled_mask == i)[0]
                    seg_slice = filtered[loc[0]:loc[-1]] 
                    rms = np.mean(np.sqrt(np.mean(seg_slice**2, axis=0)))
                    seg_energies.append(rms)
                median_E = np.median(seg_energies)
                upper_limit = median_E * 5.0
                for i in range(1, num_display + 1):
                    if seg_energies[i-1] > upper_limit:
                        final_mask[np.where(labeled_mask == i)[0]] = False
                labeled_mask, num_display = ndimage.label(final_mask)

        # --- 图表 1: 宏观概览 ---
        st.subheader(f"📊 信号概览 (CH{view_ch+1}, 动作数: {num_display})")
        
        step = 10 
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        
        t = np.arange(len(raw_data)) / fs
        t_down = t[::step]
        
        ax1.plot(t_down, raw_data[::step, view_ch], color='lightgray', alpha=0.5, label='Raw')
        ax1.plot(t_down, filtered[::step, view_ch], color='#1f77b4', linewidth=1, label='Filtered')
        ax2.plot(t_down, energy[::step], color='orange', label='Energy')
        
        if mode_code == 'VAD':
            ax2.axhline(threshold, color='red', linestyle='--', alpha=0.5, label='Threshold')
        
        ax2.fill_between(t_down, 0, np.max(energy), where=final_mask[::step], color='green', alpha=0.5, label='Selected')

        if mode_code == 'VAD':
            raw_labeled, raw_num = ndimage.label(raw_mask)
            for i in range(1, raw_num + 1):
                loc = np.where(raw_labeled == i)[0]
                if len(loc) == 0: continue
                is_accepted = np.any(final_mask[loc])
                duration_ms = (len(loc) / fs) * 1000
                if not is_accepted and duration_ms > 50:
                    t_start, t_end = loc[0] / fs, loc[-1] / fs
                    ax2.axvspan(t_start, t_end, color='red', alpha=0.1)
        
        if mode_code == 'PEAK' and num_display > 0:
            # 找到 Anchor (这里简单反推一下用于画参考线)
            # 因为 get_rhythm_mask 内部已经用投票算好了，外部不知道 anchor 在哪
            # 但我们可以根据 labeled_mask 的第一个中心简单画一个网格示意
            first_idx = np.where(labeled_mask == 1)[0]
            if len(first_idx) > 0:
                anchor = (first_idx[0] + first_idx[-1]) / 2 / fs
                for k in range(-20, 20):
                    g = anchor + k * (rhythm_int/1000)
                    if 0 <= g <= t[-1]:
                        ax2.axvline(g, color='gray', linestyle=':', alpha=0.3)

        ax2.legend(loc='upper right', fontsize='small')
        ax2.set_xlabel('Time (s)')
        st.pyplot(fig)  
        plt.close(fig)

        # --- 详细交互区 ---
        st.markdown("---")
        st.subheader("🔍 动作切片详情")
        
        if num_display > 0:
            seg_id = st.slider("选择要分析的动作片段 ID", 1, num_display, 1)
            indices = np.where(labeled_mask == seg_id)[0]
            start, end = indices[0], indices[-1]
            
            margin = int(0.05 * fs)
            plot_start = max(0, start - margin)
            plot_end = min(len(filtered), end + margin)
            seg_data = filtered[plot_start:plot_end, view_ch]
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**时域波形 (ID: {seg_id})**")
                fig_seg, ax_seg = plt.subplots(figsize=(6, 4))
                ax_seg.plot(np.arange(len(seg_data)), seg_data, color='#1f77b4')
                ax_seg.axvspan(start - plot_start, end - plot_start, color='green', alpha=0.2)
                ax_seg.set_title(f"Segment #{seg_id} (CH{view_ch+1})")
                st.pyplot(fig_seg)
                
            with c2:
                st.markdown(f"**时频图 (STFT)**")
                f_stft, t_stft, Zxx = signal.stft(seg_data, fs=fs, nperseg=nperseg, noverlap=noverlap)
                magnitude = np.abs(Zxx)
                if use_log_scale: magnitude = 20 * np.log10(magnitude + 1e-6)
                fig_stft, ax_stft = plt.subplots(figsize=(6, 4))
                pcm = ax_stft.pcolormesh(t_stft, f_stft, magnitude, shading='gouraud', cmap='jet')
                ax_stft.set_ylim(0, stft_max_freq)
                fig_stft.colorbar(pcm, ax=ax_stft)
                st.pyplot(fig_stft)
        
        # --- 画廊模式 ---
        st.markdown("---")
        st.subheader("🖼️ 所有切片缩略图 (Gallery)")
        show_gallery = st.checkbox("展开查看所有切片", value=True)
        
        if show_gallery and num_display > 0:
            cols_count = st.slider("每行显示数量", 3, 10, 6)
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