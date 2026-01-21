import pandas as pd
import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
import os
import re

# ================= 配置参数 (已同步 GUI 设置) =================
FS = 1000                   # 采样率 1000Hz
WINDOW_MS = 250             # 窗口 250ms
WINDOW_SIZE = int(FS * (WINDOW_MS / 1000))

# --- VAD (活动检测) 参数 ---
VAD_SMOOTH_MS = 100         # 平滑窗口 (GUI: 100ms)
VAD_MERGE_GAP_MS = 200      # 合并间隙 (GUI: 200ms)
THRESHOLD_RATIO = 0.15      # 阈值系数 (GUI: 0.15)

# --- 过滤逻辑参数 ---
ENABLE_NOTCH = True         # 启用工频陷波
NOTCH_FREQ = 50             # 干扰频率 50Hz
ENABLE_REFINE = True        # 启用时长门控 (1s/500ms 逻辑)
ENABLE_RHYTHM = True        # 启用节奏过滤
EXPECTED_INTERVAL_MS = 4000 # 节奏基准 4秒
INTERVAL_RATIO = 0.90       # 最小间距比例 (GUI: 0.90)

def parse_filename_info(filepath):
    """解析文件名，返回 (Subject, Date, Label, Timestamp)"""
    filename = os.path.basename(filepath)
    parts = filepath.split(os.sep)
    subject = parts[-3] if len(parts) >= 3 else "Unknown"
    date = parts[-2] if len(parts) >= 3 else "Unknown"
    
    label_match = re.search(r'DF(\d+)\.', filename)
    label = int(label_match.group(1)) if label_match else None
    
    return subject, date, label, filename

def get_rhythm_mask(energy, fs, interval_ms=4000, window_ms=300, noise_cv_threshold=0.2):
    """
    [移植自 new_auto_train.py] 4s 固定节奏峰值提取逻辑 + 相位投票
    """
    mask = np.zeros_like(energy, dtype=bool)
    
    # 1. 寻找候选峰 (低阈值，先尽可能多抓)
    min_dist = int(2.0 * fs) 
    noise_floor = np.percentile(energy, 10)
    # 这里的 1.5 倍底噪是一个比较宽容的初筛
    peaks, _ = signal.find_peaks(energy, distance=min_dist, height=noise_floor * 1.5)
    
    if len(peaks) == 0:
        return mask
    
    # 2. 相位投票 (Phase Voting) 确定锚点
    interval_samples = int((interval_ms / 1000) * fs)
    if interval_samples < 1: interval_samples = 1

    phases = peaks % interval_samples
    
    bin_width = int(0.2 * fs) # 200ms 容差
    bins = np.arange(0, interval_samples + bin_width, bin_width)
    counts, bin_edges = np.histogram(phases, bins=bins)
    
    if len(counts) == 0: return mask

    best_bin_idx = np.argmax(counts)
    phase_start = bin_edges[best_bin_idx]
    phase_end = bin_edges[best_bin_idx+1]
    
    # 筛选 On-beat peaks
    candidates_mask = (phases >= phase_start) & (phases < phase_end)
    candidates = peaks[candidates_mask]
    
    if len(candidates) > 0:
        # 选能量最大的合群峰作为 Anchor
        best_sub_idx = np.argmax(energy[candidates])
        anchor_peak = candidates[best_sub_idx]
    else:
        anchor_peak = peaks[0]

    # 3. 生成网格并搜索
    half_win = int((window_ms / 1000) * fs) // 2
    search_radius = int(1.0 * fs) # 在网格点前后 1s 搜索
    valid_centers = []
    max_len = len(energy)
    
    # Forward & Backward Search
    for direction in [1, -1]:
        curr_grid = anchor_peak if direction == 1 else anchor_peak - interval_samples
        
        while 0 <= curr_grid < max_len:
            s_start = max(0, curr_grid - search_radius)
            s_end = min(max_len, curr_grid + search_radius)
            region = energy[s_start:s_end]
            
            if len(region) > 0:
                local_max_idx = np.argmax(region)
                abs_center = s_start + local_max_idx
                # 再次校验峰值强度，防止提取到纯底噪 (1.2倍底噪)
                if energy[abs_center] > noise_floor * 1.2:
                    valid_centers.append(abs_center)
            
            if direction == 1: curr_grid += interval_samples
            else: curr_grid -= interval_samples

    valid_centers = sorted(list(set(valid_centers)))
    
    # 4. 生成 Mask (CV 过滤持续噪音)
    for c in valid_centers:
        s = max(0, c - half_win)
        e = min(max_len, c + half_win)
        
        seg_vals = energy[s:e]
        if len(seg_vals) == 0: continue

        mean_e = np.mean(seg_vals)
        std_e = np.std(seg_vals)
        cv = std_e / (mean_e + 1e-6)
        
        ref_energy = energy[anchor_peak]
        # 如果能量很大但 CV 很小 (平稳噪音)，剔除
        if mean_e > ref_energy * 0.3 and cv < noise_cv_threshold:
             continue
             
        mask[s:e] = True
        
    return mask

def refine_mask_logic(mask, fs, energy=None):
    """
    优化后的掩码逻辑：
    1. 识别并屏蔽持续噪音（及其前后1s区域）。
    2. 处理粘连的长动作。
    3. 过滤过短的碎片。
    """
    labeled, num = ndimage.label(mask)
    new_mask = np.zeros_like(mask, dtype=bool)
    
    # [NEW] 1. 定义一个“噪音屏蔽罩”，初始化为全 False
    noise_ban_mask = np.zeros_like(mask, dtype=bool)
    
    # 常用时间常数
    samples_1s = int(1.0 * fs)      # 1秒对应的点数 (用于屏蔽)
    samples_500ms = int(0.5 * fs)   # 500ms (用于截取)
    structure_len = int(0.4 * fs)   # 400ms (用于切断粘连)
    
    for i in range(1, num + 1):
        loc = np.where(labeled == i)[0]
        if len(loc) == 0: continue
        
        duration_ms = (len(loc) / fs) * 1000
        
        # --- A. 处理长片段 (>5s) ---
        if duration_ms > 5000:
            is_noise = False
            
            # [噪音检测] CV 变异系数逻辑
            if energy is not None:
                seg_energy = energy[loc]
                mean_e = np.mean(seg_energy)
                std_e = np.std(seg_energy)
                cv = std_e / (mean_e + 1e-6)
                
                # 如果是平稳噪音 (CV < 0.2)
                if cv < 0.2: 
                    is_noise = True
                    # 核心逻辑：建立噪音禁区
                    # 将该片段的范围，以及前后 1s 的范围，都在 ban_mask 中标记为 True
                    ban_start = max(0, loc[0] - samples_1s)
                    ban_end = min(len(mask), loc[-1] + samples_1s)
                    noise_ban_mask[ban_start:ban_end] = True
            
            # 如果确认是噪音，直接跳过处理（不用往 new_mask 里加东西了）
            if is_noise:
                continue

            # [粘连处理] 如果不是噪音，但太长，说明是粘连动作 -> 尝试切开
            seg_mask = np.zeros_like(mask)
            seg_mask[loc] = True
            
            structure = np.ones(structure_len) 
            opened_mask = ndimage.binary_opening(seg_mask, structure=structure)
            sub_labeled, sub_num = ndimage.label(opened_mask)
            
            for j in range(1, sub_num + 1):
                sub_loc = np.where(sub_labeled == j)[0]
                sub_dur = (len(sub_loc) / fs) * 1000
                
                # 子片段长度检查
                if sub_dur <= 1000:
                    if 500 < sub_dur <= 1000: # 取中间
                        center = int(np.mean(sub_loc))
                        half = samples_500ms // 2
                        s = max(0, center - half)
                        e = min(len(mask), center + half)
                        new_mask[s:e] = True
                    else: # < 500ms 的子片段通常是切分产生的合法短动作
                        new_mask[sub_loc] = True
            
        # --- B. 处理中等片段 (被丢弃) ---
        elif 1000 < duration_ms <= 5000:
            continue

        # --- C. 处理短片段 (500ms ~ 1s) -> 取中间 ---
        elif 500 < duration_ms <= 1000:
            center = int(np.mean(loc))
            half = samples_500ms // 2
            start = max(0, center - half)
            end = min(len(mask), center + half)
            new_mask[start:end] = True
            
        # --- D. 处理极短片段 (<= 500ms) -> 保留 ---
        else:
            new_mask[loc] = True
            
    # 2. 最终过滤：应用噪音屏蔽罩
    # 任何落在“禁区”内的有效信号（new_mask 为 True 的点），如果 noise_ban_mask 也是 True，就被强制置为 False
    # 也就是：new_mask = new_mask AND (NOT noise_ban_mask)
    new_mask[noise_ban_mask] = False
    
    return new_mask

def time_warp(data, sigma=0.2, knot=4):
    """时间扭曲增强"""
    orig_steps = np.arange(data.shape[0])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, data.shape[1]))
    ret = np.zeros_like(data)
    for i in range(data.shape[1]):
        time_warp = np.interp(orig_steps, np.linspace(0, data.shape[0]-1., num=knot+2), random_warps[:, i])
        cum_warp = np.cumsum(time_warp)
        scale = (data.shape[0]-1) / cum_warp[-1]
        new_times = cum_warp * scale
        ret[:, i] = np.interp(orig_steps, new_times, data[:, i])
    return ret

def time_shift(data, shift_limit=0.1):
    """时间平移增强"""
    shift_amt = int(data.shape[0] * shift_limit * np.random.uniform(-1, 1))
    return np.roll(data, shift_amt, axis=0)

def channel_mask(data, mask_prob=0.15):
    """通道遮挡增强"""
    temp = data.copy()
    if np.random.random() < mask_prob:
        c = np.random.randint(0, data.shape[1])
        temp[:, c] = 0
    return temp

def add_noise(data, noise_level=0.01):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def scale_amplitude(data, scale_range=(0.8, 1.2)):
    factor = np.random.uniform(scale_range[0], scale_range[1])
    return data * factor


def process_selected_files(file_list, progress_callback=None, stride_ms=100, augment_config=None, segmentation_config=None):
    """
    核心处理流程：支持 VAD (阈值) 和 Peak (固定节奏) 两种模式
    """
    if augment_config is None: augment_config = {}
    
    # === 1. 解析分割配置 ===
    # 默认为 VAD 模式以保持兼容性
    if segmentation_config is None:
        segmentation_config = {'method': 'vad'} 
    
    method = segmentation_config.get('method', 'vad')
    
    # Peak 模式特有参数
    rhythm_ms = segmentation_config.get('rhythm_ms', 4000)      # 节奏间隔
    peak_win_ms = segmentation_config.get('peak_win_ms', 350)  # 峰值左右截取的窗口大小
    
    # 增强配置
    multiplier = augment_config.get('multiplier', 1)
    enable_rest = augment_config.get('enable_rest', True)
    # ... (其他增强参数获取保持不变) ...
    enable_warp = augment_config.get('enable_warp', False)
    enable_shift = augment_config.get('enable_shift', False)
    enable_mask = augment_config.get('enable_mask', False)
    enable_noise = augment_config.get('enable_noise', False)
    enable_scaling = augment_config.get('enable_scaling', False)

    # 自动计算静息样本比例
    rest_ratio_per_file = 0
    if enable_rest:
        unique_labels = set()
        for f in file_list:
            try:
                _, _, label, _ = parse_filename_info(f)
                if label is not None: unique_labels.add(label)
            except: pass
        num_act_classes = len(unique_labels) if len(unique_labels) > 0 else 1
        rest_ratio_per_file = 1.1 / num_act_classes
    
    X_list = []
    y_list = []
    groups_list = [] 
    
    current_stride_size = int(FS * (stride_ms / 1000))
    if current_stride_size < 1: current_stride_size = 1
    
    total = len(file_list)
    
    for idx, f in enumerate(file_list):
        if progress_callback:
            progress_callback(idx / total, f"Processing: {os.path.basename(f)}")
            
        try:
            _, _, label, _ = parse_filename_info(f)
            if label is None: continue
            
            df = pd.read_csv(f)
            cols = [c for c in df.columns if 'CH' in c]
            raw_data = df[cols].values
            
            # --- 1. CH5 信号增益修正 ---
            if raw_data.shape[1] >= 5:
                raw_data[:, 4] = raw_data[:, 4] * 2.5
                
            # --- 2. 信号滤波链 ---
            data_proc = raw_data.copy()
            if ENABLE_NOTCH:
                b_notch, a_notch = signal.iirnotch(NOTCH_FREQ, 30, FS)
                data_proc = signal.filtfilt(b_notch, a_notch, data_proc, axis=0)
            
            b, a = signal.butter(4, [20, 450], btype='bandpass', fs=FS)
            data_clean = signal.filtfilt(b, a, data_proc, axis=0)
            
            # 计算能量
            energy = np.sqrt(np.mean(data_clean**2, axis=1))
            win_len = int((VAD_SMOOTH_MS/1000) * FS)
            energy_smooth = np.convolve(energy, np.ones(win_len)/win_len, mode='same')
            
            # 基础阈值 (VAD和Peak模式都需要用到阈值来过滤明显的噪音区)
            noise_floor = np.percentile(energy_smooth, 10)
            peak_level = np.percentile(energy_smooth, 99)
            threshold = noise_floor + THRESHOLD_RATIO * (peak_level - noise_floor)

            final_segments = []
            
            # ================= 分支逻辑开始 =================
            
            if method == 'peak':
                # >>>>> 模式 A: 固定节奏峰值分割 (Peak Segmentation) <<<<<
                
                # 注意：GUI 传入的 rhythm_ms 是间隔，peak_win_ms 是窗口
                rhythm_mask = get_rhythm_mask(
                    energy_smooth, FS, 
                    interval_ms=rhythm_ms, 
                    window_ms=peak_win_ms,
                    noise_cv_threshold=0.2
                )
                
                # 将 Mask 转换为 segments 列表，以便后续代码复用
                labeled, num_seg = ndimage.label(rhythm_mask)
                
                for i in range(1, num_seg + 1):
                    loc = np.where(labeled == i)[0]
                    # 确保片段长度足够
                    if len(loc) > int(FS * 0.05): 
                        final_segments.append({
                            'start': loc[0],
                            'end': loc[-1],
                            'center': int(np.mean(loc))
                        })
                
                # 更新 raw_mask (用于后续提取 Rest)
                raw_mask = rhythm_mask
                    
            else:
                # >>>>> 模式 B: 能量阈值 VAD (原逻辑) <<<<<
                
                raw_mask = energy_smooth > threshold
                gap_samples = int((VAD_MERGE_GAP_MS/1000) * FS)
                raw_mask = ndimage.binary_closing(raw_mask, structure=np.ones(gap_samples))
                
                if ENABLE_REFINE:
                    final_mask = refine_mask_logic(raw_mask, FS, energy=energy_smooth)
                else:
                    final_mask = raw_mask
                    
                # 提取候选片段
                labeled_mask, num_features = ndimage.label(final_mask)
                candidate_segments = []
                for i in range(1, num_features + 1):
                    indices = np.where(labeled_mask == i)[0]
                    if len(indices) > int(0.05 * FS): # 最小长度保护
                        candidate_segments.append({
                            'start': indices[0],
                            'end': indices[-1],
                            'center': (indices[0] + indices[-1]) / 2
                        })
                
                # 能量强度过滤 (剔除微弱误触)
                if len(candidate_segments) > 0:
                    segment_energies = []
                    for seg in candidate_segments:
                        seg_data = data_clean[seg['start']:seg['end']]
                        rms = np.mean(np.sqrt(np.mean(seg_data**2, axis=0)))
                        segment_energies.append(rms)
                    median_energy = np.median(segment_energies)
                    energy_threshold = median_energy * 5.0
                    filtered_candidates = [seg for i, seg in enumerate(candidate_segments) if segment_energies[i] < energy_threshold]
                    # ==========================================
                    candidate_segments = filtered_candidates

                # 节奏过滤 (VAD 模式下的 Rhythm Filter)
                if ENABLE_RHYTHM and len(candidate_segments) > 1:
                    expected_interval = EXPECTED_INTERVAL_MS * (FS / 1000)
                    min_gap = expected_interval * INTERVAL_RATIO
                    
                    final_segments = []
                    if candidate_segments:
                        final_segments.append(candidate_segments[0])
                        last_center = candidate_segments[0]['center']
                        for i in range(1, len(candidate_segments)):
                            curr_center = candidate_segments[i]['center']
                            if (curr_center - last_center) > min_gap:
                                final_segments.append(candidate_segments[i])
                                last_center = curr_center
                else:
                    final_segments = candidate_segments

            # ================= 共同切片逻辑 =================
            
            action_samples_count = 0 
            
            for seg_idx, seg in enumerate(final_segments):
                segment_data = data_clean[seg['start']:seg['end']]
                
                # Z-Score 归一化
                seg_mean = np.mean(segment_data, axis=0)
                seg_std = np.std(segment_data, axis=0)
                segment_norm = (segment_data - seg_mean) / (seg_std + 1e-6)
                
                # 滑动窗口切片
                # 注意：Peak 模式下，如果 segment 长度正好等于 peak_win_ms，且 stride 很小，可能只会产出几个切片
                for w_start in range(0, len(segment_norm) - WINDOW_SIZE + 1, current_stride_size):
                    window = segment_norm[w_start : w_start + WINDOW_SIZE]
                    
                    # 原始样本
                    X_list.append(window)
                    y_list.append(label)
                    groups_list.append(f"{f}_act_{seg_idx}")
                    action_samples_count += 1
                    
                    # 增强样本
                    for _ in range(multiplier - 1):
                        aug_window = window.copy()
                        if enable_warp and np.random.random() > 0.3: aug_window = time_warp(aug_window)
                        if enable_shift and np.random.random() > 0.3: aug_window = time_shift(aug_window)
                        if enable_scaling and np.random.random() > 0.3: aug_window = scale_amplitude(aug_window)
                        if enable_mask and np.random.random() > 0.7: aug_window = channel_mask(aug_window)
                        if enable_noise: aug_window = add_noise(aug_window, noise_level=0.02)
                        
                        X_list.append(aug_window)
                        y_list.append(label)
                        groups_list.append(f"{f}_act_{seg_idx}_aug")

            # --- 提取静息样本 (Rest) ---
            if enable_rest:  
                # 不直接使用 ~raw_mask，而是重新计算严格的低能量区域
                # 这样可以防止把节拍之间的噪音误当做 Rest 训练
                
                # 1. 重新计算一个宽泛的 VAD Mask (基于能量阈值)
                # 注意：这里需要用到前面计算好的 energy_smooth
                noise_floor = np.percentile(energy_smooth, 10)
                peak_level = np.percentile(energy_smooth, 99)
                
                # 使用与 VAD 模式相同的阈值系数 (0.15)
                vad_threshold = noise_floor + 0.15 * (peak_level - noise_floor)
                vad_mask = energy_smooth > vad_threshold
                
                # 2. 对 VAD Mask 取反，得到真正的“安静区”
                rest_mask_base = ~vad_mask 
                
                # 3. 腐蚀 (Erosion) 确保远离动作边缘
                # 稍微加大安全距离 (150ms)，保证数据纯净
                safe_margin = int(0.15 * FS)
                rest_mask_base = ndimage.binary_erosion(rest_mask_base, structure=np.ones(safe_margin))
                
                labeled_rest, num_rest = ndimage.label(rest_mask_base)
                
                # 计算目标数量：保持与 GUI 逻辑兼容
                # GUI 逻辑倾向于让 Rest 数量与动作数量平衡
                target_rest_count = int(action_samples_count * multiplier * rest_ratio_per_file) 
                if target_rest_count < 5: target_rest_count = 5

                all_rest_windows = []
                for i in range(1, num_rest + 1):
                    r_indices = np.where(labeled_rest == i)[0]
                    # 长度检查
                    if len(r_indices) > WINDOW_SIZE:
                        r_seg_data = data_clean[r_indices[0]:r_indices[-1]]
                        
                        # Z-Score Norm
                        r_mean = np.mean(r_seg_data, axis=0)
                        r_std = np.std(r_seg_data, axis=0)
                        r_std = np.where(r_std < 0.01, 1.0, r_std) 
                        r_seg_norm = (r_seg_data - r_mean) / (r_std + 1e-6)
                        
                        # 较大的步长采样，避免静息数据过于重复
                        rest_stride = WINDOW_SIZE 
                        for w_start in range(0, len(r_seg_norm) - WINDOW_SIZE, rest_stride):
                            all_rest_windows.append(r_seg_norm[w_start : w_start + WINDOW_SIZE])

                if len(all_rest_windows) > 0:
                    # 随机打乱并抽取
                    indices = np.arange(len(all_rest_windows))
                    np.random.shuffle(indices)
                    selected_indices = indices[:target_rest_count]
                    
                    for idx in selected_indices:
                        X_list.append(all_rest_windows[idx])
                        y_list.append(0) # Label 0 for Rest
                        groups_list.append(f"{f}_rest")

        except Exception as e:
            print(f"Error processing {f}: {e}")
            import traceback
            traceback.print_exc()
            
    if len(X_list) == 0:
        return np.array([]), np.array([]), np.array([])
        
    return np.array(X_list), np.array(y_list), np.array(groups_list)