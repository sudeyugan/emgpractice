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

def process_selected_files(file_list, progress_callback=None, stride_ms=100, augment_config=None):
    """
    核心处理流程：严格同步 app_gui.py 的信号处理链
    Raw -> CH5 Gain -> Notch -> Bandpass -> Energy -> VAD -> Refine -> Rhythm -> Slice
    """
    if augment_config is None: augment_config = {}
    
    # 增强配置
    multiplier = augment_config.get('multiplier', 1)
    enable_rest = augment_config.get('enable_rest', True)
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
                
            # --- 2. 信号滤波链 (与 GUI 一致) ---
            data_proc = raw_data.copy()
            
            # A. 工频陷波 (Notch) - GUI: 启用, 50Hz
            if ENABLE_NOTCH:
                b_notch, a_notch = signal.iirnotch(NOTCH_FREQ, 30, FS)
                data_proc = signal.filtfilt(b_notch, a_notch, data_proc, axis=0)
            
            # B. 带通滤波 (Bandpass) - GUI: 20-450Hz
            b, a = signal.butter(4, [20, 450], btype='bandpass', fs=FS)
            data_clean = signal.filtfilt(b, a, data_proc, axis=0)
            
            # --- 3. VAD 掩码生成 ---
            # 计算能量
            energy = np.sqrt(np.mean(data_clean**2, axis=1))
            
            # 平滑 - GUI: 100ms
            win_len = int((VAD_SMOOTH_MS/1000) * FS)
            energy_smooth = np.convolve(energy, np.ones(win_len)/win_len, mode='same')
            
            # 阈值 - GUI: 0.15
            noise_floor = np.percentile(energy_smooth, 10)
            peak_level = np.percentile(energy_smooth, 99)
            threshold = noise_floor + THRESHOLD_RATIO * (peak_level - noise_floor)
            
            raw_mask = energy_smooth > threshold
            
            # 合并间隙 - GUI: 200ms
            gap_samples = int((VAD_MERGE_GAP_MS/1000) * FS)
            raw_mask = ndimage.binary_closing(raw_mask, structure=np.ones(gap_samples))
            
            # --- 4. 过滤逻辑优化 (Refine & Rhythm) ---
            
            # A. 时长门控 (Refine) - GUI: 启用
            if ENABLE_REFINE:
                final_mask = refine_mask_logic(raw_mask)
            else:
                final_mask = raw_mask
                
            # 提取候选片段
            labeled_mask, num_features = ndimage.label(final_mask)
            candidate_segments = []
            for i in range(1, num_features + 1):
                indices = np.where(labeled_mask == i)[0]
                # 再次校验长度 (refine_logic 已经做过，但为了安全)
                if len(indices) > 0:
                    candidate_segments.append({
                        'start': indices[0],
                        'end': indices[-1],
                        'center': (indices[0] + indices[-1]) / 2
                    })
            
            if len(candidate_segments) > 0:
                segment_energies = []
                for seg in candidate_segments:
                    # 使用 data_clean (多通道) 计算平均 RMS
                    seg_data = data_clean[seg['start']:seg['end']]
                    # 1. 算出每个通道的 RMS -> 2. 对所有通道取平均
                    rms = np.mean(np.sqrt(np.mean(seg_data**2, axis=0)))
                    segment_energies.append(rms)
                
                # 计算基准 (中位数)
                median_energy = np.median(segment_energies)
                # 设定倍率阈值 (通常 5.0 倍于中位数的视为异常)
                energy_threshold = median_energy * 5.0
                
                filtered_candidates = []
                for i, seg in enumerate(candidate_segments):
                    if segment_energies[i] < energy_threshold:
                        filtered_candidates.append(seg)
                    # else: print(f"剔除高能异常: {segment_energies[i]:.2f} > {energy_threshold:.2f}")
                
                candidate_segments = filtered_candidates

            # B. 节奏过滤 (Rhythm) - GUI: 启用 4s, 0.90
            final_segments = []
            if ENABLE_RHYTHM and len(candidate_segments) > 1:
                # 使用固定 4秒 间隔
                expected_interval = EXPECTED_INTERVAL_MS * (FS / 1000) # samples
                min_gap = expected_interval * INTERVAL_RATIO
                
                final_segments.append(candidate_segments[0])
                last_center = candidate_segments[0]['center']
                
                for i in range(1, len(candidate_segments)):
                    curr_center = candidate_segments[i]['center']
                    # 只有当距离上一个有效动作足够远时才保留
                    if (curr_center - last_center) > min_gap:
                        final_segments.append(candidate_segments[i])
                        last_center = curr_center
            else:
                final_segments = candidate_segments

            # --- 5. 切片与增强 ---
            action_samples_count = 0 
            
            for seg_idx, seg in enumerate(final_segments):
                segment_data = data_clean[seg['start']:seg['end']]
                
                # Z-Score 归一化 (使用段内统计量)
                seg_mean = np.mean(segment_data, axis=0)
                seg_std = np.std(segment_data, axis=0)
                segment_norm = (segment_data - seg_mean) / (seg_std + 1e-6)
                
                # 滑动窗口切片
                for w_start in range(0, len(segment_norm) - WINDOW_SIZE, current_stride_size):
                    window = segment_norm[w_start : w_start + WINDOW_SIZE]
                    
                    # 原始样本
                    X_list.append(window)
                    y_list.append(label)
                    groups_list.append(f"{f}_act_{seg_idx}")
                    action_samples_count += 1
                    
                    # 增强样本 (Few-shot)
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

            # --- 6. 提取静息样本 (Rest) ---
            if enable_rest:  
                # 注意：静息样本应从 mask 之外提取，mask 应该包含所有被视为"活动"的区域
                # 为了安全，我们对原始 VAD 结果取反，而不是 refine 后的，以避免把剔除的噪音当成静息
                rest_mask_base = ~raw_mask 
                safe_margin = int(0.1 * FS)
                rest_mask_base = ndimage.binary_erosion(rest_mask_base, structure=np.ones(safe_margin))
                labeled_rest, num_rest = ndimage.label(rest_mask_base)
                
                target_rest_count = int(action_samples_count * multiplier * rest_ratio_per_file) 
                if target_rest_count < 5: target_rest_count = 5

                all_rest_windows = []
                for i in range(1, num_rest + 1):
                    r_indices = np.where(labeled_rest == i)[0]
                    if len(r_indices) > WINDOW_SIZE:
                        r_seg_data = data_clean[r_indices[0]:r_indices[-1]]
                        r_mean = np.mean(r_seg_data, axis=0)
                        r_std = np.std(r_seg_data, axis=0)
                        r_std = np.where(r_std < 0.01, 1.0, r_std) 
                        r_seg_norm = (r_seg_data - r_mean) / (r_std + 1e-6)
                        
                        rest_stride = current_stride_size * 2
                        for w_start in range(0, len(r_seg_norm) - WINDOW_SIZE, rest_stride):
                            all_rest_windows.append(r_seg_norm[w_start : w_start + WINDOW_SIZE])

                if len(all_rest_windows) > 0:
                    indices = np.arange(len(all_rest_windows))
                    np.random.shuffle(indices)
                    selected_indices = indices[:target_rest_count]
                    
                    for idx in selected_indices:
                        X_list.append(all_rest_windows[idx])
                        y_list.append(0) # Label 0 for Rest
                        groups_list.append(f"{f}_rest")

        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    if len(X_list) == 0:
        return np.array([]), np.array([]), np.array([])
        
    return np.array(X_list), np.array(y_list), np.array(groups_list)