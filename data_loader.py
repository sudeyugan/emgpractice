import pandas as pd
import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
import os
import re

# ================= 默认参数 (已根据图片和 preprocess.py 对齐) =================
FS = 1000                   # 采样率 1000Hz
WINDOW_MS = 250             # 窗口 250ms
WINDOW_SIZE = int(FS * (WINDOW_MS / 1000))

# VAD (活动检测) 参数
VAD_SMOOTH_MS = 200         # 能量平滑窗口 (ms)
VAD_MERGE_GAP_MS = 300      # 合并间隙 (ms)
MIN_SEGMENT_MS = 300        # 最小动作长度 (ms)
THRESHOLD_RATIO = 0.15      # 阈值系数

# 节奏过滤参数
INTERVAL_RATIO = 0.70       # 最小间距比例 (与图片 0.70 一致)

def parse_filename_info(filepath):
    """解析文件名，返回 (Subject, Date, Label, Timestamp)"""
    filename = os.path.basename(filepath)
    # 假设路径结构 data/Subject/Date/filename
    parts = filepath.split(os.sep)
    subject = parts[-3] if len(parts) >= 3 else "Unknown"
    date = parts[-2] if len(parts) >= 3 else "Unknown"
    
    # 提取 Label: DF1.1 -> 1
    label_match = re.search(r'DF(\d+)\.', filename)
    label = int(label_match.group(1)) if label_match else None
    
    return subject, date, label, filename

def get_active_mask(data):
    """VAD 基础掩码生成 (同步 app_gui 逻辑)"""
    # 1. 带通滤波
    b, a = signal.butter(4, [20, 450], btype='bandpass', fs=FS)
    filtered = signal.filtfilt(b, a, data, axis=0)
    
    # 2. 计算能量 (RMS)
    energy = np.sqrt(np.mean(filtered**2, axis=1))
    
    # 3. 平滑能量
    win_len = int((VAD_SMOOTH_MS/1000) * FS)
    energy_smooth = np.convolve(energy, np.ones(win_len)/win_len, mode='same')
    
    # 4. 动态阈值计算
    noise_floor = np.percentile(energy_smooth, 10)
    peak_level = np.percentile(energy_smooth, 99)
    threshold = noise_floor + THRESHOLD_RATIO * (peak_level - noise_floor)
    
    # 5. 生成掩码并缝合间隙
    mask = energy_smooth > threshold
    gap_samples = int((VAD_MERGE_GAP_MS/1000) * FS)
    mask = ndimage.binary_closing(mask, structure=np.ones(gap_samples))
    return mask

def time_warp(data, sigma=0.2, knot=4):
    """
    时间扭曲：模拟动作忽快忽慢
    """
    orig_steps = np.arange(data.shape[0])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, data.shape[1]))
    warp_steps = (np.linspace(0, data.shape[0]-1., num=knot+2))
    
    ret = np.zeros_like(data)
    for i in range(data.shape[1]):
        time_warp = np.interp(orig_steps, np.linspace(0, data.shape[0]-1., num=knot+2), random_warps[:, i])
        cum_warp = np.cumsum(time_warp)
        scale = (data.shape[0]-1) / cum_warp[-1]
        new_times = cum_warp * scale
        ret[:, i] = np.interp(orig_steps, new_times, data[:, i])
    return ret

def time_shift(data, shift_limit=0.1):
    """
    时间平移：模拟动作触发时机的微小差异
    """
    shift_amt = int(data.shape[0] * shift_limit * np.random.uniform(-1, 1))
    return np.roll(data, shift_amt, axis=0)

def channel_mask(data, mask_prob=0.15):
    """
    通道遮挡：模拟某个电极接触不良 (强迫模型利用其他通道)
    """
    temp = data.copy()
    if np.random.random() < mask_prob:
        c = np.random.randint(0, data.shape[1])
        temp[:, c] = 0
    return temp

def add_noise(data, noise_level=0.01):
    """高斯白噪声"""
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def scale_amplitude(data, scale_range=(0.8, 1.2)):
    """幅度缩放"""
    factor = np.random.uniform(scale_range[0], scale_range[1])
    return data * factor

def process_selected_files(file_list, progress_callback=None, use_rhythm_filter=True, stride_ms=100, augment_config=None):
    """
    更新：自动提取静息 (Rest) 数据作为 Label 0
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

    rest_ratio_per_file = 0
    if enable_rest:
        unique_labels = set()
        for f in file_list:
            try:
                _, _, label, _ = parse_filename_info(f)
                if label is not None: unique_labels.add(label)
            except: pass
        
        num_act_classes = len(unique_labels) if len(unique_labels) > 0 else 1
        # 目标：让 Rest 总量 ≈ 动作总量的 1.1 倍
        rest_ratio_per_file = 1.1 / num_act_classes
        print(f"[Info] 静息模式开启。动作类别数: {num_act_classes}, 单文件静息系数: {rest_ratio_per_file:.3f}")
    else:
        print("[Info] 静息模式关闭。只训练动作样本。")
    
    X_list = []
    y_list = []
    groups_list = [] 
    
    current_stride_size = int(FS * (stride_ms / 1000))
    if current_stride_size < 1: current_stride_size = 1
    
    total = len(file_list)
    
    for idx, f in enumerate(file_list):
        if progress_callback:
            progress_callback(idx / total, f"处理中: {os.path.basename(f)}")
            
        try:
            _, _, label, _ = parse_filename_info(f)
            if label is None: continue
            
            df = pd.read_csv(f)
            cols = [c for c in df.columns if 'CH' in c]
            raw_data = df[cols].values
            
            # 1. 滤波
            b, a = signal.butter(4, [20, 450], btype='bandpass', fs=FS)
            data_clean = signal.filtfilt(b, a, raw_data, axis=0)
            
            # 2. VAD 动作检测
            mask = get_active_mask(raw_data) # 获取动作掩码

            # PART A: 提取动作样本 (Label = 1, 2, 3...)
            labeled_mask, num_raw_features = ndimage.label(mask)
            
            candidate_segments = []
            for i in range(1, num_raw_features + 1):
                indices = np.where(labeled_mask == i)[0]
                if len(indices) >= int((MIN_SEGMENT_MS/1000) * FS):
                    candidate_segments.append({
                        'start': indices[0],
                        'end': indices[-1],
                        'center': (indices[0] + indices[-1]) / 2
                    })

            if len(candidate_segments) > 0:
                # 1. 计算所有片段的 RMS 能量
                segment_energies = []
                for seg in candidate_segments:
                    # 使用 data_clean 计算，它已经滤除了低频漂移和高频噪点
                    seg_data = data_clean[seg['start']:seg['end']]
                    # 计算 RMS (Root Mean Square) 并对所有通道求平均
                    rms = np.mean(np.sqrt(np.mean(seg_data**2, axis=0)))
                    segment_energies.append(rms)
                
                # 2. 计算基准 (使用中位数，防止被异常值拉偏)
                median_energy = np.median(segment_energies)
                
                # 3. 执行过滤
                filtered_segments = []
                # 设定倍率阈值 (你提到的是 5 倍)
                energy_threshold_ratio = 5.0 
                
                for i, seg in enumerate(candidate_segments):
                    # 如果该片段能量小于 5倍基准，或者是该文件唯一的片段(无法比较)，则保留
                    if segment_energies[i] < median_energy * energy_threshold_ratio:
                        filtered_segments.append(seg)
                    else:
                        pass
                        # print(f"  [Filter] 剔除高能异常(翻腕?): RMS={segment_energies[i]:.2f} (基准: {median_energy:.2f})")
                
                candidate_segments = filtered_segments

            # 节奏过滤
            final_segments = []
            if use_rhythm_filter and len(candidate_segments) > 1:
                centers = np.array([s['center'] for s in candidate_segments])
                diffs = np.diff(centers)
                median_interval = np.median(diffs)
                final_segments.append(candidate_segments[0])
                for i in range(1, len(candidate_segments)):
                    last_center = final_segments[-1]['center']
                    curr_center = candidate_segments[i]['center']
                    if (curr_center - last_center) > median_interval * INTERVAL_RATIO:
                        final_segments.append(candidate_segments[i])
            else:
                final_segments = candidate_segments
            
            # 记录当前文件产生了多少个动作样本，用于平衡静息数据
            action_samples_count = 0 
            
            for seg_idx, seg in enumerate(final_segments):
                segment_data = data_clean[seg['start']:seg['end']]
                
                # Z-Score 归一化
                seg_mean = np.mean(segment_data, axis=0)
                seg_std = np.std(segment_data, axis=0)
                segment_norm = (segment_data - seg_mean) / (seg_std + 1e-6)
                
                for w_start in range(0, len(segment_norm) - WINDOW_SIZE, current_stride_size):
                    window = segment_norm[w_start : w_start + WINDOW_SIZE]
                    
                    # 动作样本 (Base)
                    X_list.append(window)
                    y_list.append(label) # 原始标签
                    groups_list.append(f"{f}_act_{seg_idx}")
                    action_samples_count += 1
                    
                    # 动作增强 (Few-shot)
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

            # PART B: 提取静息样本 (Label = 0)
            if enable_rest:  
                rest_mask = ~mask 
                safe_margin = int(0.1 * FS)
                rest_mask = ndimage.binary_erosion(rest_mask, structure=np.ones(safe_margin))
                labeled_rest, num_rest = ndimage.label(rest_mask)
                
                # 计算需要多少静息样本
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
                        win = all_rest_windows[idx]
                        X_list.append(win)
                        y_list.append(0) # 标签 0
                        groups_list.append(f"{f}_rest")
                        # 仅在需要时微量增强静息
                        if enable_noise and np.random.random() > 0.5:
                             pass 

        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    if len(X_list) == 0:
        return np.array([]), np.array([]), np.array([])
        
    return np.array(X_list), np.array(y_list), np.array(groups_list)