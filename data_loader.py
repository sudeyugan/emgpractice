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

def add_noise(data, noise_level=0.01):
    """加入高斯白噪声"""
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def scale_amplitude(data, scale_range=(0.8, 1.2)):
    """随机幅度缩放"""
    factor = np.random.uniform(scale_range[0], scale_range[1])
    return data * factor

def time_warp(data, sigma=0.2, knot=4):
    """
    时间扭曲：通过插值改变信号局部的时间流速
    (模拟动作忽快忽慢)
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
    时间平移：随机左右移动窗口
    """
    shift_amt = int(data.shape[0] * shift_limit * np.random.uniform(-1, 1))
    return np.roll(data, shift_amt, axis=0)

def channel_mask(data, mask_prob=0.15):
    """
    通道遮挡：随机把某一列（通道）置零
    (模拟电极接触不良，强迫模型看整体)
    """
    temp = data.copy()
    if np.random.random() < mask_prob:
        # 随机选一个通道置零
        c = np.random.randint(0, data.shape[1])
        temp[:, c] = 0
    return temp

def process_selected_files(file_list, progress_callback=None, use_rhythm_filter=True, stride_ms=100, augment_config=None):
    """
    augment_config: dict, e.g. {'enable_noise': True, 'noise_level': 0.02, 'enable_scaling': True}
    """
    if augment_config is None:
        augment_config = {}
        
    X_list = []
    y_list = []
    groups_list = [] 
    
    # 动态计算步长大小
    current_stride_size = int(FS * (stride_ms / 1000))
    if current_stride_size < 1: current_stride_size = 1
    
    total = len(file_list)
    
    for idx, f in enumerate(file_list):
        if progress_callback:
            progress_callback(idx / total, f"处理中: {os.path.basename(f)}")
            
        try:
            # 1. 解析信息
            _, _, label, _ = parse_filename_info(f)
            if label is None: continue
            
            # 2. 读取数据
            df = pd.read_csv(f)
            cols = [c for c in df.columns if 'CH' in c]
            raw_data = df[cols].values
            
            # 3. 滤波 (用于训练的数据)
            b, a = signal.butter(4, [20, 450], btype='bandpass', fs=FS)
            data_clean = signal.filtfilt(b, a, raw_data, axis=0)
            
            # 4. VAD 检测与初步片段提取
            mask = get_active_mask(raw_data)
            labeled_mask, num_raw_features = ndimage.label(mask)
            
            candidate_segments = []
            for i in range(1, num_raw_features + 1):
                indices = np.where(labeled_mask == i)[0]
                # 长度过滤
                if len(indices) >= int((MIN_SEGMENT_MS/1000) * FS):
                    candidate_segments.append({
                        'start': indices[0],
                        'end': indices[-1],
                        'center': (indices[0] + indices[-1]) / 2
                    })

            # 5. 核心：等间距节奏过滤 (同步 preprocess.py/app_gui 逻辑)
            final_segments = []
            if use_rhythm_filter and len(candidate_segments) > 1:
                centers = np.array([s['center'] for s in candidate_segments])
                diffs = np.diff(centers)
                median_interval = np.median(diffs) # 计算基准节奏
                
                final_segments.append(candidate_segments[0]) # 默认保留第一个
                for i in range(1, len(candidate_segments)):
                    last_center = final_segments[-1]['center']
                    curr_center = candidate_segments[i]['center']
                    # 只有满足间距比例要求的才保留
                    if (curr_center - last_center) > median_interval * INTERVAL_RATIO:
                        final_segments.append(candidate_segments[i])
            else:
                final_segments = candidate_segments
            
            # 6. 遍历最终片段进行切片
            for idx, seg in enumerate(final_segments):

                segment_data = data_clean[seg['start']:seg['end']]

                # 段内归一化 (Z-Score) - 这是一个很好的实践，保持住
                seg_mean = np.mean(segment_data, axis=0)
                seg_std = np.std(segment_data, axis=0)
                segment_norm = (segment_data - seg_mean) / (seg_std + 1e-6)
                
                # 滑动窗口切片 (使用动态 stride)
                for w_start in range(0, len(segment_norm) - WINDOW_SIZE, current_stride_size):
                    w_end = w_start + WINDOW_SIZE
                    window = segment_norm[w_start:w_end]
                    
                    # === A. 保存原始窗口 ===
                    X_list.append(window)
                    y_list.append(label)
                    groups_list.append(f"{f}_seg{idx}")
                    
                    # === B. 数据增强 (生成额外的窗口) ===
                    # 只有在 augment_config 存在且非空时执行
                    if augment_config:
                        # 1. 随机幅度缩放
                        if augment_config.get('enable_scaling', False):
                            aug_window = scale_amplitude(window, scale_range=(0.8, 1.2))
                            X_list.append(aug_window)
                            y_list.append(label)
                            groups_list.append(f"{f}_seg{idx}")
                            
                        # 2. 高斯噪声 (可以在缩放的基础上加，也可以单独加，这里演示独立加)
                        if augment_config.get('enable_noise', False):
                            # 注意：Z-Score后的数据通常在 -3 到 3 之间，0.05 的噪声已经很明显了
                            aug_window = add_noise(window, noise_level=0.05) 
                            X_list.append(aug_window)
                            y_list.append(label)
                            groups_list.append(f"{f}_seg{idx}")

        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    if len(X_list) == 0:
        return np.array([]), np.array([]), np.array([])
        
    return np.array(X_list), np.array(y_list), np.array(groups_list)