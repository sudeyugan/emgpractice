import pandas as pd
import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
import os
import re

# ================= 默认参数 (已根据图片和 preprocess.py 对齐) =================
FS = 1000                   # 采样率 1000Hz
WINDOW_MS = 250             # 窗口 250ms
STRIDE_MS = 100             # 步长 100ms
WINDOW_SIZE = int(FS * (WINDOW_MS / 1000))
STRIDE_SIZE = int(FS * (STRIDE_MS / 1000))

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

def process_selected_files(file_list, progress_callback=None, use_rhythm_filter=True):
    """
    处理选中的文件列表，并集成节奏过滤逻辑
    """
    X_list = []
    y_list = []
    groups_list = [] # 用于记录每个窗口属于哪个文件
    offset = int(25 * (FS / 1000))
    total = len(file_list)
    
    for idx, f in enumerate(file_list):
        if progress_callback:
            progress_callback(idx / total, f"正在处理: {os.path.basename(f)}")
            
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
            
            # 6. 遍历最终确定的片段进行滑动窗口切片
            for seg in final_segments:
                new_start = seg['start'] + offset
                new_end = seg['end'] - offset
                segment_data = data_clean[new_start:new_end]
                
                # 段内 Z-Score 归一化
                seg_mean = np.mean(segment_data, axis=0)
                seg_std = np.std(segment_data, axis=0)
                segment_norm = (segment_data - seg_mean) / (seg_std + 1e-6)
                
                # 滑动窗口切片
                for w_start in range(0, len(segment_norm) - WINDOW_SIZE, STRIDE_SIZE):
                    w_end = w_start + WINDOW_SIZE
                    window = segment_norm[w_start:w_end]
                    
                    X_list.append(window)
                    y_list.append(label)
                    groups_list.append(f)
                    
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    if len(X_list) == 0:
        return np.array([]), np.array([]), np.array([])
        
    return np.array(X_list), np.array(y_list), np.array(groups_list)