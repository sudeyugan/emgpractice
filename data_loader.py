import pandas as pd
import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
import os
import re

# ================= 默认参数 (与原 preprocess.py 保持一致) =================
FS = 1000
WINDOW_MS = 250
STRIDE_MS = 100
WINDOW_SIZE = int(FS * (WINDOW_MS / 1000))
STRIDE_SIZE = int(FS * (STRIDE_MS / 1000))

# VAD 参数
VAD_SMOOTH_MS = 200
VAD_MERGE_GAP_MS = 300
MIN_SEGMENT_MS = 300

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
    """(复用原 preprocess.py 的 VAD 逻辑)"""
    b, a = signal.butter(4, [20, 450], btype='bandpass', fs=FS)
    filtered = signal.filtfilt(b, a, data, axis=0)
    energy = np.sqrt(np.mean(filtered**2, axis=1))
    
    win_len = int((VAD_SMOOTH_MS/1000) * FS)
    energy_smooth = np.convolve(energy, np.ones(win_len)/win_len, mode='same')
    
    noise_floor = np.percentile(energy_smooth, 10)
    peak_level = np.percentile(energy_smooth, 99)
    threshold = noise_floor + 0.15 * (peak_level - noise_floor)
    
    mask = energy_smooth > threshold
    gap_samples = int((VAD_MERGE_GAP_MS/1000) * FS)
    mask = ndimage.binary_closing(mask, structure=np.ones(gap_samples))
    return mask

def process_selected_files(file_list, progress_callback=None):
    """
    处理选中的文件列表
    :param file_list: 文件的完整路径列表
    :param progress_callback:Streamlit 的进度条回调函数
    """
    X_list = []
    y_list = []
    
    total = len(file_list)
    
    for idx, f in enumerate(file_list):
        if progress_callback:
            progress_callback(idx / total, f"正在处理: {os.path.basename(f)}")
            
        try:
            # 1. 解析标签
            _, _, label, _ = parse_filename_info(f)
            if label is None: continue
            
            # 2. 读取数据
            df = pd.read_csv(f)
            cols = [c for c in df.columns if 'CH' in c]
            raw_data = df[cols].values
            
            # 3. 滤波
            b, a = signal.butter(4, [20, 450], btype='bandpass', fs=FS)
            data_clean = signal.filtfilt(b, a, raw_data, axis=0)
            
            # 4. VAD 检测
            mask = get_active_mask(raw_data)
            labeled_mask, num_features = ndimage.label(mask)
            
            # 5. 遍历动作片段
            for segment_idx in range(1, num_features + 1):
                indices = np.where(labeled_mask == segment_idx)[0]
                if len(indices) < int((MIN_SEGMENT_MS/1000) * FS):
                    continue
                
                # 简单过滤：这里省略了 preprocess.py 中复杂的节奏过滤，
                # 如果需要严格过滤，可以将那部分代码也搬过来。
                
                # 6. 切片
                segment_data = data_clean[indices[0]:indices[-1]]
                
                # 段内归一化
                seg_mean = np.mean(segment_data, axis=0)
                seg_std = np.std(segment_data, axis=0)
                segment_norm = (segment_data - seg_mean) / (seg_std + 1e-6)
                
                # 滑动窗口
                for w_start in range(0, len(segment_norm) - WINDOW_SIZE, STRIDE_SIZE):
                    w_end = w_start + WINDOW_SIZE
                    window = segment_norm[w_start:w_end]
                    X_list.append(window)
                    y_list.append(label)
                    
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    if len(X_list) == 0:
        return np.array([]), np.array([])
        
    return np.array(X_list), np.array(y_list)