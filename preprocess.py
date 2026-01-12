import pandas as pd
import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
import os
import glob
import re

# ================= 配置区域 =================
# 1. 路径设置
DATA_ROOT = 'data'
OUTPUT_DIR = 'processed_data'

# 2. 筛选策略 (逐步扩展模型时改这里)
TARGET_SUBJECT = 'charles'  # 设置为 None 则处理所有人
TARGET_DATE = '20250213'    # 设置为 None 则处理所有日期
# TARGET_DATE = None        # 示例：处理 charles 所有日期的数据

# 3. 信号处理参数
FS = 1000                   # 采样率 1000Hz
WINDOW_MS = 250             # 窗口 250ms
STRIDE_MS = 100             # 步长 100ms
WINDOW_SIZE = int(FS * (WINDOW_MS / 1000))
STRIDE_SIZE = int(FS * (STRIDE_MS / 1000))

# 4. 动作检测 (VAD) 参数
VAD_SMOOTH_MS = 200         # 平滑窗口，越大越不容易断
VAD_MERGE_GAP_MS = 300      # 两个动作如果间隔小于300ms，视为一个连续动作
MIN_SEGMENT_MS = 300        # 如果切出来的动作小于300ms，扔掉（可能是噪音）

# ===========================================

def parse_filename(filename):
    """
    文件名解析：RAW_EMG_RightHand_DF1.1_ARband_charles-20250303_20250303141311.csv
    1. 提取标签: DF1.1 -> 1
    2. 提取时间戳: 末尾的 20250303141311
    """
    # 提取标签 (DF后的数字，直到点号)
    label_match = re.search(r'DF(\d+)\.', filename)
    # 提取时间戳 (文件名末尾的14位数字)
    ts_match = re.search(r'(\d{14})\.csv$', filename)
    
    label = int(label_match.group(1)) if label_match else None
    timestamp_str = ts_match.group(1) if ts_match else None
    
    return label, timestamp_str

def get_active_mask(data):
    """
    高级活动段检测：
    1. 带通滤波 -> 能量计算
    2. 双阈值检测
    3. 膨胀操作 (Filling Gaps) 连接断断续续的动作
    """
    # 1. 预滤波 (仅用于检测，不改变原始数据)
    b, a = signal.butter(4, [20, 450], btype='bandpass', fs=FS)
    filtered = signal.filtfilt(b, a, data, axis=0)
    energy = np.sqrt(np.mean(filtered**2, axis=1))
    
    # 2. 平滑能量
    win_len = int((VAD_SMOOTH_MS/1000) * FS)
    energy_smooth = np.convolve(energy, np.ones(win_len)/win_len, mode='same')
    
    # 3. 动态阈值
    noise_floor = np.percentile(energy_smooth, 10) # 底噪
    peak_level = np.percentile(energy_smooth, 99)  # 峰值
    threshold = noise_floor + 0.15 * (peak_level - noise_floor)
    
    # 4. 生成掩码
    mask = energy_smooth > threshold
    
    # 5. 缝合间隙 (Morphological Closing)
    # 如果两个动作间隔 < 300ms，把它们连起来
    gap_samples = int((VAD_MERGE_GAP_MS/1000) * FS)
    mask = ndimage.binary_closing(mask, structure=np.ones(gap_samples))
    
    return mask

def process_data():
    X_list = []
    y_list = []
    meta_list = [] # 用于存储 [文件名, 文件时间戳, 窗口起始相对时间]
    
    # 构建搜索路径: data/charles/20250213/RAW_EMG*.csv
    search_pattern = os.path.join(DATA_ROOT, 
                                  TARGET_SUBJECT if TARGET_SUBJECT else '*', 
                                  TARGET_DATE if TARGET_DATE else '*', 
                                  "RAW_EMG*.csv")
    
    files = glob.glob(search_pattern)
    print(f"Searching: {search_pattern}")
    print(f"Found {len(files)} files.")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for f in files:
        fname = os.path.basename(f)
        
        # 排除 SOLVED 等非 RAW 文件 (虽然glob已经限制了RAW开头，双重保险)
        if not fname.startswith("RAW_EMG"): continue
        
        # 解析信息
        label, file_ts = parse_filename(fname)
        if label is None:
            print(f"[Skip] Can't parse label: {fname}")
            continue
            
        print(f"Processing: {fname} | Label: {label}")
        
        # 读取数据
        try:
            df = pd.read_csv(f)
            # 提取 CH1-CH5
            cols = [c for c in df.columns if 'CH' in c]
            raw_data = df[cols].values
        except Exception as e:
            print(f"  Error reading file: {e}")
            continue
            
        # --- 信号处理流水线 ---
        
        # 1. 真正的滤波 (用于训练的数据)
        b, a = signal.butter(4, [20, 450], btype='bandpass', fs=FS)
        data_clean = signal.filtfilt(b, a, raw_data, axis=0)
        
        # 2. VAD 提取动作区
        mask = get_active_mask(raw_data) # 使用原始数据或滤波数据做检测均可
        
        # 3. 遍历每个连续动作段
        labeled_mask, num_features = ndimage.label(mask)
        
        for segment_idx in range(1, num_features+1):
            # 获取该段的索引
            indices = np.where(labeled_mask == segment_idx)[0]
            if len(indices) < int((MIN_SEGMENT_MS/1000) * FS):
                continue # 太短的忽略
                
            start_pos = indices[0]
            end_pos = indices[-1]
            segment_data = data_clean[start_pos:end_pos]
            
            # 4. 段内归一化 (Z-Score)
            # 注意：在段内做归一化可以对抗不同力度的幅度差异
            seg_mean = np.mean(segment_data, axis=0)
            seg_std = np.std(segment_data, axis=0)
            segment_norm = (segment_data - seg_mean) / (seg_std + 1e-6)
            
            # 5. 滑动窗口切片
            for w_start in range(0, len(segment_norm) - WINDOW_SIZE, STRIDE_SIZE):
                w_end = w_start + WINDOW_SIZE
                window = segment_norm[w_start:w_end]
                
                # 保存数据
                X_list.append(window)
                y_list.append(label)
                
                # 保存元数据 (方便以后找回这个窗口对应的时间点)
                # 绝对时间 = 文件时间 + (段开始 + 窗口开始)/FS
                abs_offset_samples = start_pos + w_start
                meta_list.append([fname, file_ts, abs_offset_samples])

    # 转换并保存
    X_arr = np.array(X_list)
    y_arr = np.array(y_list)
    meta_df = pd.DataFrame(meta_list, columns=['FileName', 'FileTimestamp', 'SampleIndex'])
    
    print(f"\nResult: Generated {len(X_arr)} samples.")
    print(f"X shape: {X_arr.shape}")
    
    if len(X_arr) > 0:
        np.save(os.path.join(OUTPUT_DIR, 'X.npy'), X_arr)
        np.save(os.path.join(OUTPUT_DIR, 'y.npy'), y_arr)
        meta_df.to_csv(os.path.join(OUTPUT_DIR, 'meta.csv'), index=False)
        print("Saved to processed_data/")
    else:
        print("No samples generated. Check your paths or threshold.")

if __name__ == '__main__':
    process_data()