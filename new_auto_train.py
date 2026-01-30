import os
import sys
import time
import glob
import datetime
import re
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.ndimage as ndimage
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit

# 引用现有模块 (确保这些文件在同一目录下)
import train_utils
import model as model_lib

def time_warp(data, sigma=0.2, knot=4):
    """时间扭曲"""
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
    """时间平移"""
    shift_amt = int(data.shape[0] * shift_limit * np.random.uniform(-1, 1))
    return np.roll(data, shift_amt, axis=0)

def channel_mask(data, mask_prob=0.15):
    """通道遮挡"""
    temp = data.copy()
    if np.random.random() < mask_prob:
        c = np.random.randint(0, data.shape[1])
        temp[:, c] = 0
    return temp

def load_and_resample_imu(emg_filepath, target_length):
    """
    根据 EMG 文件路径查找对应的 IMU 文件，并将其重采样到 target_length (1000Hz)
    """
    # 假设文件名格式一致，只是前缀不同：RAW_EMG_... -> RAW_IMU_...
    imu_filepath = emg_filepath.replace("RAW_EMG", "RAW_IMU")
    
    if not os.path.exists(imu_filepath):
        # 尝试另一种常见情况：如果文件夹结构不同，可能需要更复杂的查找
        # 这里假设它们在同一目录下
        return None

    try:
        df = pd.read_csv(imu_filepath)
        # 确保列名正确，根据你提供的文件内容: AX,AY,AZ,GX,GY,GZ
        required_cols = ['AX', 'AY', 'AZ', 'GX', 'GY', 'GZ']
        
        # 检查列是否存在
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: IMU file {os.path.basename(imu_filepath)} missing columns.")
            return None
            
        imu_data = df[required_cols].values # Shape: (N_imu, 6)
        
        # --- 重采样逻辑 (200Hz -> 1000Hz) ---
        # 使用线性插值将 IMU 数据拉伸到与 EMG 数据相同的长度 (target_length)
        x_old = np.linspace(0, 1, len(imu_data))
        x_new = np.linspace(0, 1, target_length)
        
        imu_resampled = np.zeros((target_length, 6))
        for i in range(6):
            imu_resampled[:, i] = np.interp(x_new, x_old, imu_data[:, i])
            
        return imu_resampled

    except Exception as e:
        print(f"Error loading IMU {imu_filepath}: {e}")
        return None

def augment_dataset_in_memory(X, y, groups, config):
    """
    在内存中对数据集进行增强。
    X: (N, Time, Channels)
    """
    multiplier = config.get('multiplier', 1)
    if multiplier <= 1:
        return X, y, groups
    
    print(f"    ⚡ 正在执行内存增强 (倍率: {multiplier}x)...")
    
    X_aug_list, y_aug_list, groups_aug_list = [], [], []
    total = len(X)
    
    # 提取配置
    enable_warp = config.get('enable_warp', False)
    enable_shift = config.get('enable_shift', False)
    enable_scale = config.get('enable_scaling', False)
    enable_mask = config.get('enable_mask', False)
    enable_noise = config.get('enable_noise', False)
    
    for i in range(total):
        # 1. 始终保留原始样本
        X_aug_list.append(X[i])
        y_aug_list.append(y[i])
        groups_aug_list.append(groups[i])
        
        # 2. 生成 (multiplier - 1) 个增强副本
        for _ in range(multiplier - 1):
            aug_x = X[i].copy() # 必须 copy
            
            # 概率应用增强 (参数可根据需要微调)
            if enable_warp and np.random.random() > 0.5:
                aug_x = time_warp(aug_x)
            
            if enable_shift and np.random.random() > 0.5:
                aug_x = time_shift(aug_x)
                
            if enable_scale and np.random.random() > 0.3:
                # 简单的幅度缩放实现
                factor = np.random.uniform(0.8, 1.2)
                aug_x = aug_x * factor
                
            if enable_mask and np.random.random() > 0.7:
                aug_x = channel_mask(aug_x)
                
            if enable_noise: 
                # 高斯噪声
                noise = np.random.normal(0, 0.02, aug_x.shape)
                aug_x = aug_x + noise
            
            X_aug_list.append(aug_x)
            y_aug_list.append(y[i])
            # 增强样本共享同一个 Group ID，确保它们总是被分在一起（虽然这里是在 split 后增强，但保持 ID 一致是个好习惯）
            groups_aug_list.append(groups[i]) 
            
    return np.array(X_aug_list, dtype=np.float32), np.array(y_aug_list), np.array(groups_aug_list)

# ==================== 0. 配置区域 ====================

# 1. 目标设置
TARGET_SUBJECTS = ["charles", "gavvin", "gerard", "giland", "jessie", "legend"] 
TARGET_LABELS = [5, 6, 8]            # 指定动作标签
TARGET_DATES = None                     # None 表示所有日期

# 2. 实验模型 (Grid Search)
MODELS_TO_TEST = [
    ("Advanced_CRNN", model_lib.build_advanced_crnn),
    ("TCN", model_lib.build_tcn_model),
    ("ResNet1D", model_lib.build_resnet_model),
]

OPTIMIZERS_TO_TEST = [
    ("Adam", tf.keras.optimizers.Adam, 0.001, {}),
    ("AdamW", tf.keras.optimizers.AdamW, 0.001, {'weight_decay': 1e-4}),
    ("Nadam", tf.keras.optimizers.Nadam, 0.001, {}),
]

VOTING_OPTIONS = [False] # 是否开启投票

# 3. 核心参数 (Rhythm Logic)
CONFIG = {
    'fs': 1000,                # 采样率
    'use_imu': True,
    'rhythm_interval_ms': 4000,# [关键] 动作间隔 (节拍器速度)
    'rhythm_window_ms': 352,   # [关键] 每次截取的窗口大小 (以峰值为中心)
    'epochs': 100,
    'batch_size': 128,
    'window_ms': 350,          # 输入模型的窗口大小
    'stride_ms': 350,           # 切片步长
    'test_size': 0.2,
    'split_strategy': "混合切分 (看到所有天/人)", 
}

# 4. 数据增强
AUGMENT_CONFIG = {
    'enable_rest': True,       # 是否采集静息数据 (Label 0)
    'multiplier': 3,           # 数据倍增系数
    'enable_scaling': True,
    'enable_noise': True,
    'enable_warp': False,      # 时间扭曲 (耗时，视情况开启)
    'enable_shift': False,
    'enable_mask': False
}

LOG_DIR = "1.26_auto_train_logs_rhythm_withoutstride"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# ==================== 1. 核心算法 (移植自 new_app_gui.py) ====================

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
    [Core Logic] 4s 固定节奏峰值提取逻辑 + 相位投票
    """
    mask = np.zeros_like(energy, dtype=bool)
    
    # 1. 寻找候选峰
    min_dist = int(2.0 * fs) 
    noise_floor = np.percentile(energy, 10)
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
    search_radius = int(1.0 * fs)
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
                # 再次校验峰值强度，防止提取到纯底噪
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
        mean_e = np.mean(seg_vals)
        std_e = np.std(seg_vals)
        cv = std_e / (mean_e + 1e-6)
        
        ref_energy = energy[anchor_peak]
        # 如果能量很大但 CV 很小 (平稳噪音)，剔除
        if mean_e > ref_energy * 0.3 and cv < noise_cv_threshold:
             continue
             
        mask[s:e] = True
        
    return mask

# ==================== 2. 数据处理流水线 ====================

def process_files_with_rhythm(file_list, config, augment_config):
    """
    修改版：取消切片 (No Slicing/Stride)
    每个检测到的动作段只取中心的一个窗口作为样本。
    """
    X_list, y_list, groups_list = [], [], []
    
    fs = config['fs']
    win_size = int(fs * (config['window_ms'] / 1000))
    # stride 变量不再需要，因为我们不再滑动
    
    # 增强参数
    multiplier = augment_config.get('multiplier', 1)
    enable_rest = augment_config.get('enable_rest', True)
    
    # 用于计算静息样本比例
    total_act_samples = 0
    
    print(f"⏳ 正在处理 {len(file_list)} 个文件 (Mode: No Slicing, Center Crop)...")
    use_imu = config.get('use_imu', False)
    
    for i, f_path in enumerate(file_list):
        current_file_act_count = 0
        try:
            subject, date, label, fname = parse_filename_info(f_path)
            if label is None: continue

            df = pd.read_csv(f_path)
            cols = [c for c in df.columns if 'CH' in c]
            raw_emg = df[cols].values
            if raw_emg.shape[1] >= 5: raw_emg[:, 4] = raw_emg[:, 4] * 2.5
            
            # --- Load IMU & Merge ---
            if use_imu:
                imu_data = load_and_resample_imu(f_path, len(raw_emg))
                if imu_data is not None:
                    raw_data = np.hstack((raw_emg, imu_data))
                else:
                    print(f"⚠️ Skip {os.path.basename(f_path)}: IMU missing")
                    continue
            else:
                raw_data = raw_emg
                
            emg_cols = raw_emg.shape[1]
            data_proc = raw_data.copy()
                
            # --- Filter Chain ---
            b_notch, a_notch = signal.iirnotch(50, 30, fs)
            data_proc[:, :emg_cols] = signal.filtfilt(b_notch, a_notch, data_proc[:, :emg_cols], axis=0)
            
            b, a = signal.butter(4, [20, 450], btype='bandpass', fs=fs)
            data_clean = data_proc
            data_clean[:, :emg_cols] = signal.filtfilt(b, a, data_proc[:, :emg_cols], axis=0)
            
            # Energy Calculation
            emg_part = data_clean[:, :emg_cols]
            energy = np.sqrt(np.mean(emg_part**2, axis=1))
            win_len = int(0.1 * fs)
            energy_smooth = np.convolve(energy, np.ones(win_len)/win_len, mode='same')
            
            # --- Mask Logic ---
            mask = get_rhythm_mask(
                energy_smooth, fs, 
                interval_ms=config['rhythm_interval_ms'],
                window_ms=config['rhythm_window_ms'],
                noise_cv_threshold=0.2
            )
            
            # --- Active Segments Processing ---
            labeled, num_seg = ndimage.label(mask)
            
            for seg_idx in range(1, num_seg + 1):
                loc = np.where(labeled == seg_idx)[0]
            
                # 确保形状正确 (双重保险)
                # 1. 找到该动作片段在原始数据中的中心点
                center_idx = loc[0] + len(loc) // 2
                
                # 2. 计算需要的起始和结束位置 (在原始长数据 data_clean 中)
                half_win = win_size // 2
                w_start = center_idx - half_win
                w_end = w_start + win_size
                
                # 3. 边界处理 (如果动作刚好在文件开头或结尾)
                pad_left = 0
                pad_right = 0
                
                if w_start < 0:
                    pad_left = -w_start # 需要在左边补多少0
                    w_start = 0
                if w_end > len(data_clean):
                    pad_right = w_end - len(data_clean) # 需要在右边补多少0
                    w_end = len(data_clean)
                
                # 4. 从原始数据中截取 (这样就自动包含了动作周围的静息数据，补足了时长)
                seg_data = data_clean[w_start : w_end]
                
                # 5. 如果碰到文件边缘导致长度不够，进行零填充 (Padding)
                if pad_left > 0 or pad_right > 0:
                    # ((pad_left, pad_right), (0, 0)) 表示只在时间维度(行)前后补零，通道维度(列)不补
                    seg_data = np.pad(seg_data, ((pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
                
                # Z-Score Norm (Per segment)
                seg_mean = np.mean(seg_data, axis=0)
                seg_std = np.std(seg_data, axis=0)
                seg_norm = (seg_data - seg_mean) / (seg_std + 1e-6)
                
                # 截取唯一窗口
                window = seg_norm
                
                # Original
                X_list.append(window)
                current_file_act_count += 1
                y_list.append(label)
                groups_list.append(f"{fname}_seg{seg_idx}")
                total_act_samples += 1
                
                # Augmentation
                for _ in range(multiplier - 1):
                    aug_win = window.copy()
                    if augment_config.get('enable_warp', False) and np.random.random() > 0.5:
                        aug_win = time_warp(aug_win)
                    if augment_config.get('enable_shift', False) and np.random.random() > 0.5:
                        aug_win = time_shift(aug_win) # 这里的 shift 是 roll，不改变长度，仍然适用
                    if augment_config.get('enable_scaling', True) and np.random.random() > 0.3:
                            aug_win *= np.random.uniform(0.8, 1.2)
                    if augment_config.get('enable_mask', False) and np.random.random() > 0.7:
                        aug_win = channel_mask(aug_win)
                    if augment_config.get('enable_noise', True):
                        aug_win += np.random.normal(0, 0.02, aug_win.shape)

                    X_list.append(aug_win)
                    y_list.append(label)
                    groups_list.append(f"{fname}_seg{seg_idx}")

            # --- Rest (Silence) Processing (修改点 2: 随机抽取) ---
            if enable_rest:
                noise_floor = np.percentile(energy_smooth, 10)
                peak_level = np.percentile(energy_smooth, 99)
                vad_threshold = noise_floor + 0.15 * (peak_level - noise_floor)
                vad_mask = energy_smooth > vad_threshold
                rest_mask = ~vad_mask
                
                safe_margin = int(0.15 * fs)
                rest_mask = ndimage.binary_erosion(rest_mask, structure=np.ones(safe_margin))
                
                labeled_rest, num_rest = ndimage.label(rest_mask)
                
                # 目标：静息样本数量为动作样本的 20%
                target_rest = int(current_file_act_count * 0.2) + 2
                
                # 收集所有足够长的静息段
                valid_rest_segments = []
                for r_idx in range(1, num_rest + 1):
                    r_loc = np.where(labeled_rest == r_idx)[0]
                    if len(r_loc) > win_size:
                        valid_rest_segments.append(data_clean[r_loc[0]:r_loc[-1]])
                
                # 从这些段中随机截取 target_rest 个样本
                collected_rest = 0
                retries = 0
                max_retries = target_rest * 2 # 防止死循环
                
                if valid_rest_segments:
                    while collected_rest < target_rest and retries < max_retries:
                        # 随机选一个段
                        seg = valid_rest_segments[np.random.randint(len(valid_rest_segments))]
                        if len(seg) <= win_size:
                            retries += 1
                            continue
                            
                        # 随机选一个起始点
                        max_start = len(seg) - win_size
                        if max_start <= 0: start_idx = 0
                        else: start_idx = np.random.randint(0, max_start)
                        
                        r_raw_win = seg[start_idx : start_idx + win_size]
                        
                        # Norm
                        r_mean = np.mean(r_raw_win, axis=0)
                        r_std = np.std(r_raw_win, axis=0)
                        r_std = np.where(r_std < 0.01, 1.0, r_std)
                        r_norm = (r_raw_win - r_mean) / (r_std + 1e-6)
                        
                        X_list.append(r_norm)
                        y_list.append(0)
                        groups_list.append(f"{fname}_rest_{collected_rest}")
                        collected_rest += 1
                        retries += 1

        except Exception as e:
            print(f"Error reading {f_path}: {e}")
            import traceback
            traceback.print_exc()
            
    print(f"✅ 处理完成: 样本数 {len(X_list)}")
    return np.array(X_list), np.array(y_list), np.array(groups_list)

# ==================== 3. 辅助类 (Mock Streamlit) ====================
class MockProgressBar:
    def progress(self, value): pass

class MockStatusText:
    def text(self, msg): print(f"    └─ {msg}")

# ==================== 4. 主程序 ====================

def run_automation():
    # 1. 查找所有目标文件
    search_pattern = os.path.join("data", "*", "*", "RAW_EMG*.csv")
    all_files = glob.glob(search_pattern)
    target_files = []
    
    print(f"🔍 正在筛选文件...")
    for f in all_files:
        s, d, l, _ = parse_filename_info(f)
        if s not in TARGET_SUBJECTS: continue
        if TARGET_DATES and d not in TARGET_DATES: continue
        if l not in TARGET_LABELS: continue
        target_files.append(f)
        
    if not target_files:
        print("❌ 未找到匹配文件")
        return

    # === 关键修改步骤 1: 加载所有数据 (Clean Load) ===
    print(f"⏳ 正在加载所有数据 (共 {len(target_files)} 个文件)...")
    
    # 创建一个“纯净”的配置，强制 multiplier=1，关闭所有增强
    # 这样我们只加载最原始的数据
    clean_config = AUGMENT_CONFIG.copy()
    clean_config['multiplier'] = 1
    clean_config['enable_noise'] = False
    clean_config['enable_warp'] = False
    clean_config['enable_shift'] = False
    clean_config['enable_mask'] = False
    clean_config['enable_scaling'] = False
    # 注意：enable_rest 保持原样，因为我们需要静息数据
    
    X_all, y_all, groups_all = process_files_with_rhythm(
        target_files, CONFIG, clean_config
    )
    
    if len(X_all) == 0:
        print("❌ 加载数据为空，退出。")
        return

    print(f"✅ 原始数据加载完毕: {X_all.shape}")

    # === 关键修改步骤 2: 样本级混合切分 (GroupShuffleSplit) ===
    # GroupShuffleSplit 会保证同一个 Group (即同一个动作的所有切片) 不会被拆分到 train 和 test
    # 由于我们还没有增强，现在的 Group 就是原始动作 ID (e.g., "filename_seg1")
    
    print(f"✂️ 正在执行混合切分 (Test Size: {CONFIG['test_size']})...")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=CONFIG['test_size'], random_state=42)
    train_idx, test_idx = next(gss.split(X_all, y_all, groups=groups_all))
    
    X_train_raw, y_train_raw = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]
    
    # 保存训练集的 Groups 信息，稍后增强时要用
    groups_train_raw = groups_all[train_idx]
    
    print(f"   Train (Raw): {X_train_raw.shape}")
    print(f"   Test  (Clean): {X_test.shape}")

    # === 关键修改步骤 3: 仅对训练集进行内存增强 ===
    # 此时使用全局定义的 AUGMENT_CONFIG (里面包含了 multiplier, noise 等设置)
    
    if AUGMENT_CONFIG.get('multiplier', 1) > 1 or AUGMENT_CONFIG.get('enable_noise', False):
        print("🚀 正在对训练集应用数据增强...")
        X_train, y_train, groups_train = augment_dataset_in_memory(
            X_train_raw, y_train_raw, groups_train_raw, AUGMENT_CONFIG
        )
    else:
        X_train, y_train, groups_train = X_train_raw, y_train_raw, groups_train_raw
    
    # 3. 标签映射 (Label Mapping)
    all_labels = np.unique(np.concatenate([y_train, y_test]))
    label_map = {original: new for new, original in enumerate(all_labels)}
    
    y_train_mapped = np.array([label_map[i] for i in y_train])
    y_test_mapped = np.array([label_map[i] for i in y_test])
    
    print(f"📊 最终数据集规模 (Ready for Training):")
    print(f"   Train: {X_train.shape} [Augmented]")
    print(f"   Test:  {X_test.shape} [Clean]")
    print(f"   Labels: {label_map}")
    # 4. 训练循环
    MODELS_DIR = "1.26_trained_models_rhythm_wihoutstride"
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
    
    total_exp = len(MODELS_TO_TEST) * len(OPTIMIZERS_TO_TEST) * len(VOTING_OPTIONS)
    curr_exp = 0
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(label_map)
    
    for model_name, model_builder in MODELS_TO_TEST:
        for opt_name, opt_cls, lr, opt_kwargs in OPTIMIZERS_TO_TEST:
            if opt_name == "SGD":
                current_epochs = 200
            else:
                current_epochs = CONFIG['epochs']
            for use_vote in VOTING_OPTIONS:
                curr_exp += 1
                exp_id = f"{model_name}_{opt_name}_Vote{use_vote}"
                print(f"\n🚀 [{curr_exp}/{total_exp}] Start: {exp_id}")
                
                tf.keras.backend.clear_session()
                model = model_builder(input_shape, num_classes)
                optimizer = opt_cls(learning_rate=lr, **opt_kwargs)
                
                # 开始训练
                start_t = time.time()
                try:
                    history = train_utils.train_with_voting_mechanism(
                        model, X_train, y_train_mapped, groups_train,
                        X_test, y_test_mapped,
                        epochs=current_epochs,
                        batch_size=CONFIG['batch_size'],
                        samples_per_group=3,
                        vote_weight=0.5 if use_vote else 0.0,
                        st_progress_bar=MockProgressBar(),
                        st_status_text=MockStatusText(),
                        voting_start_epoch=25,
                        optimizer=optimizer
                    )
                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue
                    
                duration = time.time() - start_t
                
                # 保存
                model.save(os.path.join(MODELS_DIR, f"{exp_id}.keras"))
                
                # 记录
                save_log(
                    exp_id, 
                    model, 
                    history, 
                    X_test, 
                    y_test_mapped, 
                    label_map, 
                    duration,
                    opt_name,       # 传入优化器名称
                    lr,             # 传入学习率
                    use_vote,       # 传入是否投票
                    current_epochs  # 动态判断的 Epoch 数
                )

def save_log(exp_id, model, history, X_test, y_test, label_map, duration, opt_name, lr, use_voting, actual_epochs):
    # 1. 计算预测报告
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    # 获取详细字典格式报告，转为 DataFrame 以便美观打印
    report_dict = classification_report(
        y_test, y_pred, 
        target_names=[str(k) for k in label_map.keys()],
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()
    
    # 2. 准备文件名
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    log_file = os.path.join(LOG_DIR, f"{timestamp}_{exp_id}.txt")
    
    final_acc = history['val_accuracy'][-1]
    final_loss = history['val_loss'][-1]

    # 3. 写入详细信息 (这部分是原版 auto_train.py 的精华)
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Experiment ID: {exp_id}\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Duration: {duration:.1f}s\n")
        f.write("="*40 + "\n")
        f.write(f"Subjects: {TARGET_SUBJECTS}\n")
        f.write(f"Labels: {TARGET_LABELS}\n")
        f.write(f"Model: {model.name}\n")
        f.write(f"Optimizer: {opt_name} (LR={lr})\n")
        f.write(f"Epochs: {actual_epochs}\n")  # [关键] 记录实际跑了多少轮
        f.write(f"Voting Mode: {'ON' if use_voting else 'OFF'}\n")
        f.write("-" * 20 + "\n")
        f.write(f"Batch Size: {CONFIG['batch_size']}\n")
        f.write(f"Augment Config: {AUGMENT_CONFIG}\n") # 记录增强配置
        f.write("="*40 + "\n")
        f.write(f"Final Val Accuracy: {final_acc*100:.2f}%\n")
        f.write(f"Final Val Loss: {final_loss:.4f}\n")
        f.write("\n--- Classification Report ---\n")
        f.write(report_df.to_string())

    print(f"💾 Detailed Log saved: {log_file}")

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus: 
        try: tf.config.experimental.set_memory_growth(gpus[0], True)
        except: pass
    run_automation()