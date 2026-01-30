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
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit

# === PyTorch Imports ===
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# === 自定义模块 ===
# 假设 sparse_model.py 和 jelly_cht.py 在同一目录下
import sparse_model 
import jelly_cht

# ==================== 0. 配置区域 ====================

# 1. 目标设置
TARGET_SUBJECTS = ["charles", "gavvin", "gerard", "giland", "jessie", "legend"] 
TARGET_LABELS = [5, 6, 7, 8]  # 指定动作标签
TARGET_DATES = None        # None 表示所有日期

# 2. 训练配置
CONFIG = {
    'fs': 1000,
    'use_imu': True,
    'rhythm_interval_ms': 4000,
    'rhythm_window_ms': 352,
    'epochs': 100,             # CHT 稀疏训练通常需要较多轮次来演化
    'batch_size': 64,          # PyTorch 常用 batch size
    'window_ms': 350,
    'test_size': 0.2,
    'learning_rate': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# 3. CHT 稀疏训练配置
CHT_CONFIG = {
    'enable': True,                # 是否开启稀疏训练
    'sparsity': 0.5,               # 目标稀疏度 (例如 0.5 表示去掉 50% 的连接)
    'remove_method': 'weight_magnitude_soft', # 剪枝策略: weight_magnitude, weight_magnitude_soft, ri, ri_soft
    'regrow_method': 'cht',        # 生长策略: cht, SET
    'zeta': 0.1,                   # 剪枝比例
    'non_sparse_layers': ['init_conv', 'fc'], # 不进行稀疏化的层名称 (通常保留第一层和最后一层)
}

# 4. 数据增强
AUGMENT_CONFIG = {
    'enable_rest': True,
    'multiplier': 3,
    'enable_scaling': True,
    'enable_noise': True,
    'enable_warp': False,
    'enable_shift': False,
    'enable_mask': False
}

LOG_DIR = "1.26_auto_train_logs_pytorch_tcn"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# ==================== 1. 数据增强工具函数 (保持不变) ====================

def time_warp(data, sigma=0.2, knot=4):
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
    shift_amt = int(data.shape[0] * shift_limit * np.random.uniform(-1, 1))
    return np.roll(data, shift_amt, axis=0)

def channel_mask(data, mask_prob=0.15):
    temp = data.copy()
    if np.random.random() < mask_prob:
        c = np.random.randint(0, data.shape[1])
        temp[:, c] = 0
    return temp

def add_noise(data, noise_level=0.02):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def scale_amplitude(data, scale_range=(0.8, 1.2)):
    factor = np.random.uniform(scale_range[0], scale_range[1])
    return data * factor

def load_and_resample_imu(emg_filepath, target_length):
    imu_filepath = emg_filepath.replace("RAW_EMG", "RAW_IMU")
    if not os.path.exists(imu_filepath): return None
    try:
        df = pd.read_csv(imu_filepath)
        required_cols = ['AX', 'AY', 'AZ', 'GX', 'GY', 'GZ']
        if not all(col in df.columns for col in required_cols): return None
        imu_data = df[required_cols].values 
        x_old = np.linspace(0, 1, len(imu_data))
        x_new = np.linspace(0, 1, target_length)
        imu_resampled = np.zeros((target_length, 6))
        for i in range(6):
            imu_resampled[:, i] = np.interp(x_new, x_old, imu_data[:, i])
        return imu_resampled
    except: return None

def augment_dataset_in_memory(X, y, groups, config):
    multiplier = config.get('multiplier', 1)
    if multiplier <= 1: return X, y, groups
    print(f"    ⚡ 正在执行内存增强 (倍率: {multiplier}x)...")
    X_aug, y_aug, g_aug = [], [], []
    total = len(X)
    for i in range(total):
        X_aug.append(X[i])
        y_aug.append(y[i])
        g_aug.append(groups[i])
        for _ in range(multiplier - 1):
            aug_x = X[i].copy()
            if config.get('enable_warp') and np.random.random() > 0.5: aug_x = time_warp(aug_x)
            if config.get('enable_shift') and np.random.random() > 0.5: aug_x = time_shift(aug_x)
            if config.get('enable_scaling') and np.random.random() > 0.3: aug_x = scale_amplitude(aug_x)
            if config.get('enable_mask') and np.random.random() > 0.7: aug_x = channel_mask(aug_x)
            if config.get('enable_noise'): aug_x = add_noise(aug_x)
            X_aug.append(aug_x)
            y_aug.append(y[i])
            g_aug.append(groups[i])
    return np.array(X_aug, dtype=np.float32), np.array(y_aug), np.array(g_aug)

# ==================== 2. 数据处理核心 (保持不变) ====================

def parse_filename_info(filepath):
    filename = os.path.basename(filepath)
    parts = filepath.split(os.sep)
    subject = parts[-3] if len(parts) >= 3 else "Unknown"
    date = parts[-2] if len(parts) >= 3 else "Unknown"
    label_match = re.search(r'DF(\d+)\.', filename)
    label = int(label_match.group(1)) if label_match else None
    return subject, date, label, filename

def get_rhythm_mask(energy, fs, interval_ms=4000, window_ms=300, noise_cv_threshold=0.2):
    mask = np.zeros_like(energy, dtype=bool)
    min_dist = int(2.0 * fs) 
    noise_floor = np.percentile(energy, 10)
    peaks, _ = signal.find_peaks(energy, distance=min_dist, height=noise_floor * 1.5)
    if len(peaks) == 0: return mask
    
    interval_samples = int((interval_ms / 1000) * fs)
    if interval_samples < 1: interval_samples = 1
    phases = peaks % interval_samples
    bin_width = int(0.2 * fs)
    bins = np.arange(0, interval_samples + bin_width, bin_width)
    counts, bin_edges = np.histogram(phases, bins=bins)
    best_bin_idx = np.argmax(counts)
    phase_start, phase_end = bin_edges[best_bin_idx], bin_edges[best_bin_idx+1]
    
    candidates = peaks[(phases >= phase_start) & (phases < phase_end)]
    anchor_peak = candidates[np.argmax(energy[candidates])] if len(candidates) > 0 else peaks[0]

    half_win = int((window_ms / 1000) * fs) // 2
    search_radius = int(1.0 * fs)
    valid_centers = []
    max_len = len(energy)
    
    for direction in [1, -1]:
        curr_grid = anchor_peak if direction == 1 else anchor_peak - interval_samples
        while 0 <= curr_grid < max_len:
            s_start = max(0, curr_grid - search_radius)
            s_end = min(max_len, curr_grid + search_radius)
            region = energy[s_start:s_end]
            if len(region) > 0:
                abs_center = s_start + np.argmax(region)
                if energy[abs_center] > noise_floor * 1.2: valid_centers.append(abs_center)
            if direction == 1: curr_grid += interval_samples
            else: curr_grid -= interval_samples

    valid_centers = sorted(list(set(valid_centers)))
    for c in valid_centers:
        s = max(0, c - half_win)
        e = min(max_len, c + half_win)
        seg_vals = energy[s:e]
        if len(seg_vals) == 0: continue
        cv = np.std(seg_vals) / (np.mean(seg_vals) + 1e-6)
        if np.mean(seg_vals) > energy[anchor_peak] * 0.3 and cv < noise_cv_threshold: continue
        mask[s:e] = True
    return mask

def process_files(file_list, config, augment_config):
    X_list, y_list, groups_list = [], [], []
    fs = config['fs']
    win_size = int(fs * (config['window_ms'] / 1000))
    use_imu = config.get('use_imu', False)
    
    print(f"⏳ 正在处理 {len(file_list)} 个文件...")
    
    for f_path in file_list:
        try:
            subject, date, label, fname = parse_filename_info(f_path)
            if label is None: continue
            
            df = pd.read_csv(f_path)
            cols = [c for c in df.columns if 'CH' in c]
            raw_emg = df[cols].values
            if raw_emg.shape[1] >= 5: raw_emg[:, 4] *= 2.5
            
            if use_imu:
                imu_data = load_and_resample_imu(f_path, len(raw_emg))
                if imu_data is not None: raw_data = np.hstack((raw_emg, imu_data))
                else: continue
            else: raw_data = raw_emg
            
            emg_cols = raw_emg.shape[1]
            data_proc = raw_data.copy()
            b_n, a_n = signal.iirnotch(50, 30, fs)
            data_proc[:, :emg_cols] = signal.filtfilt(b_n, a_n, data_proc[:, :emg_cols], axis=0)
            b, a = signal.butter(4, [20, 450], btype='bandpass', fs=fs)
            data_proc[:, :emg_cols] = signal.filtfilt(b, a, data_proc[:, :emg_cols], axis=0)
            data_clean = data_proc

            energy = np.sqrt(np.mean(data_clean[:, :emg_cols]**2, axis=1))
            energy_smooth = np.convolve(energy, np.ones(int(0.1*fs))/int(0.1*fs), mode='same')
            
            mask = get_rhythm_mask(energy_smooth, fs, config['rhythm_interval_ms'], config['rhythm_window_ms'])
            labeled, num_seg = ndimage.label(mask)
            
            current_act_count = 0
            for seg_idx in range(1, num_seg + 1):
                loc = np.where(labeled == seg_idx)[0]
                center_idx = loc[0] + len(loc) // 2
                w_start = center_idx - win_size // 2
                w_end = w_start + win_size
                
                # Padding
                pad_left, pad_right = 0, 0
                if w_start < 0: pad_left, w_start = -w_start, 0
                if w_end > len(data_clean): pad_right, w_end = w_end - len(data_clean), len(data_clean)
                
                seg_data = data_clean[w_start : w_end]
                if pad_left > 0 or pad_right > 0:
                    seg_data = np.pad(seg_data, ((pad_left, pad_right), (0, 0)), mode='constant')
                
                # Z-Score Norm
                seg_norm = (seg_data - np.mean(seg_data, axis=0)) / (np.std(seg_data, axis=0) + 1e-6)
                
                X_list.append(seg_norm)
                y_list.append(label)
                groups_list.append(f"{fname}_seg{seg_idx}")
                current_act_count += 1
            
            # Rest Processing
            if augment_config.get('enable_rest', True):
                noise_floor = np.percentile(energy_smooth, 10)
                peak_level = np.percentile(energy_smooth, 99)
                rest_mask = ~(energy_smooth > (noise_floor + 0.15*(peak_level-noise_floor)))
                rest_mask = ndimage.binary_erosion(rest_mask, structure=np.ones(int(0.15*fs)))
                labeled_rest, num_rest = ndimage.label(rest_mask)
                
                valid_segs = []
                for r in range(1, num_rest+1):
                    loc = np.where(labeled_rest == r)[0]
                    if len(loc) > win_size: valid_segs.append(data_clean[loc[0]:loc[-1]])
                
                target_rest = int(current_act_count * 0.2) + 2
                collected = 0
                tries = 0
                if valid_segs:
                    while collected < target_rest and tries < target_rest * 2:
                        seg = valid_segs[np.random.randint(len(valid_segs))]
                        if len(seg) <= win_size: 
                            tries+=1; continue
                        start = np.random.randint(0, len(seg) - win_size)
                        r_win = seg[start:start+win_size]
                        r_norm = (r_win - np.mean(r_win, axis=0)) / (np.std(r_win, axis=0) + 1e-6)
                        X_list.append(r_norm)
                        y_list.append(0)
                        groups_list.append(f"{fname}_rest_{collected}")
                        collected += 1
                        tries += 1

        except Exception as e:
            print(f"Error {f_path}: {e}")
            
    return np.array(X_list), np.array(y_list), np.array(groups_list)

# ==================== 3. 训练流程 (PyTorch) ====================

def run_training_loop(model, train_loader, test_loader, config, label_map):
    device = torch.device(config['device'])
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # === 初始化 CHT 稀疏训练器 ===
    cht = None
    if CHT_CONFIG['enable']:
        print("⚡ 初始化 CHT 稀疏训练器...")
        cht = jelly_cht.CHT(
            model=model,
            sparsity=CHT_CONFIG['sparsity'],
            remove_method=CHT_CONFIG['remove_method'],
            regrow_method=CHT_CONFIG['regrow_method'],
            mum_epoch=config['epochs'],
            zeta=CHT_CONFIG['zeta'],
            chain_removal_list=[], # 暂无特定链式约束
            non_sparse_layer_list=CHT_CONFIG['non_sparse_layers'],
            device=device
        )
    
    print(f"🚀 开始训练 (Device: {device})")
    start_time = time.time()
    
    best_acc = 0.0
    
    for epoch in range(config['epochs']):
        # --- Training ---
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        train_loss /= total
        train_acc = correct / total
        
        # --- CHT 权重演化 ---
        if cht:
            cht.weight_evolution(epoch)
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        if val_acc > best_acc:
            best_acc = val_acc
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(LOG_DIR, "best_model.pth"))
            
        print(f"Epoch [{epoch+1}/{config['epochs']}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    duration = time.time() - start_time
    print(f"\n✅ 训练完成，耗时: {duration:.1f}s")
    
    # 生成最终报告
    report = classification_report(
        all_targets, all_preds, 
        target_names=[str(k) for k in label_map.keys()],
        output_dict=True
    )
    return report, duration

def save_log(exp_id, report_dict, duration, config):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filename = os.path.join(LOG_DIR, f"{timestamp}_{exp_id}.txt")
    
    report_df = pd.DataFrame(report_dict).transpose()
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Experiment ID: {exp_id}\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Duration: {duration:.1f} s\n")
        f.write("="*40 + "\n")
        f.write(f"Subjects: {TARGET_SUBJECTS}\n")
        f.write(f"Labels: {TARGET_LABELS}\n")
        f.write(f"Config: {config}\n")
        f.write(f"CHT Config: {CHT_CONFIG}\n")
        f.write("="*40 + "\n")
        f.write(f"Final Accuracy: {report_dict['accuracy']:.4f}\n")
        f.write("\n--- Classification Report ---\n")
        f.write(report_df.to_string())
    
    print(f"💾 日志已保存: {filename}")

def run_automation():
    # 1. 查找文件
    search_pattern = os.path.join("data", "*", "*", "RAW_EMG*.csv")
    all_files = glob.glob(search_pattern)
    target_files = []
    
    print("🔍 筛选文件中...")
    for f in all_files:
        s, d, l, _ = parse_filename_info(f)
        if s in TARGET_SUBJECTS and (TARGET_DATES is None or d in TARGET_DATES) and l in TARGET_LABELS:
            target_files.append(f)
            
    if not target_files:
        print("❌ 未找到匹配文件")
        return

    # 2. 加载原始数据
    clean_config = AUGMENT_CONFIG.copy()
    clean_config['multiplier'] = 1
    # 关闭增强以获取原始数据
    for k in ['enable_noise', 'enable_warp', 'enable_shift', 'enable_mask', 'enable_scaling']:
        clean_config[k] = False
        
    X_all, y_all, groups_all = process_files(target_files, CONFIG, clean_config)
    print(f"✅ 原始数据: {X_all.shape}")

    # 3. 数据切分
    gss = GroupShuffleSplit(n_splits=1, test_size=CONFIG['test_size'], random_state=42)
    train_idx, test_idx = next(gss.split(X_all, y_all, groups=groups_all))
    
    X_train_raw, y_train_raw = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]
    groups_train = groups_all[train_idx]
    
    # 4. 对训练集进行增强
    if AUGMENT_CONFIG['multiplier'] > 1:
        X_train, y_train, _ = augment_dataset_in_memory(X_train_raw, y_train_raw, groups_train, AUGMENT_CONFIG)
    else:
        X_train, y_train = X_train_raw, y_train_raw
        
    print(f"📊 训练集 (Augmented): {X_train.shape}, 测试集: {X_test.shape}")
    
    # 5. 标签映射
    unique_labels = np.unique(np.concatenate([y_train, y_test]))
    label_map = {orig: new for new, orig in enumerate(unique_labels)}
    y_train_mapped = np.array([label_map[y] for y in y_train])
    y_test_mapped = np.array([label_map[y] for y in y_test])
    
    # 6. 转为 PyTorch Tensor 并创建 DataLoader
    # PyTorch Conv1d 需要 (Batch, Channels, Length)，原始数据是 (Batch, Length, Channels)
    # 需要 transpose(1, 2)
    X_train_tensor = torch.FloatTensor(X_train).permute(0, 2, 1) 
    y_train_tensor = torch.LongTensor(y_train_mapped)
    X_test_tensor = torch.FloatTensor(X_test).permute(0, 2, 1)
    y_test_tensor = torch.LongTensor(y_test_mapped)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # 7. 模型初始化
    num_classes = len(label_map)
    input_channels = X_train.shape[2] # 原始数据的最后一个维度是通道数
    
    print(f"🔧 初始化 TCN 模型 (In: {input_channels}, Out: {num_classes})...")
    model = sparse_model.TCNModel(input_channels=input_channels, num_classes=num_classes)
    
    # 8. 运行训练
    exp_id = f"PyTorch_TCN_CHT_{'On' if CHT_CONFIG['enable'] else 'Off'}"
    report, duration = run_training_loop(model, train_loader, test_loader, CONFIG, label_map)
    
    save_log(exp_id, report, duration, CONFIG)

if __name__ == "__main__":
    run_automation()