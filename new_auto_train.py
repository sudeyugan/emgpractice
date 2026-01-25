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

# ==================== 0. 配置区域 ====================

# 1. 目标设置
TARGET_SUBJECTS = ["charles", "gavvin", "gerard", "giland", "jessie", "legend"] 
TARGET_LABELS = [1, 3, 4, 5, 6, 7, 8, 9, 15]            # 指定动作标签
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
    'rhythm_interval_ms': 4000,# [关键] 动作间隔 (节拍器速度)
    'rhythm_window_ms': 350,   # [关键] 每次截取的窗口大小 (以峰值为中心)
    'epochs': 100,
    'batch_size': 128,
    'window_ms': 250,          # 输入模型的窗口大小
    'stride_ms': 50,           # 切片步长
    'test_size': 0.2,
    'split_strategy': "混合切分 (看到所有天/人)", 
}

# 4. 数据增强
AUGMENT_CONFIG = {
    'enable_rest': True,       # 是否采集静息数据 (Label 0)
    'multiplier': 1,           # 数据倍增系数
    'enable_scaling': True,
    'enable_noise': True,
    'enable_warp': False,      # 时间扭曲 (耗时，视情况开启)
    'enable_shift': False,
    'enable_mask': False
}

LOG_DIR = "1.24_9_auto_train_logs_rhythm"
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
    使用固定节奏逻辑处理文件列表
    """
    X_list, y_list, groups_list = [], [], []
    
    fs = config['fs']
    win_size = int(fs * (config['window_ms'] / 1000))
    stride = int(fs * (config['stride_ms'] / 1000))
    
    # 增强参数
    multiplier = augment_config.get('multiplier', 1)
    enable_rest = augment_config.get('enable_rest', True)
    
    # 用于计算静息样本比例
    total_act_samples = 0
    
    print(f"⏳ 正在处理 {len(file_list)} 个文件 (Mode: Rhythm Phase Voting)...")
    
    for i, f_path in enumerate(file_list):
        try:
            subject, date, label, fname = parse_filename_info(f_path)
            if label is None: continue
            
            # --- Load ---
            df = pd.read_csv(f_path)
            cols = [c for c in df.columns if 'CH' in c]
            raw_data = df[cols].values
            if raw_data.shape[1] >= 5: # CH5 fix
                raw_data[:, 4] = raw_data[:, 4] * 2.5
                
            # --- Filter Chain ---
            # 1. Notch
            b_notch, a_notch = signal.iirnotch(50, 30, fs)
            data_notch = signal.filtfilt(b_notch, a_notch, raw_data, axis=0)
            
            # 2. Bandpass
            b, a = signal.butter(4, [20, 450], btype='bandpass', fs=fs)
            data_clean = signal.filtfilt(b, a, data_notch, axis=0)
            
            # 3. Energy & Smooth (for Masking)
            energy = np.sqrt(np.mean(data_clean**2, axis=1))
            win_len = int(0.1 * fs) # 100ms smooth
            energy_smooth = np.convolve(energy, np.ones(win_len)/win_len, mode='same')
            
            # --- New Core Logic ---
            mask = get_rhythm_mask(
                energy_smooth, fs, 
                interval_ms=config['rhythm_interval_ms'],
                window_ms=config['rhythm_window_ms'],
                noise_cv_threshold=0.2
            )
            
            # --- Slicing Active Segments ---
            labeled, num_seg = ndimage.label(mask)
            
            for seg_idx in range(1, num_seg + 1):
                loc = np.where(labeled == seg_idx)[0]
                if len(loc) < win_size: continue # 太短的不够切
                
                seg_data = data_clean[loc[0]:loc[-1]]
                
                # Z-Score Norm (Per segment)
                seg_mean = np.mean(seg_data, axis=0)
                seg_std = np.std(seg_data, axis=0)
                seg_norm = (seg_data - seg_mean) / (seg_std + 1e-6)
                
                # Sliding Window
                for w_start in range(0, len(seg_norm) - win_size + 1, stride):
                    window = seg_norm[w_start : w_start + win_size]
                    
                    # Original
                    X_list.append(window)
                    y_list.append(label)
                    groups_list.append(f"{fname}_seg{seg_idx}")
                    total_act_samples += 1
                    
                    # Augmentation (Simple Noise/Scale for now)
                    for _ in range(multiplier - 1):
                        aug_win = window.copy()

                        if augment_config.get('enable_warp', False) and np.random.random() > 0.5:
                            aug_win = time_warp(aug_win)
                            
                        # 2. 时间平移
                        if augment_config.get('enable_shift', False) and np.random.random() > 0.5:
                            aug_win = time_shift(aug_win)
                            
                        # 3. 幅度缩放
                        if augment_config.get('enable_scaling', True) and np.random.random() > 0.3:
                             aug_win *= np.random.uniform(0.8, 1.2)
                             
                        # 4. 通道遮挡
                        if augment_config.get('enable_mask', False) and np.random.random() > 0.7:
                            aug_win = channel_mask(aug_win)
                            
                        # 5. 高斯噪声 (通常最后加)
                        if augment_config.get('enable_noise', True):
                            aug_win += np.random.normal(0, 0.02, aug_win.shape)

                        X_list.append(aug_win)
                        y_list.append(label)
                        groups_list.append(f"{fname}_seg{seg_idx}_aug")

            # --- Slicing Rest (Silence) ---
            if enable_rest:
                # [修改] 改回使用能量阈值定义静息，而不是节奏 Mask 的补集
                # 1. 重新计算一个宽泛的 VAD Mask (基于能量)
                noise_floor = np.percentile(energy_smooth, 10)
                peak_level = np.percentile(energy_smooth, 99)
                # 阈值系数 0.15 是经验值，与 GUI 保持一致
                vad_threshold = noise_floor + 0.15 * (peak_level - noise_floor)
                
                vad_mask = energy_smooth > vad_threshold
                
                # 2. 对 VAD Mask 取反，得到静息区
                rest_mask = ~vad_mask
                
                # 3. 腐蚀 (Erosion) 确保远离动作边缘 (这里保持原样或稍微调大一点 margin)
                safe_margin = int(0.15 * fs) # 150ms margin
                rest_mask = ndimage.binary_erosion(rest_mask, structure=np.ones(safe_margin))
                
                # 后面的逻辑保持不变...
                labeled_rest, num_rest = ndimage.label(rest_mask)
                # 随机抽取一定数量的 Rest，避免过多
                target_rest = int(total_act_samples * 0.2) + 2 # 每文件约 20%
                
                rest_buffer = []
                for r_idx in range(1, num_rest + 1):
                    r_loc = np.where(labeled_rest == r_idx)[0]
                    if len(r_loc) > win_size:
                        r_data = data_clean[r_loc[0]:r_loc[-1]]
                        # Norm
                        r_mean = np.mean(r_data, axis=0)
                        r_std = np.std(r_data, axis=0)
                        r_std = np.where(r_std < 0.01, 1.0, r_std)
                        r_norm = (r_data - r_mean) / (r_std + 1e-6)
                        
                        # Big Stride for rest
                        for w_s in range(0, len(r_norm) - win_size, win_size):
                            rest_buffer.append(r_norm[w_s:w_s+win_size])
                            
                if rest_buffer:
                    np.random.shuffle(rest_buffer)
                    for rw in rest_buffer[:target_rest]:
                        X_list.append(rw)
                        y_list.append(0) # Label 0
                        groups_list.append(f"{fname}_rest")

        except Exception as e:
            print(f"Error reading {f_path}: {e}")
            
    print(f"✅ 处理完成: 样本数 {len(X_list)}")
    return np.array(X_list), np.array(y_list), np.array(groups_list)

# ==================== 3. 辅助类 (Mock Streamlit) ====================
class MockProgressBar:
    def progress(self, value): pass

class MockStatusText:
    def text(self, msg): print(f"    └─ {msg}")

# ==================== 4. 主程序 ====================

def run_automation():
    # 1. 查找文件
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

    # 2. 生成数据 (使用新函数)
    X, y, groups = process_files_with_rhythm(target_files, CONFIG, AUGMENT_CONFIG)
    
    if len(X) == 0:
        print("❌ 样本数为0，请检查 rhythm_interval 或 文件的能量是否过低")
        return
        
    # 3. 准备数据
    # 映射 Label: 5,6,7,8 -> 1,2,3,4 (0预留给Rest)
    unique_labels = sorted(np.unique(y))
    label_map = {original: new for new, original in enumerate(unique_labels)}
    y_mapped = np.array([label_map[i] for i in y])
    
    # Split
    train_idx, test_idx = train_utils.smart_split(
        X, y_mapped, groups, CONFIG['split_strategy'], test_size=CONFIG['test_size']
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_mapped[train_idx], y_mapped[test_idx]
    groups_train = groups[train_idx]
    
    print(f"📊 训练集: {X_train.shape}, 测试集: {X_test.shape}, 类别数: {len(unique_labels)}")
    print(f"   Label Map: {label_map}")

    # 4. 训练循环
    MODELS_DIR = "1.24_9_trained_models_rhythm"
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
    
    total_exp = len(MODELS_TO_TEST) * len(OPTIMIZERS_TO_TEST) * len(VOTING_OPTIONS)
    curr_exp = 0
    
    input_shape = (X.shape[1], X.shape[2])
    num_classes = len(unique_labels)
    
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
                        model, X_train, y_train, groups_train,
                        X_test, y_test,
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
                    y_test, 
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