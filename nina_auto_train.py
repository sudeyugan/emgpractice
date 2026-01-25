import scipy.io as sio
import os
import sys
import time
import glob
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
import scipy.ndimage as ndimage
import gc

# 引用现有模块
import train_utils
import nina_model as model_lib  # 避免变量名冲突

# ==================== 0. 配置区域 (根据需求修改) ====================

# 1. 目标设置
TARGET_SUBJECTS = [f"s{i}" for i in range(1, 39)]  
TARGET_LABELS = [1, 2, 3, 4, 5, 6, 7, 8]                       # 只取这8个动作

# 2. 实验变量 (Grid Search)
MODELS_TO_TEST = [
    ("Simple_CNN", model_lib.build_simple_cnn),
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

# 3. 固定参数
CONFIG = {
    'fs': 2000,                # 采样率                  
    'epochs': 100,
    'batch_size': 256,
    'test_size': 0.2,          # 测试集比例
    'split_strategy': "留文件验证 (同天/同人)",
    'label_smoothing': 0.1,    # 标签平滑
    'voting_start_epoch': 20,  # 投票开启时间
    'voting_weight': 0.5,      # 投票权重
    'samples_per_group': 5,    # 投票组采样数
}

# 4. 数据增强配置 (幅度缩放 + 高斯噪声)
AUGMENT_CONFIG = {
    'enable_rest': True,       # 加入静息
    'multiplier': 1,           # 自动化测试通常不无限倍增，设为1或2即可，或者按需调整
    'enable_scaling': True,    # 幅度缩放
    'enable_noise': True,      # 高斯噪声
    'enable_warp': False,
    'enable_shift': False,
    'enable_mask': False
}

LOG_DIR = "ninaDB2_E1_auto_train_logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# ==================== 1. 辅助类 (Mock Streamlit) ====================
# 为了复用 train_utils 代码，我们需要模拟 Streamlit 的进度条和文本对象
class MockProgressBar:
    def progress(self, value):
        # 针对 Log 模式的优化：直接跳过，什么都不打印
        # 这样日志文件里就不会有成千上万行 "[======...]" 了
        pass

class MockStatusText:
    def text(self, msg):
        # 将关键信息打印到控制台
        print(f"    └─ {msg}")

# ==================== 2. 数据加载函数 ====================

def process_mat_files(data_root="data40"):
    X_list = []
    y_list = []
    groups_list = []
    
    # 1. 遍历 s1 到 s38
    for subject_id in range(1, 39):
        subject_name = f"s{subject_id}"
        # 寻找对应的 E1 文件: data/s1/S1_A1_E1.mat
        folder_path = os.path.join(data_root, subject_name)
        mat_file = os.path.join(folder_path, f"S{subject_id}_A1_E1.mat")
        
        if not os.path.exists(mat_file):
            print(f"⚠️ 跳过: 找不到 {mat_file}")
            continue
            
        print(f"正在处理: {mat_file}")
        
        try:
            # === 读取 MAT 文件 ===
            mat_data = sio.loadmat(mat_file)
            
            # 1. 获取 EMG 数据 (取前8列，并做特定的归一化)
            # 假设 emg 变量名就是 'emg'
            raw_emg = mat_data['emg'][:, :8] 
            raw_emg = raw_emg[::2, :]
            stimulus = mat_data['restimulus'].flatten()[::2]
            emg_data = raw_emg

            subj_act_X, subj_act_y, subj_act_groups = [], [], []
            subj_rest_X, subj_rest_y, subj_rest_groups = [], [], []
            
            # === 核心切片逻辑 ===
            
            # 技巧：使用 np.diff 找到状态变化的边缘
            # 0->1 (动作开始), 1->0 (动作结束)
            # 为了处理方便，我们在前后补0
            stim_padded = np.concatenate(([0], stimulus, [0]))
            diff = np.diff(stim_padded)
            
            # 找到所有动作的 起始索引 和 结束索引
            # where(diff != 0) 会返回变化点，我们需要成对处理
            # 这种方法对于整洁的数据（000111000222000）很有效
            
            # 但更简单的方法可能是：利用 ndimage.label 找连通域（沿用你之前的思路）
            labeled_array, num_features = ndimage.label(stimulus > 0)
            WIN_RADIUS = 1500
            # --- 提取动作样本 ---
            for i in range(1, num_features + 1):
                indices = np.where(labeled_array == i)[0]
                current_label = int(np.median(stimulus[indices]))
                
                if current_label not in TARGET_LABELS:
                    continue
                
                center_idx = int((indices[0] + indices[-1]) / 2)
                
                start_win = center_idx - WIN_RADIUS
                end_win = center_idx + WIN_RADIUS
                
                if start_win < 0 or end_win > len(emg_data):
                    continue
                
                window = emg_data[start_win:end_win]
                
                #  Instance Normalization (段内 Z-Score)
                # 将任意量级的数据标准化到 均值0，方差1
                mean = np.mean(window, axis=0)
                std = np.std(window, axis=0)
                std[std < 1e-8] = 1.0 # 防止除零
                
                window_norm = (window - mean) / std
                
                subj_act_X.append(window_norm)
                subj_act_y.append(current_label)
                subj_act_groups.append(f"{subject_name}_act_{i}")
                
            # --- 提取静息样本 (Rest) ---
            buffer_size = 100 * 10 # 降采样后 Buffer 也相应减半 (原 100*20)
            mask_active = stimulus > 0
            mask_forbidden = ndimage.binary_dilation(mask_active, structure=np.ones(buffer_size))
            mask_rest = ~mask_forbidden 
            
            labeled_rest, num_rest = ndimage.label(mask_rest)
            
            for i in range(1, num_rest + 1):
                r_indices = np.where(labeled_rest == i)[0]
                
                # 长度检查 (原 300*20 -> 现 300*10)
                if len(r_indices) < 300 * 10: continue
                
                center_idx = int((r_indices[0] + r_indices[-1]) / 2)
                start_win = center_idx - WIN_RADIUS
                end_win = center_idx + WIN_RADIUS
                
                if start_win < 0 or end_win > len(emg_data): continue

                window = emg_data[start_win:end_win]
                
                # 静息数据也要做同样的标准化，保证分布一致
                mean = np.mean(window, axis=0)
                std = np.std(window, axis=0)
                std[std < 1e-8] = 1.0
                window_norm = (window - mean) / std
                
                subj_rest_X.append(window_norm)
                subj_rest_y.append(0) 
                subj_rest_groups.append(f"{subject_name}_rest_{i}")

            # 合并数据
            if len(subj_act_X) > 0:
                X_list.extend(subj_act_X)
                y_list.extend(subj_act_y)
                groups_list.extend(subj_act_groups)

                # 平衡静息样本
                num_classes_found = len(np.unique(subj_act_y))
                if num_classes_found > 0:
                    target_rest_count = int(len(subj_act_X) / num_classes_found)
                else:
                    target_rest_count = 0
                
                if len(subj_rest_X) > target_rest_count and target_rest_count > 0:
                    selected_indices = np.random.choice(len(subj_rest_X), target_rest_count, replace=False)
                    for idx in selected_indices:
                        X_list.append(subj_rest_X[idx])
                        y_list.append(subj_rest_y[idx])
                        groups_list.append(subj_rest_groups[idx])
                else:
                    X_list.extend(subj_rest_X)
                    y_list.extend(subj_rest_y)
                    groups_list.extend(subj_rest_groups)

        except Exception as e:
            print(f"❌ 处理出错 {mat_file}: {e}")
            # import traceback
            # traceback.print_exc()

    return np.array(X_list), np.array(y_list), np.array(groups_list)

# ==================== 3. 核心训练循环 ====================
def run_automation():
    # 1. 准备数据
    
    # 模拟进度条
    mock_bar = MockProgressBar()
    mock_status = MockStatusText()
    
    # 加载数据 (只加载一次，节省时间)
    X, y, groups = process_mat_files(data_root="data")
    
    if len(X) == 0:
        print("❌ 生成样本数为 0，退出。")
        return
        
    print(f"\n📊 数据准备就绪: X={X.shape}, y={y.shape}, Classes={np.unique(y)}")
    
    # 切分数据
    train_idx, test_idx = train_utils.smart_split(
        X, y, groups, CONFIG['split_strategy'], test_size=CONFIG['test_size']
    )    
    
    train_idx = np.array(train_idx).astype(int)
    test_idx = np.array(test_idx).astype(int)


    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]
    
    # 映射标签 (确保是 0, 1, 2, 3...)
    unique_labels = np.unique(y)
    num_classes = len(unique_labels)
    label_map = {original: new for new, original in enumerate(unique_labels)}
    y_train_mapped = np.array([label_map[i] for i in y_train])
    y_test_mapped = np.array([label_map[i] for i in y_test])
    
    input_shape = (X.shape[1], X.shape[2])

    MODELS_DIR = "ninaDB2_trained_models"  
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    # 2. 循环实验
    total_experiments = len(MODELS_TO_TEST) * len(OPTIMIZERS_TO_TEST) * len(VOTING_OPTIONS)
    current_exp = 0
    
    for model_name, model_builder in MODELS_TO_TEST:
        for opt_name, opt_class, lr, opt_params in OPTIMIZERS_TO_TEST:
            for use_voting in VOTING_OPTIONS:
                current_exp += 1
                exp_id = f"{model_name}_{opt_name}_Vote{use_voting}"
                print(f"\n\n🚀 [{current_exp}/{total_experiments}] 开始实验: {exp_id}")
                print("-" * 50)
                
                # 构建模型
                tf.keras.backend.clear_session() # 清理内存
                model = model_builder(input_shape, num_classes)
                
                # 构建优化器
                try:
                    optimizer = opt_class(learning_rate=lr, **opt_params)
                except Exception as e:
                    print(f"优化器初始化失败: {e}, 跳过。")
                    continue
                
                # 训练
                start_time = time.time()
                try:
                    history_dict = train_utils.train_with_voting_mechanism(
                        model, 
                        X_train, y_train_mapped, groups_train,
                        X_test, y_test_mapped,
                        epochs=CONFIG['epochs'],
                        batch_size=CONFIG['batch_size'],
                        samples_per_group=CONFIG['samples_per_group'],
                        vote_weight=CONFIG['voting_weight'] if use_voting else 0.0,
                        st_progress_bar=mock_bar,
                        st_status_text=mock_status,
                        use_mixup=False, # 自动化脚本暂不开启Mixup以保持简单，可按需开启
                        label_smoothing=CONFIG['label_smoothing'],
                        voting_start_epoch=CONFIG['voting_start_epoch'] if use_voting else 0,
                        optimizer=optimizer
                    )
                except Exception as e:
                    print(f"\n❌ 训练崩溃: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                    
                duration = time.time() - start_time
                
                save_name = f"{exp_id}.keras" # TF 2.10+ 推荐 .keras，旧版可用 .h5
                save_path = os.path.join(MODELS_DIR, save_name)
                
                try:
                    model.save(save_path)
                    print(f"    💾 最佳模型已保存至: {save_path}")
                except Exception as e:
                    print(f"    ⚠️ 模型保存失败: {e}")

                # 3. 评估与日志保存
                save_log(exp_id, model, history_dict, X_test, y_test_mapped, 
                         label_map, duration, opt_name, lr, use_voting)

def save_log(exp_id, model, history, X_test, y_test, label_map, duration, opt_name, lr, use_voting):
    # 计算预测
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 生成报告
    report_dict = classification_report(
        y_test, y_pred, 
        target_names=[str(k) for k in label_map.keys()], 
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()
    
    # 文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filename = os.path.join(LOG_DIR, f"{timestamp}_{exp_id}.txt")
    
    final_acc = history['val_accuracy'][-1]
    final_loss = history['val_loss'][-1]
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Experiment ID: {exp_id}\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Duration: {duration:.1f} seconds\n")
        f.write("="*40 + "\n")
        f.write(f"Subjects: {TARGET_SUBJECTS}\n")
        f.write(f"Labels: {TARGET_LABELS}\n")
        f.write(f"Model: {model.name}\n")
        f.write(f"Optimizer: {opt_name} (LR={lr})\n")
        f.write(f"Voting Mode: {'ON' if use_voting else 'OFF'}")
        if use_voting:
            f.write(f" (Start Epoch: {CONFIG['voting_start_epoch']})\n")
        else:
            f.write("\n")
        f.write("-" * 20 + "\n")
        f.write(f"Epochs: {CONFIG['epochs']}\n")
        f.write(f"Batch Size: {CONFIG['batch_size']}\n")
        f.write(f"Augment: {AUGMENT_CONFIG}\n")
        f.write("="*40 + "\n")
        f.write(f"Final Val Accuracy: {final_acc*100:.2f}%\n")
        f.write(f"Final Val Loss: {final_loss:.4f}\n")
        f.write("\n--- Classification Report ---\n")
        f.write(report_df.to_string())
        
    print(f"\n💾 日志已保存: {filename}")

if __name__ == "__main__":
    # 设置 GPU 显存增长，防止 OOM
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
    run_automation()