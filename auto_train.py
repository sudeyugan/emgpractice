import os
import sys
import time
import glob
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report

# 引用现有模块
import data_loader
import train_utils
import model as model_lib  # 避免变量名冲突

# ==================== 0. 配置区域 (根据需求修改) ====================

# 1. 目标设置
TARGET_SUBJECTS = ["charles", "gavvin", "gerard", "giland", "jessie", "legend"]  #在此处填写你要测试的“特定几个人”的名字
TARGET_LABELS = [5, 6, 7, 8]            # 指定动作标签
TARGET_DATES = None                     # None 表示所有日期，或者写 ["20250213"]

# 2. 实验变量 (Grid Search)
MODELS_TO_TEST = [
    # 格式: (模型名称, 构建函数)
    ("Simple_CNN", model_lib.build_simple_cnn),
    ("Advanced_CRNN", model_lib.build_advanced_crnn),
    ("ResNet1D", model_lib.build_resnet_model),
    ("TCN", model_lib.build_tcn_model)
]

OPTIMIZERS_TO_TEST = [
    # 格式: (名称, 类/函数, 学习率, 其他参数)
    ("Adam", tf.keras.optimizers.Adam, 0.001, {}),
    ("AdamW", tf.keras.optimizers.AdamW, 0.001, {'weight_decay': 1e-4}),
    ("SGD", tf.keras.optimizers.SGD, 0.01, {'momentum': 0.9}),
]

VOTING_OPTIONS = [False, True] # 是否开启投票

# 3. 固定参数
CONFIG = {
    'epochs': 100,
    'batch_size': 128,
    'stride_ms': 50,           # 切片步长 50ms
    'test_size': 0.2,          # 测试集比例
    'split_strategy': "混合切分 (看到所有天/人)", 
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

LOG_DIR = "auto_train_logs"
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
def find_target_files(data_root="data"):
    target_files = []
    # 遍历所有层级寻找 RAW_EMG
    pattern = os.path.join(data_root, "*", "*", "RAW_EMG*.csv")
    all_files = glob.glob(pattern)
    
    print(f"🔍 扫描中... 共发现 {len(all_files)} 个文件，正在筛选...")
    
    for f in all_files:
        subject, date, label, fname = data_loader.parse_filename_info(f)
        
        # 筛选逻辑
        if subject not in TARGET_SUBJECTS: continue
        if TARGET_DATES and date not in TARGET_DATES: continue
        if label not in TARGET_LABELS: continue
        
        target_files.append(f)
        
    return sorted(target_files)

# ==================== 3. 核心训练循环 ====================
def run_automation():
    # 1. 准备数据
    files = find_target_files()
    if not files:
        print("❌ 未找到符合条件的文件，请检查路径和配置。")
        return

    print(f"✅ 选中文件数: {len(files)}")
    print("⏳ 正在预处理数据 (这可能需要几分钟)...")
    
    # 模拟进度条
    mock_bar = MockProgressBar()
    mock_status = MockStatusText()
    
    # 加载数据 (只加载一次，节省时间)
    X, y, groups = data_loader.process_selected_files(
        files, 
        progress_callback=lambda p, t: None, # 预处理时不刷屏
        stride_ms=CONFIG['stride_ms'],
        augment_config=AUGMENT_CONFIG
    )
    
    if len(X) == 0:
        print("❌ 生成样本数为 0，退出。")
        return
        
    print(f"\n📊 数据准备就绪: X={X.shape}, y={y.shape}, Classes={np.unique(y)}")
    
    # 切分数据
    train_idx, test_idx = train_utils.smart_split(
        X, y, groups, CONFIG['split_strategy'], test_size=CONFIG['test_size']
    )
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

    MODELS_DIR = "trained_models"  # [NEW]
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