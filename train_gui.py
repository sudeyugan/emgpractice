import streamlit as st
import os
import tensorflow as tf
import time
from tensorflow.keras.callbacks import Callback

# 获取所有可见的物理 GPU 设备
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 设置显存按需增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("已开启显存按需增长模式")
    except RuntimeError as e:
        print(e)
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras.callbacks import ReduceLROnPlateau

# 引入我们自定义的模块
import data_loader
from model import build_simple_cnn, build_advanced_crnn


class StreamlitKerasCallback(Callback):
    """
    用于连接 Keras 训练过程与 Streamlit 进度条的自定义回调
    """
    def __init__(self, total_epochs, progress_bar, status_text):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.progress_bar.progress(0)
        self.status_text.text("🚀 准备开始训练...")

    def on_epoch_end(self, epoch, logs=None):
        # epoch 是从 0 开始的下标，所以 +1
        current_epoch = epoch + 1
        
        # 1. 更新进度条 (防止 EarlyStopping 导致比例溢出，限制在 0-1 之间)
        progress = min(current_epoch / self.total_epochs, 1.0)
        self.progress_bar.progress(progress)
        
        # 2. 计算时间
        elapsed_time = time.time() - self.start_time
        avg_time_per_epoch = elapsed_time / current_epoch
        remaining_epochs = self.total_epochs - current_epoch
        eta_seconds = avg_time_per_epoch * remaining_epochs
        
        # 格式化时间字符串
        eta_str = time.strftime("%M:%S", time.gmtime(eta_seconds))
        elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed_time))
        
        # 3. 获取指标 (Loss & Accuracy)
        loss = logs.get('loss', 0)
        acc = logs.get('accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        val_acc = logs.get('val_accuracy', 0)
        
        # 4. 更新状态文本
        status_msg = (
            f"Epoch {current_epoch}/{self.total_epochs} | "
            f"⏳ 剩余: {eta_str} (已用: {elapsed_str}) | "
            f"Loss: {loss:.4f} Acc: {acc:.1%} | "
            f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.1%}"
        )
        self.status_text.text(status_msg)

    def on_train_end(self, logs=None):
        # 训练结束（包括早停），将进度条拉满并提示
        self.progress_bar.progress(100)
        self.status_text.text("✅ 训练已完成！")

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',    # 监控验证集的损失值
    factor=0.5,             # 学习率调整倍数：当触发时，新学习率 = 旧学习率 * 0.5
    patience=5,             # 耐心值：如果连续 5 个 epoch 验证集损失都没有改善，则触发
    min_lr=1e-6,            # 学习率下限：防止学习率被减到过小
    verbose=1               # 触发时在终端打印消息
)

st.set_page_config(layout="wide", page_title="EMG 训练工作站")

# ================= 1. 文件扫描逻辑 =================
@st.cache_data
def scan_data_folder(root_dir):
    """扫描文件夹，构建 Subject -> Date -> Labels 结构"""
    structure = {}
    file_map = {} # 存储 label -> file_path list，用于快速检索
    
    # 查找所有 RAW_EMG 文件
    pattern = os.path.join(root_dir, "*", "*", "RAW_EMG*.csv")
    files = glob.glob(pattern)
    
    for f in files:
        subject, date, label, fname = data_loader.parse_filename_info(f)
        if label is None: continue
        
        if subject not in structure: structure[subject] = {}
        if date not in structure[subject]: structure[subject][date] = set()
        
        structure[subject][date].add(label)
        
        # 构建索引键
        key = (subject, date, label)
        if key not in file_map: file_map[key] = []
        file_map[key].append(f)
        
    return structure, file_map

def smart_split(X, y, groups, strategy, test_size=0.2, manual_target=None):
    """
    groups: 这里的 groups 传入的是文件名列表 (每个样本对应的文件名 path)
    manual_target: 用户手动指定的验证集对象 (文件名 或 日期文件夹名)
    """
    indices = np.arange(len(X))
    train_idx, test_idx = [], []
    
    unique_files = np.unique(groups)
    
    # --- 策略 1: 混合大乱炖 (保持不变) ---
    if strategy == "混合切分 (看到所有天/人)":
        for f in unique_files:
            f_indices = indices[groups == f]
            split_point = int(len(f_indices) * (1 - test_size))
            train_idx.extend(f_indices[:split_point])
            test_idx.extend(f_indices[split_point:])
            
    # --- 策略 2: 严格留一文件 (Leave-One-File-Out) ---
    elif strategy == "留文件验证 (同天/同人)":
        # === 新增：手动模式 ===
        if manual_target:
            # groups 里存的是全路径，manual_target 是文件名 (basename)
            # 找到所有属于该文件的样本索引
            is_test = np.array([os.path.basename(f) == manual_target for f in groups])
            test_idx = indices[is_test]
            train_idx = indices[~is_test]
        else:
            # === 原有：随机模式 ===
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
            train_i, test_i = next(gss.split(X, y, groups=groups))
            train_idx, test_idx = indices[train_i], indices[test_i]

    # --- 策略 3: 严格留一日期/对象 (Leave-Group-Out) ---
    elif strategy == "留日期/对象验证 (泛化能力)":
        # 提取 Group ID (文件夹名)
        real_groups = np.array([os.path.basename(os.path.dirname(f)) for f in groups])
        
        # === 新增：手动模式 ===
        if manual_target:
            is_test = (real_groups == manual_target)
            test_idx = indices[is_test]
            train_idx = indices[~is_test]
        else:
            # === 原有：随机模式 ===
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
            train_i, test_i = next(gss.split(X, y, groups=real_groups))
            train_idx, test_idx = indices[train_i], indices[test_i]
        
    return np.array(train_idx), np.array(test_idx)

# ================= 界面布局 =================

st.title("🧠 EMG 交互式训练系统")

with st.sidebar:
    st.header("1. 数据选择")
    DATA_ROOT = "data"
    
    if not os.path.exists(DATA_ROOT):
        st.error(f"未找到 {DATA_ROOT} 文件夹")
        st.stop()
        
    structure, file_map = scan_data_folder(DATA_ROOT)
    
# --- 级联选择器 ---
    # 1. 选择对象 (Subject)
    # 逻辑：默认选中第一个
    all_subjects = sorted(structure.keys())
    selected_subjects = st.multiselect(
        "选择测试者 (Subjects)", 
        all_subjects, 
        default=all_subjects[:1] # 保持不变，已经是默认选第一个
    )
    
    # 2. 选择日期 (Date) - 基于选中的对象
    available_dates = set()
    for s in selected_subjects:
        if s in structure:
            available_dates.update(structure[s].keys())
    
    # 排序日期列表
    sorted_dates = sorted(list(available_dates))
    
    # 修改点：default=sorted_dates[:1] 表示默认只选第一个
    selected_dates = st.multiselect(
        "选择日期 (Dates)", 
        sorted_dates, 
        default=sorted_dates[:1] 
    )
    
    # 3. 选择动作 (Labels) - 基于选中的对象和日期
    available_labels = set()
    for s in selected_subjects:
        for d in selected_dates:
            if s in structure and d in structure[s]:
                available_labels.update(structure[s][d])
    
    # 排序标签列表
    sorted_labels = sorted(list(available_labels))
    
    # 修改点：default=sorted_labels[:1] 表示默认只选第一个
    selected_labels = st.multiselect(
        "选择动作 ID (Labels)", 
        sorted_labels, 
        default=sorted_labels[:1]
    )

    st.markdown("---")
    
    # 统计选中文件
    target_files = []
    for s in selected_subjects:
        for d in selected_dates:
            for l in selected_labels:
                key = (s, d, l)
                if key in file_map:
                    target_files.extend(file_map[key])
    
    st.info(f"已选中 **{len(target_files)}** 个 CSV 文件")

    st.header("2. 增强与训练配置")
    
    with st.expander("🛠️ 数据增强 (Data Augmentation)", expanded=True):
        st.caption("通过增加数据多样性来防止过拟合并提升投票效果。")
        
        # 1. 动态步长 (Level 1)
        # 默认 100ms，设小一点（比如 50ms）可以成倍增加窗口数量
        train_stride_ms = st.slider("切片步长 (Stride ms)", 10, 200, 100, 10, 
                                    help="越小产生的窗口越多，投票基数越大。建议 50ms 左右。")
        
        # 2. 信号扰动 (Level 2)
        enable_scaling = st.checkbox("启用随机幅度缩放 (Scaling)", value=False)
        enable_noise = st.checkbox("启用高斯噪声 (Gaussian Noise)", value=False)
        
        augment_config = {
            'enable_scaling': enable_scaling,
            'enable_noise': enable_noise
        }
    st.markdown("---")
    st.markdown("##### 🧠 模型架构选择")
    model_type = st.selectbox(
        "选择模型核心",
        ["Lite: Simple CNN (推荐单人)", "Pro: Multi-Scale CRNN (推荐多人/跨天)"],
        index=0,
        help="Lite版：训练快，适合小样本；Pro版：抗干扰强，需要较多数据。"
    )

    st.markdown("##### 🧪 验证策略选择")
    split_mode = st.radio(
        "你想怎么验证模型？",
        (
            "1. 混合切分 (推荐)", 
            "2. 留文件验证 (进阶)",
            "3. 留日期/对象验证 (高难)"
        ),
        index=0
    )
    
    strategy_map = {
        "1. 混合切分 (推荐)": "混合切分 (看到所有天/人)",
        "2. 留文件验证 (进阶)": "留文件验证 (同天/同人)",
        "3. 留日期/对象验证 (高难)": "留日期/对象验证 (泛化能力)"
    }
    selected_strategy = strategy_map[split_mode]

    # === 新增：根据策略显示“指定验证集”的下拉框 ===
    manual_val_target = None
    
    if "留文件" in selected_strategy:
        # 从 target_files 中提取所有文件名
        if target_files:
            file_options = sorted(list(set([os.path.basename(f) for f in target_files])))
            manual_val_target = st.selectbox(
                "🎯 指定哪一个文件做测试？", 
                file_options,
                help="选中的文件将完全不参与训练，只用来做最后的考试。"
            )
            
    elif "留日期" in selected_strategy:
        # 从 target_files 中提取所有日期文件夹名
        if target_files:
            group_options = sorted(list(set([os.path.basename(os.path.dirname(f)) for f in target_files])))
            manual_val_target = st.selectbox(
                "🎯 指定哪一天/对象做测试？", 
                group_options,
                help="选中的日期/对象的所有数据都将作为测试集，用于验证模型的跨天泛化能力。"
            )


    st.markdown("---") 
    epochs = st.number_input("Epochs", 10, 200, 50)
    batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
    test_size = st.slider("测试集比例", 0.1, 0.5, 0.2)
    
    run_btn = st.button("🚀 开始处理并训练", type="primary")
# ================= 主逻辑区域 =================

if run_btn and target_files:
    # --- 1. 数据处理阶段 ---
    st.subheader("1. 数据预处理")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 调用 data_loader 处理数据
    X, y, groups = data_loader.process_selected_files(
        target_files, 
        progress_callback=lambda p, t: (progress_bar.progress(p), status_text.text(t)),
        stride_ms=train_stride_ms,   # <--- 传入动态步长
        augment_config=augment_config # <--- 传入增强配置
    )
    
    status_text.text("处理完成！")
    progress_bar.progress(100)
    
    if len(X) == 0:
        st.error("生成的样本数为 0，请检查文件是否包含有效动作数据。")
        st.stop()
        
    st.success(f"成功生成样本数据: X={X.shape}, y={y.shape}")
    st.write(f"包含动作类别: {np.unique(y)}")
    
    # --- 2. 训练阶段 ---
    st.subheader("2. 模型训练")
    
    train_idx, test_idx = smart_split(X, y, groups, selected_strategy, test_size)
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    st.info(f"数据集划分结果 ({selected_strategy}):\n"
            f"- 训练集: {X_train.shape[0]} 样本\n"
            f"- 测试集: {X_test.shape[0]} 样本")

    # 重新映射标签 (保持不变)
    unique_labels = np.unique(y)
    label_map = {original: new for new, original in enumerate(unique_labels)}
    y_train_mapped = np.array([label_map[i] for i in y_train])
    y_test_mapped = np.array([label_map[i] for i in y_test])
    
    # 检查测试集是否包含训练集没有的标签 (防止报错)
    if len(np.unique(y_test)) < len(unique_labels) and "跨文件" in selected_strategy:
        st.warning("⚠️ 注意：测试集中某些动作类别可能缺失，这通常是因为选中的文件太少，导致按文件切分时把某个动作的所有文件都分到了训练集。")
    
    num_classes = len(unique_labels)
    
    st.subheader(f"正在构建模型: {model_type.split(':')[0]}")
    
    input_shape = (X.shape[1], X.shape[2])
    
    # === 根据选择调用不同的模型构建函数 ===
    if "Lite" in model_type:
        model = build_simple_cnn(input_shape=input_shape, num_classes=num_classes)
        st.caption("已加载 Simple CNN：结构轻量，专注局部特征。")
    else:
        model = build_advanced_crnn(input_shape=input_shape, num_classes=num_classes)
        st.caption("已加载 Multi-Scale CRNN：多尺度视野 + 时序记忆。")
        
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    st.write("---")
    st.caption("训练监控面板")
    train_progress_bar = st.progress(0) # 进度条
    train_status_text = st.empty()      # 用于显示文字详情的占位符
    
    # === 新增：实例化自定义回调 ===
    st_callback = StreamlitKerasCallback(
        total_epochs=epochs, 
        progress_bar=train_progress_bar, 
        status_text=train_status_text
    )

    # 训练回调 (显示训练过程)
    # 这里的 st.spinner 可以去掉或者保留，因为我们已经有进度条了，保留着也不冲突
    with st.spinner("正在初始化训练..."):
        history = model.fit(
            X_train, y_train_mapped,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test_mapped),
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True), 
                reduce_lr, 
                st_callback  # <--- 重点：把我们刚才写的 callback 加进去
            ],
            verbose=0 # 保持 0，因为我们自己接管了输出
        )
    
    st.success("训练完成！")
    
    # --- 3. 结果可视化 ---
    st.subheader("3. 训练结果")
    
    col1, col2 = st.columns(2)
    
    # 准确率曲线
    with col1:
        fig1, ax1 = plt.subplots()
        ax1.plot(history.history['accuracy'], label='Train Acc')
        ax1.plot(history.history['val_accuracy'], label='Val Acc')
        ax1.set_title("Accuracy Curve")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        st.pyplot(fig1)
        
    # 损失曲线
    with col2:
        fig2, ax2 = plt.subplots()
        ax2.plot(history.history['loss'], label='Train Loss')
        ax2.plot(history.history['val_loss'], label='Val Loss')
        ax2.set_title("Loss Curve")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        st.pyplot(fig2)
    
    # 最终评估
    loss, acc = model.evaluate(X_test, y_test_mapped, verbose=0)
    st.metric("最终测试集准确率 (Test Accuracy)", f"{acc*100:.2f}%")
    
    st.markdown("---")
    st.subheader("🗳️ 多数投票模拟 (Majority Voting Simulation)")
    st.caption("由于我们减小了 Stride，每个动作片段会被切成多个窗口。这里模拟真实推理：统计属于同一个动作文件的所有窗口预测结果，取众数作为最终结果。")
    
    # 1. 获取测试集的所有预测结果
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 2. 获取测试集对应的原始文件归属 (groups)
    # 注意：smart_split 返回的是索引，我们要用索引去取 groups
    test_groups = groups[test_idx]
    
    # 3. 按文件分组统计
    # 结构: { 'filename_df1.csv': {'true': 1, 'preds': [1, 1, 1, 2, 1]} }
    voting_results = {}
    
    for i, group_name in enumerate(test_groups):
        if group_name not in voting_results:
            voting_results[group_name] = {'true': y_test_mapped[i], 'preds': []}
        voting_results[group_name]['preds'].append(y_pred[i])
        
    # 4. 计算投票准确率
    correct_segments = 0
    total_segments = len(voting_results)
    
    st.write(f"测试集包含 **{total_segments}** 个独立的动作片段 (Segments)。")
    
    # 5. 计算最终 Segment-level Accuracy
    for fname, res in voting_results.items():
        counts = np.bincount(res['preds'], minlength=num_classes)
        voted_label = np.argmax(counts)
        if voted_label == res['true']:
            correct_segments += 1
            
    segment_acc = correct_segments / total_segments if total_segments > 0 else 0
    
    # 使用两列并排显示核心指标
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("窗口级准确率 (Window Acc)", f"{acc*100:.2f}%", help="单个250ms切片的准确率")
    with col_m2:
        st.metric("投票后准确率 (Segment Acc)", f"{segment_acc*100:.2f}%", delta=f"{(segment_acc-acc)*100:.2f}%")
    # 保存模型选项
    if st.button("💾 保存当前模型"):
        model.save("custom_selection_model.h5")
        st.toast("模型已保存为 custom_selection_model.h5")

elif run_btn and not target_files:
    st.warning("请先在左侧选择数据！")