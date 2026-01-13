import streamlit as st
import os
import tensorflow as tf

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

# 引入我们自定义的模块
import data_loader
from model import build_simple_cnn, build_advanced_crnn

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

def smart_split(X, y, groups, strategy, test_size=0.2):
    """
    groups: 这里的 groups 传入的是文件名列表 (每个样本对应的文件名)
    """
    indices = np.arange(len(X))
    train_idx, test_idx = [], []
    
    unique_files = np.unique(groups)
    
    # --- 策略 1: 混合大乱炖 (File-Dependent / Intra-File) ---
    # 逻辑：每个文件都切一刀。如果你有Day1和Day2，它们都会被切分进入训练集。
    # 解决了你的疑惑：这样模型就能学到Day1和Day2的特征了。
    if strategy == "混合切分 (看到所有天/人)":
        for f in unique_files:
            # 找到属于这个文件的所有样本
            f_indices = indices[groups == f]
            # 必须按时间顺序切，防止滑动窗口泄露
            split_point = int(len(f_indices) * (1 - test_size))
            
            # 前面做训练，后面做测试
            train_idx.extend(f_indices[:split_point])
            test_idx.extend(f_indices[split_point:])
            
    # --- 策略 2: 严格留一文件 (Leave-One-File-Out) ---
    # 逻辑：随机选几个文件做测试集。
    # 适用：同一个人，同一种动作，录了5次，想看看第6次能不能识别。
    elif strategy == "留文件验证 (同天/同人)":
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        # 这里按“文件名”分组
        train_i, test_i = next(gss.split(X, y, groups=groups))
        train_idx, test_idx = indices[train_i], indices[test_i]

    # --- 策略 3: 严格留一日期/对象 (Leave-Group-Out) ---
    # 逻辑：解析文件名中的 Date 或 Subject，完全扣除一组。
    # 适用：跨天测试（极客模式），跨人测试。
    elif strategy == "留日期/对象验证 (泛化能力)":
        # 我们需要从 groups (文件名) 中提取出日期或人名
        # 假设文件名格式包含路径： data/Subject/Date/...
        # 我们可以简化逻辑：让 GUI 传进来更高级的 group_labels，或者在这里解析
        
        # 简易实现：这里我们假设 groups 列表里存的是 full path
        # 提取上一级目录名作为 Group ID (通常是 Date 或 Subject)
        real_groups = [os.path.basename(os.path.dirname(f)) for f in groups]
        
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
    
    st.header("2. 训练配置")
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
        index=0,
        help="混合切分：准确率最高，适合绝大多数情况。\n留日期验证：验证模型是否需要每天重新训练。"
    )
    
    # 映射到函数参数
    strategy_map = {
        "1. 混合切分 (推荐)": "混合切分 (看到所有天/人)",
        "2. 留文件验证 (进阶)": "留文件验证 (同天/同人)",
        "3. 留日期/对象验证 (高难)": "留日期/对象验证 (泛化能力)"
    }
    selected_strategy = strategy_map[split_mode]


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
        progress_callback=lambda p, t: (progress_bar.progress(p), status_text.text(t))
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
    
    # 训练回调 (显示训练过程)
    with st.spinner("正在训练模型..."):
        history = model.fit(
            X_train, y_train_mapped,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test_mapped),
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=0 # 隐藏控制台输出
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
    
    # 保存模型选项
    if st.button("💾 保存当前模型"):
        model.save("custom_selection_model.h5")
        st.toast("模型已保存为 custom_selection_model.h5")

elif run_btn and not target_files:
    st.warning("请先在左侧选择数据！")