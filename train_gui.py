import streamlit as st
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# 引入我们自定义的模块
import data_loader
from model import build_cnn_model

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
    X, y = data_loader.process_selected_files(
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
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # 重新映射标签 (如果选中的 label 是 [1, 5, 8]，需要映射到 [0, 1, 2] 才能训练)
    unique_labels = np.unique(y)
    label_map = {original: new for new, original in enumerate(unique_labels)}
    y_train_mapped = np.array([label_map[i] for i in y_train])
    y_test_mapped = np.array([label_map[i] for i in y_test])
    
    num_classes = len(unique_labels)
    
    # 构建模型
    model = build_cnn_model(input_shape=(X.shape[1], X.shape[2]), num_classes=num_classes)
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