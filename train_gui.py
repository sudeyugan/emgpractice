import streamlit as st
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- 模块导入 ---
import data_loader
import train_utils  # 新导入
import ui_helper    # 新导入
from model import build_simple_cnn, build_advanced_crnn

# ================= 0. 全局设置 =================
# 获取 GPU 设置 (这段代码最好放在最前面)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

st.set_page_config(layout="wide", page_title="EMG 训练工作站")

# 初始化 Session State
if 'trained_model' not in st.session_state:
    st.session_state['trained_model'] = None

# ================= 1. 侧边栏配置 =================
st.sidebar.header("🚀 训练模式")
train_mode = st.sidebar.radio("选择模式", ["从零开始训练", "基于基模型微调 (Few-shot)"])

base_model_path = None
if train_mode == "基于基模型微调 (Few-shot)":
    base_model_path = st.sidebar.file_uploader("上传基模型 (.h5)", type=["h5"])
    num_finetune_samples = st.sidebar.slider("每个类别用于微调的样本数", 1, 10, 5)

with st.sidebar:
    st.header("1. 数据选择")
    DATA_ROOT = "data"
    
    if not os.path.exists(DATA_ROOT):
        st.error(f"未找到 {DATA_ROOT} 文件夹")
        st.stop()
        
    # 调用 ui_helper 扫描文件
    structure, file_map = ui_helper.scan_data_folder(DATA_ROOT)
    
    # --- 级联选择器 (调用 ui_helper) ---
    all_subjects = sorted(structure.keys())
    selected_subjects = ui_helper.render_multiselect_with_all(
        "选择测试者 (Subjects)", all_subjects, 'selected_subjects_key', default_first=True
    )
    
    available_dates = set()
    for s in selected_subjects:
        if s in structure: available_dates.update(structure[s].keys())
    selected_dates = ui_helper.render_multiselect_with_all(
        "选择日期 (Dates)", sorted(list(available_dates)), 'selected_dates_key', default_first=True
    )
    
    available_labels = set()
    for s in selected_subjects:
        for d in selected_dates:
            if s in structure and d in structure[s]: available_labels.update(structure[s][d])
    selected_labels = ui_helper.render_multiselect_with_all(
        "选择动作 ID (Labels)", sorted(list(available_labels)), 'selected_labels_key', default_first=True
    )

    st.markdown("---")
    
    # 统计选中文件
    target_files = []
    for s in selected_subjects:
        for d in selected_dates:
            for l in selected_labels:
                key = (s, d, l)
                if key in file_map: target_files.extend(file_map[key])
    
    st.info(f"已选中 **{len(target_files)}** 个 CSV 文件")

    st.header("2. 增强与训练配置")
    
    with st.expander("🛠️ 数据增强", expanded=True):
        train_stride_ms = st.slider("切片步长 (Stride ms)", 10, 200, 100, 10, help="建议 50ms 左右。")
        enable_scaling = st.checkbox("启用随机幅度缩放", value=False)
        enable_noise = st.checkbox("启用高斯噪声", value=False)
        augment_config = {'enable_scaling': enable_scaling, 'enable_noise': enable_noise}
        
    st.markdown("---")
    model_type = st.selectbox("选择模型核心", ["Lite: Simple CNN", "Pro: Multi-Scale CRNN"])

    split_mode = st.radio("验证策略", ("1. 混合切分", "2. 留文件验证", "3. 留日期/对象验证"))
    
    strategy_map = {
        "1. 混合切分": "混合切分 (看到所有天/人)",
        "2. 留文件验证": "留文件验证 (同天/同人)",
        "3. 留日期/对象验证": "留日期/对象验证 (泛化能力)"
    }
    selected_strategy = strategy_map[split_mode]

    manual_val_target = None
    if "留文件" in selected_strategy and target_files:
        file_options = sorted(list(set([os.path.basename(f) for f in target_files])))
        manual_val_target = st.selectbox("🎯 指定测试文件", file_options)
    elif "留日期" in selected_strategy and target_files:
        group_options = sorted(list(set([os.path.basename(os.path.dirname(f)) for f in target_files])))
        manual_val_target = st.selectbox("🎯 指定测试对象/日期", group_options)

    st.markdown("---") 
    epochs = st.number_input("Epochs", 10, 200, 50)
    batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    test_size = st.slider("测试集比例", 0.01, 0.5, 0.2)
    
    run_btn = st.button("🚀 开始处理并训练", type="primary")

# ================= 2. 主逻辑区域 =================
st.title("🧠 EMG 交互式训练系统")

if run_btn and target_files:
    # --- A. 数据处理 ---
    st.subheader("1. 数据预处理")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    X, y, groups = data_loader.process_selected_files(
        target_files, 
        progress_callback=lambda p, t: (progress_bar.progress(p), status_text.text(t)),
        stride_ms=train_stride_ms,
        augment_config=augment_config
    )
    
    status_text.text("处理完成！")
    progress_bar.progress(100)
    
    if len(X) == 0:
        st.error("样本数为 0，请检查数据。")
        st.stop()
        
    st.success(f"X={X.shape}, y={y.shape} | 类别: {np.unique(y)}")
    
    # --- B. 模型训练准备 ---
    st.subheader("2. 模型训练")
    
    # 划分数据集 (调用 train_utils)
    if train_mode == "基于基模型微调 (Few-shot)":
        train_idx, test_idx = train_utils.get_few_shot_split(X, y, num_finetune_samples)
    else:
        train_idx, test_idx = train_utils.smart_split(
            X, y, groups, selected_strategy, test_size, manual_target=manual_val_target
        )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # 标签映射
    unique_labels = np.unique(y)
    num_classes = len(unique_labels) 
    label_map = {original: new for new, original in enumerate(unique_labels)}
    y_train_mapped = np.array([label_map[i] for i in y_train])
    y_test_mapped = np.array([label_map[i] for i in y_test])

    # 构建模型
    if train_mode == "基于基模型微调 (Few-shot)":
        if base_model_path:
            with open("temp_model.h5", "wb") as f: f.write(base_model_path.getbuffer())
            model = tf.keras.models.load_model("temp_model.h5")
            for layer in model.layers[:-2]: layer.trainable = False
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                          loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            st.success("基模型加载成功 (冻结层).")
        else:
            st.error("请上传基模型")
            st.stop()
    else:
        input_shape = (X.shape[1], X.shape[2])
        if "Lite" in model_type:
            model = build_simple_cnn(input_shape, num_classes)
        else:
            model = build_advanced_crnn(input_shape, num_classes)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # --- C. 开始训练 ---
    st.caption("训练监控")
    train_progress = st.progress(0)
    train_status = st.empty()
    
    # 实例化回调 (调用 train_utils)
    st_callback = train_utils.StreamlitKerasCallback(epochs, train_progress, train_status)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    history = model.fit(
        X_train, y_train_mapped,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test_mapped),
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True), reduce_lr, st_callback],
        verbose=0
    )
    
    st.success("训练完成！")
    
    # --- D. 结果可视化 ---
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.plot(history.history['accuracy'], label='Train')
        ax.plot(history.history['val_accuracy'], label='Val')
        ax.set_title("Accuracy")
        ax.legend()
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='Train')
        ax.plot(history.history['val_loss'], label='Val')
        ax.set_title("Loss")
        ax.legend()
        st.pyplot(fig)
    
    # 投票模拟逻辑
    y_pred = np.argmax(model.predict(X_test), axis=1)
    test_groups = groups[test_idx]
    
    voting_results = {}
    for i, g in enumerate(test_groups):
        if g not in voting_results: voting_results[g] = {'true': y_test_mapped[i], 'preds': []}
        voting_results[g]['preds'].append(y_pred[i])
        
    correct = sum(1 for res in voting_results.values() 
                  if np.argmax(np.bincount(res['preds'], minlength=num_classes)) == res['true'])
    segment_acc = correct / len(voting_results) if voting_results else 0
    
    _, win_acc = model.evaluate(X_test, y_test_mapped, verbose=0)
    st.metric("Segment Level Accuracy (Voting)", f"{segment_acc*100:.2f}%", delta=f"Window Acc: {win_acc*100:.2f}%")

    st.session_state['trained_model'] = model

# --- E. 模型保存 ---
if st.session_state['trained_model']:
    st.markdown("---")
    c1, c2 = st.columns(2)
    save_name = c1.text_input("保存文件名", "my_model.h5")
    if c2.button("保存模型"):
        st.session_state['trained_model'].save(save_name)
        st.success(f"已保存至 {save_name}")

elif run_btn and not target_files:
    st.warning("请选择数据！")