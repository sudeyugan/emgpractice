import streamlit as st
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

# --- 模块导入 ---
import data_loader
import train_utils
import ui_helper
from model import build_simple_cnn, build_advanced_crnn

# ================= 0. 全局设置 =================
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
        
    structure, file_map = ui_helper.scan_data_folder(DATA_ROOT)
    
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
    
    target_files = []
    for s in selected_subjects:
        for d in selected_dates:
            for l in selected_labels:
                key = (s, d, l)
                if key in file_map: target_files.extend(file_map[key])
    
    st.info(f"已选中 **{len(target_files)}** 个 CSV 文件")

    st.header("2. 增强与训练配置")
    
    with st.expander("数据增强与采样", expanded=False):
        train_stride_ms = st.slider("切片步长 (Stride ms)", 10, 200, 100)
        st.caption("负样本策略")
        enable_rest = st.checkbox("加入静息类 (Rest, Label 0)", value=True)
        st.caption("增强策略")
        c1, c2 = st.columns(2)
        enable_scaling = c1.checkbox("幅度缩放", value=True)
        enable_noise = c2.checkbox("高斯噪声", value=True)
        enable_warp = c1.checkbox("时间扭曲", value=False)
        enable_shift = c2.checkbox("时间平移", value=False)
        enable_mask = st.checkbox("通道遮挡", value=False)
        
        aug_multiplier = 1
        if train_mode == "基于基模型微调 (Few-shot)":
            aug_multiplier = st.slider("样本倍增系数", 1, 50, 20)
        
        augment_config = {
            'enable_rest': enable_rest,
            'multiplier': aug_multiplier,
            'enable_scaling': enable_scaling, 
            'enable_noise': enable_noise,
            'enable_warp': enable_warp,
            'enable_shift': enable_shift,
            'enable_mask': enable_mask
        }
        
    st.markdown("---")
    model_type = st.selectbox("选择模型核心", ["Lite: Simple CNN", "Pro: Multi-Scale CRNN"])

    # === 新增：投票 Loss 配置区 ===
    use_voting_loss = st.checkbox("🗳️ 开启投票机制辅助训练 (Vote Loss)", value=False, 
                                  help="开启后，训练将不仅关注单切片准确率，还会优化整个动作片段的平均预测结果。")
    
    voting_weight = 0.0
    samples_per_group = 5
    if use_voting_loss:
        c1, c2 = st.columns(2)
        voting_weight = c1.slider("投票 Loss 权重", 0.1, 0.9, 0.5, help="权重越高，模型越重视整组的一致性")
        samples_per_group = c2.slider("每组采样切片数", 2, 20, 5, help="每次从一个动作中抽取多少个切片来计算平均值")

    st.markdown("---")
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
    batch_size = st.selectbox("Batch Size (Groups if Voting)", [8, 16, 32, 64, 128, 256, 512], index=1)
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
    
    if train_mode == "基于基模型微调 (Few-shot)":
        train_idx, test_idx = train_utils.get_few_shot_split(X, y, num_finetune_samples)
    else:
        train_idx, test_idx = train_utils.smart_split(
            X, y, groups, selected_strategy, test_size, manual_target=manual_val_target
        )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx] # 获取训练集的组信息，用于投票训练
    
    unique_labels = np.unique(y)
    num_classes = len(unique_labels) 
    label_map = {original: new for new, original in enumerate(unique_labels)}
    y_train_mapped = np.array([label_map[i] for i in y_train])
    y_test_mapped = np.array([label_map[i] for i in y_test])

    # 构建模型
    if train_mode == "基于基模型微调 (Few-shot)":
        if base_model_path:
            with open("temp_model.h5", "wb") as f: f.write(base_model_path.getbuffer())
            base_model = tf.keras.models.load_model("temp_model.h5")
            base_model.trainable = False 
            
            feature_output = None
            for layer in reversed(base_model.layers):
                if "global_average_pooling" in layer.name or "flatten" in layer.name:
                    feature_output = layer.output
                    break
            if feature_output is None: feature_output = base_model.layers[-3].output
            
            x = feature_output
            x = tf.keras.layers.Dropout(0.5, name="ft_dropout_1")(x) 
            x = tf.keras.layers.Dense(64, activation='relu', 
                                      kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                      name="ft_dense_1")(x)
            x = tf.keras.layers.Dropout(0.3, name="ft_dropout_2")(x)
            outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
            
            model = tf.keras.models.Model(inputs=base_model.input, outputs=outputs)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        else:
            st.error("请上传基模型 (.h5 文件)")
            st.stop()
    else:
        input_shape = (X.shape[1], X.shape[2])
        if "Lite" in model_type:
            model = build_simple_cnn(input_shape, num_classes)
        else:
            model = build_advanced_crnn(input_shape, num_classes)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # --- C. 开始训练 (分支逻辑) ---
    st.caption("训练监控")
    train_progress = st.progress(0)
    train_status = st.empty()

    if use_voting_loss:
        st.info(f"🔵 投票训练模式已激活 (Weight={voting_weight}, Samples/Group={samples_per_group})")
        
        # 调用我们在 train_utils 中新写的自定义训练循环
        history_dict = train_utils.train_with_voting_mechanism(
            model, X_train, y_train_mapped, groups_train,
            X_test, y_test_mapped,
            epochs=epochs,
            batch_size=batch_size,
            samples_per_group=samples_per_group,
            vote_weight=voting_weight,
            st_progress_bar=train_progress,
            st_status_text=train_status
        )
        
        # 伪装成 Keras history 对象以便后面画图代码复用
        class HistoryShim:
            def __init__(self, h_dict): self.history = h_dict
        history = HistoryShim(history_dict)
        
    else:
        # 标准训练模式
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
    
# --- D. 结果可视化 (基础曲线) ---
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.plot(history.history['accuracy'], label='Train')
        ax.plot(history.history['val_accuracy'], label='Val')
        ax.set_title("Window Level Accuracy")
        ax.legend()
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='Train')
        ax.plot(history.history['val_loss'], label='Val')
        ax.set_title("Loss Curve")
        ax.legend()
        st.pyplot(fig)
    
    # --- E. 深度评估报告  ---
    st.markdown("---")
    st.subheader("3. 深度评估报告")

    # 1. 准备预测数据
    # 获取切片级预测
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 2. 混淆矩阵 (Confusion Matrix)
    st.write("#### (1) 混淆矩阵 (Confusion Matrix)")
    st.caption("横轴为预测类别，纵轴为真实类别。对角线颜色越深越好。")
    
    cm = confusion_matrix(y_test_mapped, y_pred)
    class_names = [str(k) for k in label_map.keys()] # 获取类别名称
    
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    try:
        # 尝试使用 Seaborn 绘制漂亮的热力图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, 
                    yticklabels=class_names, ax=ax_cm)
    except:
        # 如果没有安装 seaborn，使用 matplotlib 兜底
        cax = ax_cm.matshow(cm, cmap='Blues')
        fig_cm.colorbar(cax)
        for (i, j), z in np.ndenumerate(cm):
            ax_cm.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        ax_cm.set_xticklabels([''] + class_names)
        ax_cm.set_yticklabels([''] + class_names)
    
    ax_cm.set_xlabel('Predicted Label')
    ax_cm.set_ylabel('True Label')
    st.pyplot(fig_cm)

    # 3. 详细分类指标 (Classification Report)
    st.write("#### (2) 详细分类指标")
    report_dict = classification_report(y_test_mapped, y_pred, 
                                        target_names=class_names, 
                                        output_dict=True)
    # 转为 DataFrame 并高亮显示
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='Greens', subset=['f1-score']))

    # 4. 基于投票的“分动作”准确率 (Per-Class Segment Accuracy)
    st.write("#### (3) 🗳️ 动作片段级投票详情 (Segment Level Analysis)")
    
    # --- 投票逻辑 ---
    test_groups = groups[test_idx]
    voting_results = {}
    
    # 收集每个片段的票数
    for i, g in enumerate(test_groups):
        if g not in voting_results: 
            voting_results[g] = {'true': y_test_mapped[i], 'preds': []}
        voting_results[g]['preds'].append(y_pred[i])
    
    # 统计结果
    segment_stats = {} # 记录每个类别的 {total: 0, correct: 0}
    for cls in label_map.keys():
        segment_stats[cls] = {'total': 0, 'correct': 0}

    total_segments = 0
    total_correct = 0

    for res in voting_results.values():
        true_label = res['true']
        # 找到票数最多的类别
        vote_pred = np.argmax(np.bincount(res['preds'], minlength=num_classes))
        
        # 转换回原始 Label 名称以便统计
        true_label_name = list(label_map.keys())[list(label_map.values()).index(true_label)]
        
        segment_stats[true_label_name]['total'] += 1
        total_segments += 1
        if vote_pred == true_label:
            segment_stats[true_label_name]['correct'] += 1
            total_correct += 1
            
    # 计算总投票准确率
    segment_acc = total_correct / total_segments if total_segments > 0 else 0
    
    # 显示大字指标
    st.metric(" 最终段级准确率 (Segment Accuracy)", f"{segment_acc*100:.2f}%", 
              help="这是实际使用时的预期准确率（经过投票修正后）")
    
    # 显示分动作详情表
    st.caption("👇 每个动作独立表现：")
    per_class_data = []
    for cls, stat in segment_stats.items():
        acc = (stat['correct'] / stat['total']) * 100 if stat['total'] > 0 else 0
        per_class_data.append({
            "动作ID (Label)": cls,
            "片段总数": stat['total'],
            "正确识别数": stat['correct'],
            "准确率 (%)": f"{acc:.1f}%"
        })
    
    st.table(pd.DataFrame(per_class_data))

# --- F. 模型保存 ---
if st.session_state['trained_model']:
    st.markdown("---")
    c1, c2 = st.columns(2)
    save_name = c1.text_input("保存文件名", "my_model.h5")
    if c2.button("保存模型"):
        st.session_state['trained_model'].save(save_name)
        st.success(f"已保存至 {save_name}")

elif run_btn and not target_files:
    st.warning("请选择数据！")
