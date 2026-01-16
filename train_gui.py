import streamlit as st
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

import datetime 
import json

# --- 模块导入 ---
import data_loader
import train_utils
import ui_helper
# 在顶部 import 区域加入
from model import build_simple_cnn, build_advanced_crnn, build_resnet_model, build_tcn_model, build_dual_stream_model

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

if 'train_results' not in st.session_state:
    st.session_state['train_results'] = None

# ================= 1. 侧边栏配置 =================
st.sidebar.header("🚀 训练模式")
train_mode = st.sidebar.radio("选择模式", ["从零开始训练", "基于基模型微调 (Few-shot)"])

base_model_path = None
unfreeze_all = False
if train_mode == "基于基模型微调 (Few-shot)":
    base_model_path = st.sidebar.file_uploader("上传基模型 (.h5)", type=["h5"])
    
    st.sidebar.markdown("---")
    st.sidebar.caption("微调策略")
    # === 全量微调开关 ===
    unfreeze_all = st.sidebar.checkbox(
        " 解冻所有层 (Full Fine-tuning)", 
        value=False,
        help="勾选此项用于 SGD 接力训练。如果不勾选，则默认为冻结特征层只训练分类头。"
    )
    # ===============================
    
    if not unfreeze_all:
        num_finetune_samples = st.sidebar.slider("每个类别用于微调的样本数", 1, 10, 5)
    else:
        # 如果是全量微调，通常使用全部数据，或者也可以由用户指定
        # 这里为了兼容，可以保持显示，或者提示用户
        st.sidebar.info("已启用全量微调：SGD 将更新模型所有参数。")
        num_finetune_samples = 99999

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
    st.header("模型与高级策略")
    
    # 模型选择列表扩展
    model_options = {
        "Lite: Simple CNN": build_simple_cnn,
        "Pro: Multi-Scale CRNN (Recommended)": build_advanced_crnn,
        "New: ResNet-1D (Deep Residual)": build_resnet_model,
        "New: TCN (Temporal ConvNet)": build_tcn_model,
        "New: Dual-Stream (Time + Freq Fusion)": build_dual_stream_model
    }
    model_choice = st.selectbox("选择模型架构", list(model_options.keys()), index=1)

    st.caption("🔧 优化器配置")
    
    # 布局：左边选优化器，右边填基础学习率
    c_opt1, c_opt2 = st.columns([1, 1])
    with c_opt1:
        # 这里的选项对应我们在调研中选出的几个
        optimizer_name = st.selectbox(
            "选择优化器", 
            ["Adam (Default)", "AdamW (SOTA)", "Nadam (RNN+)", "SGD (Expert)"], 
            index=0
        )
    with c_opt2:
        learning_rate = st.number_input(
            "学习率", 
            value=0.001, format="%.6f", step=0.0001,
            help="通常 Adam/Nadam 用 1e-3, SGD 建议 1e-2 或更小"
        )

    # 动态参数区域：根据选择显示特定参数
    opt_params = {}
    if "AdamW" in optimizer_name:
        # AdamW 核心参数是 weight_decay
        st.caption("🌊 AdamW 专属设置")
        wd = st.number_input("权重衰减 (Weight Decay)", value=1e-4, format="%.5f", step=1e-5, help="推荐 1e-4 ~ 1e-2")
        opt_params['weight_decay'] = wd
        
    elif "SGD" in optimizer_name:
        # SGD 必须配合 Momentum 才好用
        st.caption("🏎️ SGD 专属设置")
        momentum = st.slider("动量 (Momentum)", 0.0, 0.99, 0.9, 0.01, help="通常设置为 0.9")
        opt_params['momentum'] = momentum
    
    # 高级技巧开关
    use_mixup = st.checkbox("🧪 启用 Mixup 数据混合", value=False, help="混合两个样本及标签，提升泛化能力")
    label_smoothing = st.slider("Label Smoothing (标签平滑)", 0.0, 0.5, 0.0, 0.01, help="防止模型对标签过度自信，0.1通常是个好值")
    
    # 投票 Loss (保持不变)
    use_voting_loss = st.checkbox("🗳️ 开启投票机制辅助训练 (Vote Loss)", value=False)
    voting_weight = 0.0
    samples_per_group = 5 # 给一个默认值，防止报错
    
    if use_voting_loss:
        c1, c2 = st.columns(2)
        voting_weight = c1.slider("投票 Loss 权重", 0.1, 0.9, 0.5)
        samples_per_group = c2.slider("每组采样切片数", 2, 20, 5)
        
        # [NEW] 新增：投票介入时机
        voting_start_epoch = st.slider("投票介入 Epoch (Warm-up)", 0, 50, 10, 
                                       help="前 N 轮只训练基础准确率，之后再开启投票约束，防止初期梯度混乱。")
    else:
        # 给默认值防止报错
        voting_start_epoch = 0

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

    X = X.astype(np.float32)
    
    if len(X) == 0:
        st.error("样本数为 0，请检查数据。")
        st.stop()
        
    st.success(f"X={X.shape}, y={y.shape} | 类别: {np.unique(y)}")
    
    # --- B. 模型训练准备 ---
    st.subheader("2. 模型训练")
    if "AdamW" in optimizer_name:
        # 需要 TF 2.10+，如果报错请降级回 Adam
        try:
            optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=opt_params['weight_decay'])
        except AttributeError:
            st.error("您的 TensorFlow 版本过低，不支持 AdamW，已自动切换回 Adam。")
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            
    elif "Nadam" in optimizer_name:
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
        
    elif "SGD" in optimizer_name:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=opt_params['momentum'])
        
    else: # Default Adam
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
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
            # 1. 保存并加载基模型
            with open("temp_model.h5", "wb") as f: f.write(base_model_path.getbuffer())
            base_model = tf.keras.models.load_model("temp_model.h5")
            
            # === [MODIFIED] 修改微调逻辑 ===
            if unfreeze_all:
                # 策略 A: SGD 接力训练 (全量微调)
                base_model.trainable = True 
                
                # [FIX] 检查类别数是否一致，如果不一致必须重置分类头
                # 获取基模型最后一层的输出维度
                old_classes = base_model.output_shape[-1]
                
                if old_classes == num_classes:
                    st.info(f"类别数一致 ({num_classes})，保持原输出层结构。")
                    model = base_model
                else:
                    st.warning(f"检测到类别数变化 (基模型: {old_classes} -> 当前: {num_classes})，正在重置分类头...")
                    # 剥离旧的分类头 (假设最后一层是 Dense)
                    # 寻找倒数第二个特征层 (通常是 GlobalAveragePooling 或 Dropout)
                    # 这里采用一种比较通用的做法：取倒数第二层的输出
                    feature_output = base_model.layers[-2].output 
                    
                    # 重新接一个新的分类层
                    new_output = tf.keras.layers.Dense(num_classes, activation='softmax', name="new_dense_head")(feature_output)
                    model = tf.keras.models.Model(inputs=base_model.input, outputs=new_output)
                
                st.success(f"已加载模型用于 SGD 微调，所有层均可训练。")
                
            else:
                # 策略 B: Few-shot (冻结特征提取器) - 保持你原来的代码
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
            # ==============================

            # 编译模型 (使用你在上一轮对话中添加的 optimizer)
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
        else:
            st.error("请上传基模型 (.h5 文件)")
            st.stop()
    else:
        input_shape = (X.shape[1], X.shape[2])
        selected_builder = model_options[model_choice]
        model = selected_builder(input_shape, num_classes)
        
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # --- C. 开始训练 (分支逻辑) ---
    st.caption("训练监控")
    train_progress = st.progress(0)
    train_status = st.empty()
    if use_voting_loss or use_mixup or label_smoothing > 0:
        # 只要开启了任意高级特性，都建议走自定义训练循环 (train_utils.py)
        # 因为 Keras 原生 fit() 处理 Mixup 比较麻烦
        
        st.info(f"🔵 启动高级训练循环 (Voting={use_voting_loss}, Mixup={use_mixup}, Smoothing={label_smoothing})")
        
        history_dict = train_utils.train_with_voting_mechanism(
            model, X_train, y_train_mapped, groups_train,
            X_test, y_test_mapped,
            epochs=epochs,
            batch_size=batch_size,
            samples_per_group=samples_per_group,
            vote_weight=voting_weight if use_voting_loss else 0.0, # 如果没开投票，权重置0
            st_progress_bar=train_progress,
            st_status_text=train_status,
            use_mixup=use_mixup,
            label_smoothing=label_smoothing,
            voting_start_epoch=voting_start_epoch if use_voting_loss else 0,
            optimizer=optimizer
        )
        
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
    
    if hasattr(history, 'history'):
        final_history = history.history # Keras原生对象转字典
    else:
        final_history = history.history # 自定义Shim对象本身就是字典
        
    # 2. 预先计算预测值 (存起来，避免每次刷新都重算，节省时间)
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 3. 存入 Session State
    st.session_state['train_results'] = {
        'history': final_history,
        'model': model,
        'X_test': X_test,
        'y_test_mapped': y_test_mapped,
        'test_groups': groups[test_idx], # 切分后的组信息
        'test_idx': test_idx,
        'y_pred': y_pred,
        'label_map': label_map,
        'class_names': [str(k) for k in label_map.keys()],
        'optimizer_info': {
            'name': optimizer_name,
            'lr': learning_rate,
            'params': opt_params # 这是我们在 UI 部分定义的那个字典
        }
    }
    
    # 更新全局模型状态
    st.session_state['trained_model'] = model
    st.success("训练完成！结果已缓存。")
    
if st.session_state['train_results'] is not None:
    
    # 1. 从“保险箱”里取出所有数据
    res = st.session_state['train_results']
    
    # 解包变量 (方便后面代码直接复用，不用改太多变量名)
    history_dict = res['history']
    model = res['model']
    X_test = res['X_test']
    y_test_mapped = res['y_test_mapped']
    test_groups = res['test_groups']
    y_pred = res['y_pred']
    label_map = res['label_map']
    class_names = res['class_names']
    num_classes = len(label_map)

    # --- D. 结果可视化 (直接复用你原来的代码，只需改一下 history 变量名) ---
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.plot(history_dict['accuracy'], label='Train')
        ax.plot(history_dict['val_accuracy'], label='Val')
        ax.set_title("Accuracy")
        ax.legend()
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        ax.plot(history_dict['loss'], label='Train')
        ax.plot(history_dict['val_loss'], label='Val')
        ax.set_title("Loss")
        ax.legend()
        st.pyplot(fig)
    
    # --- E. 深度评估报告 ---
    st.markdown("---")
    st.subheader("3. 深度评估报告")
    
    # (1) 混淆矩阵
    st.write("#### (1) 混淆矩阵")
    cm = confusion_matrix(y_test_mapped, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    try:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
    except:
        ax_cm.matshow(cm, cmap='Blues')
    
    # 限制图片宽度
    c_small, _ = st.columns([1, 1])
    with c_small:
        st.pyplot(fig_cm)

    # (2) 详细指标
    st.write("#### (2) 详细分类指标")
    report_dict = classification_report(y_test_mapped, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='Greens', subset=['f1-score']))

    # (3) 投票分析 (这就是你之前点一下就刷新的地方)
    # 现在因为它在 st.session_state 的保护下，刷新也不会消失了
    st.markdown("---")
    
    show_segment_analysis = use_voting_loss
    if not use_voting_loss:
        st.caption("ℹ️ 提示：未开启投票训练，但可手动查看投票评估。")
        # 【关键】这个 checkbox 点击后会刷新页面，但因为 train_results 还在，
        # 所以程序会再次跑进这个 if 块，正确显示结果。
        show_segment_analysis = st.checkbox("🔍 显示片段级平滑/投票评估", value=False)
    
    if show_segment_analysis:
        st.write("#### (3) 🗳️ 动作片段级投票详情")
        
        # --- 投票计算逻辑 (直接复用) ---
        voting_results = {}
        for i, g in enumerate(test_groups): # test_groups 从缓存取的
            if g not in voting_results: 
                voting_results[g] = {'true': y_test_mapped[i], 'preds': []}
            voting_results[g]['preds'].append(y_pred[i]) # y_pred 从缓存取的
            
        segment_stats = {cls: {'total': 0, 'correct': 0} for cls in label_map.keys()}
        total_segments = 0
        total_correct = 0

        for res in voting_results.values():
            true_label = res['true']
            vote_pred = np.argmax(np.bincount(res['preds'], minlength=num_classes))
            true_label_name = list(label_map.keys())[list(label_map.values()).index(true_label)]
            
            segment_stats[true_label_name]['total'] += 1
            total_segments += 1
            if vote_pred == true_label:
                segment_stats[true_label_name]['correct'] += 1
                total_correct += 1
                
        segment_acc = total_correct / total_segments if total_segments > 0 else 0
        st.metric("最终段级准确率", f"{segment_acc*100:.2f}%")
        
        per_class_data = []
        for cls, stat in segment_stats.items():
            acc = (stat['correct'] / stat['total']) * 100 if stat['total'] > 0 else 0
            per_class_data.append({
                "动作ID": cls, "总数": stat['total'], "正确": stat['correct'], "准确率": f"{acc:.1f}%"
            })
        st.table(pd.DataFrame(per_class_data))

    st.markdown("---")
    st.subheader("4. 训练日志归档")
    
    # 1. 创建日志目录
    log_dir = "training_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # 2. 准备日志内容
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/log_{current_time}_{model_choice.split(':')[0].strip()}.txt"
    
    # 收集最终指标
    final_train_acc = history_dict['accuracy'][-1]
    final_val_acc = history_dict['val_accuracy'][-1]
    final_train_loss = history_dict['loss'][-1]
    final_val_loss = history_dict['val_loss'][-1]
    # 构建日志文本
    log_content = []
    log_content.append(f"========================================")
    log_content.append(f"   EMG 训练实验报告")
    log_content.append(f"   时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_content.append(f"========================================\n")
    opt_info = res.get('optimizer_info', {'name': 'Unknown', 'lr': 0, 'params': {}})
    
    log_content.append(f"[1. 数据配置]")
    log_content.append(f"测试对象 (Subjects): {selected_subjects}")
    log_content.append(f"数据日期 (Dates): {selected_dates}")
    log_content.append(f"动作标签 (Labels): {selected_labels}")
    log_content.append(f"文件总数: {len(target_files)}")
    log_content.append(f"切片步长: {train_stride_ms} ms")
    log_content.append(f"增强配置: {json.dumps(augment_config, ensure_ascii=False)}\n")
    
    log_content.append(f"[2. 模型与训练配置]")
    log_content.append(f"模型架构: {model_choice}")
    log_content.append(f"优化器 (Optimizer): {opt_info['name']}")
    log_content.append(f"学习率 (Learning Rate): {opt_info['lr']}")
    if opt_info['params']:
        log_content.append(f"优化器参数 (Params): {json.dumps(opt_info['params'], ensure_ascii=False)}")
    log_content.append(f"验证策略: {selected_strategy}")
    log_content.append(f"Epochs: {epochs}")
    log_content.append(f"Batch Size: {batch_size}")
    log_content.append(f"高级特性: Voting={use_voting_loss}, Mixup={use_mixup}, Smoothing={label_smoothing}")
    if use_voting_loss:
        log_content.append(f"  - Vote Weight: {voting_weight}")
        log_content.append(f"  - Samples/Group: {samples_per_group}")
        log_content.append(f"  - Start Epoch: {voting_start_epoch}")
    log_content.append("")

    log_content.append(f"[3. 训练结果 (Window Level)]")
    log_content.append(f"Final Train Acc: {final_train_acc*100:.2f}%")
    log_content.append(f"Final Val Acc:   {final_val_acc*100:.2f}%")
    log_content.append(f"Final Train Loss: {final_train_loss:.4f}")
    log_content.append(f"Final Val Loss:   {final_val_loss:.4f}\n")
    
    log_content.append(f"[4. 详细分类报告 (Val Set)]")
    # report_df 是前面生成的 DataFrame，利用 to_string 转为文本表格
    log_content.append(report_df.to_string())
    log_content.append("")
    
    if 'segment_acc' in locals():
        log_content.append(f"[5. 片段级评估 (Segment Level)]")
        log_content.append(f"最终段级准确率: {segment_acc*100:.2f}%")
        # 将 per_class_data (列表) 转换为简单的文本表格
        log_content.append(f"{'Label':<10} {'Total':<8} {'Correct':<8} {'Acc':<8}")
        for item in per_class_data:
            log_content.append(f"{str(item['动作ID']):<10} {str(item['总数']):<8} {str(item['正确']):<8} {item['准确率']:<8}")
    else:
        log_content.append(f"[5. 片段级评估]")
        log_content.append("未执行片段级评估。")

    # 3. 写入文件
    try:
        with open(log_filename, "w", encoding="utf-8") as f:
            f.write("\n".join(log_content))
        st.success(f"✅ 训练日志已自动保存至: `{log_filename}`")
        
        # 提供下载按钮 (方便远程查看)
        with open(log_filename, "r", encoding="utf-8") as f:
            st.download_button("📥 下载本次日志文件", f, file_name=os.path.basename(log_filename))
            
    except Exception as e:
        st.error(f"日志保存失败: {e}")

if st.session_state['trained_model']: 
    st.markdown("---")
    c1, c2 = st.columns(2)
    save_name = c1.text_input("保存文件名", "my_model.keras")
    if c2.button("保存模型"):
        try:
            st.session_state['trained_model'].save(save_name)
            st.success(f"已保存至 {save_name}")
        except Exception as e:
            st.error(f"保存失败: {e}")

# 这个 elif 是为了处理还没点开始的情况，也必须顶格
elif run_btn and not target_files:
    st.warning("请选择数据！")
