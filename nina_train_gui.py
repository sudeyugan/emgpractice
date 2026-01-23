import streamlit as st
import os
import time
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.ndimage as ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import datetime
import train_utils
import data_loader  # [NEW] 引入数据增强工具库


# ================= 0. 配置与工具函数 =================
st.set_page_config(layout="wide", page_title="NinaPro 微调工作站")

# 防止 GPU OOM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

# 初始化 Session State
if 'trained_model' not in st.session_state:
    st.session_state['trained_model'] = None
if 'train_history' not in st.session_state:
    st.session_state['train_history'] = None

# [NEW] 数据集增强函数 (Post-Split Augmentation)
def augment_dataset(X, y, groups, config, progress_bar=None):
    """
    对训练集进行内存内增强
    """
    multiplier = config.get('multiplier', 1)
    if multiplier <= 1:
        return X, y, groups
    
    X_aug, y_aug, groups_aug = [], [], []
    total = len(X)
    
    # 提取配置
    enable_warp = config.get('enable_warp', False)
    enable_shift = config.get('enable_shift', False)
    enable_scale = config.get('enable_scaling', False)
    enable_mask = config.get('enable_mask', False)
    enable_noise = config.get('enable_noise', False)
    
    for i in range(total):
        # 1. 加入原始样本
        X_aug.append(X[i])
        y_aug.append(y[i])
        groups_aug.append(groups[i])
        
        # 2. 生成增强样本
        for _ in range(multiplier - 1):
            aug_x = X[i].copy()
            
            # 按概率应用各种增强
            if enable_warp and np.random.random() > 0.5:
                aug_x = data_loader.time_warp(aug_x)
            
            if enable_shift and np.random.random() > 0.5:
                aug_x = data_loader.time_shift(aug_x)
                
            if enable_scale and np.random.random() > 0.3:
                aug_x = data_loader.scale_amplitude(aug_x)
                
            if enable_mask and np.random.random() > 0.7:
                aug_x = data_loader.channel_mask(aug_x)
                
            if enable_noise: # 噪声通常最后加
                aug_x = data_loader.add_noise(aug_x)
            
            X_aug.append(aug_x)
            y_aug.append(y[i])
            groups_aug.append(f"{groups[i]}_aug")
            
        if progress_bar and i % 10 == 0:
            progress_bar.progress(i / total)
            
    if progress_bar: progress_bar.progress(1.0)
    
    return np.array(X_aug, dtype=np.float32), np.array(y_aug), np.array(groups_aug)

# ================= 1. 核心：移植自 nina_auto_train.py 的数据处理 =================
def process_nina_data(data_root, selected_subjects, target_labels, 
                      stride_ms=50, split_strategy="mixed", 
                      progress_callback=None):
    """
    完全复刻 nina_auto_train.py 的数据处理逻辑 
    """
    X_list = []
    y_list = []
    groups_list = []
    
    total_files = len(selected_subjects)
    
    # 硬编码配置 (保持与 nina_auto_train 一致)
    # nina_auto_train 中 window 是 Center ± 150，即 300 点
    WINDOW_RADIUS = 150 
    WINDOW_SIZE = WINDOW_RADIUS * 2
    
    for idx, subject_name in enumerate(selected_subjects):
        subj_upper = subject_name.upper()
        
        # 增强的文件名搜索
        possible_filenames = [
            f"{subj_upper}_A1_E1.mat",   # S1_A1_E1.mat
            f"{subject_name}_A1_E1.mat", # s1_A1_E1.mat
        ]
        
        mat_file = None
        folder_path = os.path.join(data_root, subject_name)
        
        for fname in possible_filenames:
            full_path = os.path.join(folder_path, fname)
            if os.path.exists(full_path):
                mat_file = full_path
                break
        
        if progress_callback:
            progress_callback((idx / total_files), f"正在处理: {subject_name}")

        if not mat_file:
            print(f"⚠️ 跳过: 找不到 {subject_name} 的 .mat 文件")
            continue

        try:
            # === 读取 MAT 文件 ===
            mat_data = sio.loadmat(mat_file)
            
            # 1. 获取 EMG (取前8列，归一化)
            if 'emg' in mat_data:
                raw_emg = mat_data['emg']
            else:
                keys = [k for k in mat_data.keys() if 'emg' in k.lower()]
                if keys: raw_emg = mat_data[keys[0]]
                else: continue
            
            raw_emg = raw_emg[:, :8]
            emg_data = (raw_emg / 0.0024) - 1 # 归一化
            
            # 2. 获取标签
            if 'restimulus' in mat_data:
                stimulus = mat_data['restimulus'].flatten()
            elif 'stimulus' in mat_data:
                stimulus = mat_data['stimulus'].flatten()
            else:
                continue

            # === 切片逻辑 ===
            labeled_array, num_features = ndimage.label(stimulus > 0)
            
            subj_act_X, subj_act_y, subj_act_groups = [], [], []
            subj_rest_X, subj_rest_y, subj_rest_groups = [], [], []
            
            # --- A. 提取动作样本 ---
            for i in range(1, num_features + 1):
                indices = np.where(labeled_array == i)[0]
                current_label = int(np.median(stimulus[indices]))
                
                if current_label not in target_labels:
                    continue
                
                center_idx = int((indices[0] + indices[-1]) / 2)
                start_win = center_idx - WINDOW_RADIUS
                end_win = center_idx + WINDOW_RADIUS
                
                if start_win < 0 or end_win > len(emg_data):
                    continue
                
                window = emg_data[start_win:end_win]
                
                if window.shape[0] == WINDOW_SIZE:
                    subj_act_X.append(window)
                    subj_act_y.append(current_label)
                    subj_act_groups.append(f"{subject_name}_act_{i}")

            # --- B. 提取静息样本 (Rest) 
            if 0 in target_labels:
                # 1. 膨胀动作区域 (作为 Buffer)，避开动作边缘
                buffer_size = 100
                mask_active = stimulus > 0
                mask_forbidden = ndimage.binary_dilation(mask_active, structure=np.ones(buffer_size))
                mask_rest = ~mask_forbidden # 取反，得到纯净静息区
                
                labeled_rest, num_rest = ndimage.label(mask_rest)
                
                for i in range(1, num_rest + 1):
                    r_indices = np.where(labeled_rest == i)[0]
                    # 如果这段静息太短 (小于 300)，就跳过
                    if len(r_indices) < 300: continue
                    
                    # 只取这段静息的最中间一段
                    center_idx = int((r_indices[0] + r_indices[-1]) / 2)
                    start_win = center_idx - WINDOW_RADIUS
                    end_win = center_idx + WINDOW_RADIUS
                    
                    if start_win < 0 or end_win > len(emg_data): continue

                    window = emg_data[start_win:end_win]
                    
                    if window.shape[0] == WINDOW_SIZE:
                        subj_rest_X.append(window)
                        subj_rest_y.append(0) # Label 0
                        subj_rest_groups.append(f"{subject_name}_rest_{i}")

            # --- C. 合并与平衡 (Balancing) ---
            if len(subj_act_X) > 0:
                # 1. 动作数据直接加入
                X_list.extend(subj_act_X)
                y_list.extend(subj_act_y)
                groups_list.extend(subj_act_groups)

                # 2. 计算合适的静息数量 (1:1 平衡策略)
                if len(subj_rest_X) > 0:
                    unique_act_classes = np.unique(subj_act_y)
                    num_act_classes_found = len(unique_act_classes)
                    
                    if num_act_classes_found > 0:
                        target_rest_count = int(len(subj_act_X) / num_act_classes_found)
                    else:
                        target_rest_count = len(subj_rest_X) 
                    
                    # 3. 随机采样
                    if len(subj_rest_X) > target_rest_count and target_rest_count > 0:
                        selected_indices = np.random.choice(len(subj_rest_X), target_rest_count, replace=False)
                        for s_idx in selected_indices:
                            X_list.append(subj_rest_X[s_idx])
                            y_list.append(subj_rest_y[s_idx])
                            groups_list.append(subj_rest_groups[s_idx])
                    else:
                        X_list.extend(subj_rest_X)
                        y_list.extend(subj_rest_y)
                        groups_list.extend(subj_rest_groups)
            elif len(subj_rest_X) > 0:
                X_list.extend(subj_rest_X)
                y_list.extend(subj_rest_y)
                groups_list.extend(subj_rest_groups)

        except Exception as e:
            print(f"❌ 处理出错 {mat_file}: {e}")
            
    return np.array(X_list), np.array(y_list), np.array(groups_list)

# ================= 2. 侧边栏：配置 =================
st.sidebar.title("🛠️ NinaPro 微调配置")

# --- A. 模型 ---
st.sidebar.header("1. 基模型 (Base Model)")
base_model_file = st.sidebar.file_uploader(
    "上传 nina_auto_train 生成的 .keras/.h5", 
    type=["keras", "h5"]
)

# --- B. 数据源 ---
st.sidebar.header("2. 数据源 (NinaPro Data)")
data_root_input = st.sidebar.text_input("数据根目录 (包含 s1, s2...)", value="data")

all_subjects = []
if os.path.exists(data_root_input):
    try:
        items = os.listdir(data_root_input)
        all_subjects = sorted([d for d in items if os.path.isdir(os.path.join(data_root_input, d)) and d.startswith('s')])
    except:
        pass

if not all_subjects:
    st.sidebar.warning("未检测到 's*' 文件夹，请手动检查路径")
    manual_subjs = st.sidebar.text_input("或手动输入 Subject (逗号分隔)", "s1, s2")
    if manual_subjs:
        selected_subjects = [s.strip() for s in manual_subjs.split(',')]
else:
    selected_subjects = st.sidebar.multiselect("选择 Subject 进行微调", all_subjects, default=all_subjects[:1])

target_labels_str = st.sidebar.text_input("目标动作 ID (逗号分隔)", "1, 2, 5, 6")
try:
    target_labels = [int(x.strip()) for x in target_labels_str.split(',') if x.strip()]
except:
    st.sidebar.error("Label 格式错误")
    target_labels = []

# --- C. 微调参数 ---
st.sidebar.header("3. 训练参数")
# [MODIFIED] 增加 "直接评估 (Inference Only)" 选项
fine_tune_mode = st.sidebar.radio(
    "模式选择", 
    ["Few-shot (冻结特征)", "Full Fine-tune (全量微调)", "直接评估 (Inference Only)"], 
    index=0
)

# 仅在非直接评估模式下显示这些参数
is_inference_only = (fine_tune_mode == "直接评估 (Inference Only)")
unfreeze_all = (fine_tune_mode == "Full Fine-tune (全量微调)")

if not is_inference_only:
    epochs = st.sidebar.number_input("Epochs", 10, 200, 30)
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    lr = st.sidebar.number_input("Learning Rate", value=0.001, format="%.5f")
    num_shots = st.sidebar.slider("每类样本数 (Few-shot用)", 1, 10, 2) if not unfreeze_all else 9999
else:
    # 推理模式下只需少许参数
    batch_size = 32
    epochs = 0
    st.sidebar.info("ℹ️ 直接评估模式：跳过训练，直接使用基模型预测所选数据。请确保所选动作标签的顺序与模型训练时一致。")

# [NEW] 数据增强配置
with st.sidebar.expander("🧪 数据增强 (Data Augmentation)", expanded=False):
    st.caption("小样本(Few-shot)训练时建议开启")
    aug_multiplier = st.slider("样本倍增系数 (Multiplier)", 1, 20, 1, help="将训练集扩大N倍")
    
    c1, c2 = st.columns(2)
    enable_noise = c1.checkbox("高斯噪声", True)
    enable_scale = c2.checkbox("幅度缩放", True)
    enable_warp = c1.checkbox("时间扭曲", False, help="计算量较大")
    enable_shift = c2.checkbox("时间平移", False)
    enable_mask = st.checkbox("通道遮挡", False)

    augment_config = {
        'multiplier': aug_multiplier,
        'enable_noise': enable_noise,
        'enable_scaling': enable_scale,
        'enable_warp': enable_warp,
        'enable_shift': enable_shift,
        'enable_mask': enable_mask
    }

run_btn = st.sidebar.button("🚀 开始微调", type="primary")

# ================= 3. 主界面逻辑 =================
st.title("🧠 NinaPro 模型微调 (MAT版)")
st.caption("基于 nina_auto_train.py 核心逻辑 + Few-shot 数据增强")

if run_btn:
    if not base_model_file:
        st.error("请上传基模型文件！")
        st.stop()
    if not selected_subjects:
        st.error("请选择至少一个 Subject！")
        st.stop()
        
    # --- Step 1: 加载数据 ---
    st.subheader("1. 读取 MAT 数据")
    bar = st.progress(0)
    status = st.empty()
    
    X, y, groups = process_nina_data(
        data_root_input, 
        selected_subjects, 
        target_labels,
        progress_callback=lambda p, t: (bar.progress(p), status.text(t))
    )
    
    bar.progress(100)
    if len(X) == 0:
        st.error(f"未提取到任何样本！请检查 {data_root_input} 下是否有 .mat 文件，以及 Label 是否存在。")
        st.stop()
        
    if len(X) == 0:
        st.error(f"未提取到任何样本！请检查 {data_root_input} 下是否有 .mat 文件，以及 Label 是否存在。")
        st.stop()
        
    X = X.astype(np.float32)
    
    unique_labels = np.unique(y)
    y_mapped = y.astype(int)  # 直接使用原始值
    
    label_map = {val: val for val in unique_labels} 

    num_classes = int(np.max(y_mapped)) + 1
    
    st.success(f"✅ 原始数据加载成功: X={X.shape}, y={y.shape} | 包含动作: {unique_labels}")
    
    # --- Step 2: 划分数据集 ---
    if is_inference_only:
        # [NEW] 直接评估模式：所有数据都是测试集
        st.info("模式: 直接评估 (Inference Only) - 所有加载的数据将直接用于测试，不进行训练。")
        X_train = np.array([]) # 空数组
        y_train = np.array([])
        groups_train = np.array([])
        
        X_test = X
        y_test = y_mapped
        groups_test = groups # 假设 groups 也有用
        
    elif unfreeze_all:
        if len(selected_subjects) > 1:
            test_mask = np.char.startswith(groups, selected_subjects[-1])
            train_idx = np.where(~test_mask)[0]
            test_idx = np.where(test_mask)[0]
            st.info(f"验证策略: 使用 {selected_subjects[-1]} 作为验证集")
        else:
            from sklearn.model_selection import train_test_split
            idx = np.arange(len(X))
            train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_mapped)
            st.info("验证策略: 单 Subject 内部随机划分 (80/20)")
    else:
        # Few-shot: 随机抽取 N 个样本
        train_idx, test_idx = train_utils.get_few_shot_split(X, y_mapped, num_shots)
        st.info(f"验证策略: Few-shot (每类 {num_shots} 个训练样本)")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_mapped[train_idx], y_mapped[test_idx]
        groups_train = groups[train_idx]

    # --- [NEW] Step 2.5: 数据增强 (仅针对训练集) ---
    # [MODIFIED] 只有在非推理模式且开启增强时才执行
    if not is_inference_only and augment_config['multiplier'] > 1:
        st.subheader("2. 执行数据增强")
        aug_bar = st.progress(0)
        st.info(f"正在将训练集扩大 {augment_config['multiplier']} 倍 (应用: 噪声={enable_noise}, 扭曲={enable_warp}...)")
        
        X_train, y_train, groups_train = augment_dataset(
            X_train, y_train, groups_train, augment_config, progress_bar=aug_bar
        )
        st.success(f"📈 增强后训练集规模: {X_train.shape}")
    
    # --- Step 3: 加载与适配模型 ---
    st.subheader("3. 模型准备")
    
    temp_path = f"temp_{base_model_file.name}"
    with open(temp_path, "wb") as f: f.write(base_model_file.getbuffer())
    
    try:
        base_model = tf.keras.models.load_model(temp_path)
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        st.stop()
        
    if base_model.input_shape[-1] != X.shape[-1]:
        st.error(f"❌ 维度不匹配: 模型输入通道 {base_model.input_shape[-1]} vs 数据通道 {X.shape[-1]}")
        st.stop()
        
    old_classes = base_model.output_shape[-1]
    
    if old_classes >= num_classes:
        num_classes = old_classes
        # st.info(f"已对齐基模型输出维度: {num_classes} 类")
    
    # [MODIFIED] 改造模型逻辑
    if is_inference_only:
        # 推理模式：直接使用原模型
        if old_classes != num_classes:
            st.warning(f"⚠️ 警告: 模型输出类别数 ({old_classes}) 与当前数据类别数 ({num_classes}) 不一致！混淆矩阵可能无法正确对应。")
        model = base_model
        # 即使不训练，compile 也是为了后续 evaluate 能计算 loss/accuracy
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
    elif unfreeze_all:
        base_model.trainable = True
        if old_classes == num_classes:
            model = base_model
        else:
            st.warning(f"重置分类头: {old_classes} -> {num_classes} 类")
            feature_out = base_model.layers[-2].output
            new_out = tf.keras.layers.Dense(num_classes, activation='softmax')(feature_out)
            model = tf.keras.models.Model(base_model.input, new_out)
    else:
        base_model.trainable = False 
        feature_layer = None
        for layer in reversed(base_model.layers):
            if "global" in layer.name or "flatten" in layer.name:
                feature_layer = layer
                break
        
        feat_out = feature_layer.output if feature_layer else base_model.layers[-2].output
        
        x = tf.keras.layers.Dropout(0.5)(feat_out)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        model = tf.keras.models.Model(base_model.input, outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    # --- Step 4: 训练 ---
    # [MODIFIED] 仅在非推理模式下训练
    if not is_inference_only:
        st.subheader("4. 开始训练")
        t_prog = st.progress(0)
        t_status = st.empty()
        st_cb = train_utils.StreamlitKerasCallback(epochs, t_prog, t_status)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[st_cb],
            verbose=0
        )
        # 为了后面画图不报错，构造一个假的 history 对象给推理模式用
    else:
        st.subheader("4. 直接评估 (跳过训练)")
        st.write("正在使用基模型对数据进行预测...")
        # 为了代码兼容性，手动创建一个类似 history 的结构
        class MockHistory:
            def __init__(self): self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        history = MockHistory()

    
    # --- Step 5: 结果 ---
    st.subheader("5. 评估报告")
    
    # [MODIFIED] 获取评估结果
    if is_inference_only:
        loss, final_acc = model.evaluate(X_test, y_test, verbose=0)
    else:
        final_acc = history.history['val_accuracy'][-1]
        
    st.metric("测试集准确率", f"{final_acc:.2%}")
    
    # 混淆矩阵
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    
    c1, c2 = st.columns(2)
    with c1:
        if not is_inference_only:
            # 只有训练过才有曲线
            fig, ax = plt.subplots()
            ax.plot(history.history['loss'], label='Train')
            ax.plot(history.history['val_loss'], label='Val')
            ax.legend()
            ax.set_title("Loss Curve")
            st.pyplot(fig)
        else:
            st.info("直接评估模式无训练曲线")
            
    with c2:
        fig2, ax2 = plt.subplots()
        names = [str(k) for k in label_map.keys()]
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=names, yticklabels=names, cmap='Blues', ax=ax2)
        ax2.set_title("Confusion Matrix")
        st.pyplot(fig2)
        
    # 保存模型
    st.markdown("---")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    save_name = st.text_input("保存模型名称", f"finetuned_nina_{ts}.keras")
    if st.button("💾 保存当前模型"):
        if not os.path.exists("trained_models"): os.makedirs("trained_models")
        path = os.path.join("trained_models", save_name)
        model.save(path)
        st.success(f"已保存至 {path}")