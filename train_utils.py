import time
import os
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import GroupShuffleSplit


# ================= 原有功能保持不变 =================

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
        current_epoch = epoch + 1
        progress = min(current_epoch / self.total_epochs, 1.0)
        self.progress_bar.progress(progress)
        
        elapsed_time = time.time() - self.start_time
        avg_time_per_epoch = elapsed_time / current_epoch
        remaining_epochs = self.total_epochs - current_epoch
        eta_seconds = avg_time_per_epoch * remaining_epochs
        
        eta_str = time.strftime("%M:%S", time.gmtime(eta_seconds))
        
        loss = logs.get('loss', 0)
        acc = logs.get('accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        val_acc = logs.get('val_accuracy', 0)
        
        status_msg = (
            f"Epoch {current_epoch}/{self.total_epochs} | "
            f"⏳ 剩余: {eta_str} | "
            f"Loss: {loss:.4f} Acc: {acc:.1%} | "
            f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.1%}"
        )
        self.status_text.text(status_msg)

    def on_train_end(self, logs=None):
        self.progress_bar.progress(100)
        self.status_text.text("✅ 训练已完成！")

def smart_split(X, y, groups, strategy, test_size=0.2, manual_target=None):
    """
    根据不同策略划分训练集和测试集
    """
    indices = np.arange(len(X))
    train_idx, test_idx = [], []
    
    unique_files = np.unique(groups)
    
    # --- 策略 1: 混合切分 ---
    if strategy == "混合切分 (看到所有天/人)":
        for f in unique_files:
            f_indices = indices[groups == f]
            split_point = int(len(f_indices) * (1 - test_size))
            train_idx.extend(f_indices[:split_point])
            test_idx.extend(f_indices[split_point:])
            
    # --- 策略 2: 留文件验证 ---
    elif strategy == "留文件验证 (同天/同人)":
        if manual_target:
            is_test = np.array([os.path.basename(g.split('_seg')[0]) == manual_target for g in groups])
            test_idx = indices[is_test]
            train_idx = indices[~is_test]
        else:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
            train_i, test_i = next(gss.split(X, y, groups=groups))
            train_idx, test_idx = indices[train_i], indices[test_i]

    # --- 策略 3: 留日期/对象验证 ---
    elif strategy == "留日期/对象验证 (泛化能力)":
        real_groups = np.array([os.path.basename(os.path.dirname(f)) for f in groups])
        
        if manual_target:
            is_test = (real_groups == manual_target)
            test_idx = indices[is_test]
            train_idx = indices[~is_test]
        else:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
            train_i, test_i = next(gss.split(X, y, groups=real_groups))
            train_idx, test_idx = indices[train_i], indices[test_i]
        
    return np.array(train_idx), np.array(test_idx)

def get_few_shot_split(X, y, n_samples_per_class):
    train_idx = []
    test_idx = []
    unique_labels = np.unique(y)
    
    for label in unique_labels:
        label_indices = np.where(y == label)[0]
        np.random.shuffle(label_indices)
        train_idx.extend(label_indices[:n_samples_per_class])
        test_idx.extend(label_indices[n_samples_per_class:])
        
    return np.array(train_idx), np.array(test_idx)

# ================= 新增：投票训练支持函数 =================

def group_batch_generator(X, y, groups, batch_size, samples_per_group=5):
    """
    生成器：每次产出一个 Batch，其中包含 `batch_size` 个组（Segment）。
    每个组包含 `samples_per_group` 个切片。
    返回形状: (batch_size, samples_per_group, time_steps, features)
    """
    unique_groups = np.unique(groups)
    num_groups = len(unique_groups)
    indices_by_group = {g: np.where(groups == g)[0] for g in unique_groups}
    
    # 打乱组的顺序
    np.random.shuffle(unique_groups)
    
    for i in range(0, num_groups, batch_size):
        batch_groups = unique_groups[i : i + batch_size]
        if len(batch_groups) < batch_size: continue # 丢弃最后不足的一个batch
        
        batch_X = []
        batch_y = []
        
        for g in batch_groups:
            indices = indices_by_group[g]
            # 如果该组切片不够，允许重复采样；如果够，不重复
            replace = len(indices) < samples_per_group
            chosen_idx = np.random.choice(indices, samples_per_group, replace=replace)
            
            batch_X.append(X[chosen_idx])
            # 假设同一组的标签是一样的，取第一个即可
            batch_y.append(y[indices[0]])
            
        yield np.array(batch_X), np.array(batch_y)

def train_with_voting_mechanism(model, X_train, y_train, groups_train, 
                                X_test, y_test, 
                                epochs, batch_size, 
                                samples_per_group, vote_weight, 
                                st_progress_bar, st_status_text):
    """
    自定义训练循环：引入投票一致性 Loss
    """
    # 优化器与损失函数
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    # 记录器
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    
    history = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}
    
    start_time = time.time()
    st_progress_bar.progress(0)
    st_status_text.text("🚀 正在初始化投票训练机制...")

    # 预处理验证集 (不需要分组，按标准方式评估)
    # 使用 tf.data 提升性能
    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size * samples_per_group)

    for epoch in range(epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        
        # 获取分组数据生成器
        data_gen = group_batch_generator(X_train, y_train, groups_train, batch_size, samples_per_group)
        
        # --- 训练步 ---
        for step, (x_batch_groups, y_batch) in enumerate(data_gen):
            # x_batch_groups shape: (B, N, T, F)
            # y_batch shape: (B,)
            
            B, N, T, F = x_batch_groups.shape
            
            # 展平输入以喂给模型: (B*N, T, F)
            x_flat = tf.reshape(x_batch_groups, (B * N, T, F))
            # 扩展标签: (B,) -> (B*N,)
            y_flat = np.repeat(y_batch, N)
            
            with tf.GradientTape() as tape:
                # 1. 前向传播 (得到 Logits 或 Softmax，假设模型最后一层是 Softmax)
                logits_flat = model(x_flat, training=True) # (B*N, Classes)
                
                # 2. 计算 Instance Loss (标准切片级 Loss)
                loss_instance = loss_fn(y_flat, logits_flat)
                
                # 3. 计算 Voting Loss (组级 Loss)
                # 变回 (B, N, Classes)
                logits_grouped = tf.reshape(logits_flat, (B, N, -1))
                
                # 核心：计算该组的平均概率分布 (Soft Voting)
                # 这一步强迫模型学会：哪怕单张切片不准，平均下来必须准
                avg_preds = tf.reduce_mean(logits_grouped, axis=1) # (B, Classes)
                
                loss_vote = loss_fn(y_batch, avg_preds)
                
                # 4. 混合 Loss
                total_loss = (1.0 - vote_weight) * loss_instance + vote_weight * loss_vote

            # 反向传播
            grads = tape.gradient(total_loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            
            # 记录指标
            epoch_loss_avg.update_state(total_loss)
            train_acc_metric.update_state(y_flat, logits_flat)
            
        # --- 验证步 ---
        for x_val, y_val in val_dataset:
            val_logits = model(x_val, training=False)
            val_acc_metric.update_state(y_val, val_logits)
            # 计算 val_loss (这里只算标准的)
            v_loss = loss_fn(y_val, val_logits)

        # --- 收集 Epoch 结果 ---
        train_acc = train_acc_metric.result()
        val_acc = val_acc_metric.result()
        curr_loss = epoch_loss_avg.result()
        
        history['accuracy'].append(float(train_acc))
        history['loss'].append(float(curr_loss))
        history['val_accuracy'].append(float(val_acc))
        history['val_loss'].append(float(v_loss)) # 近似值
        
        # 重置状态
        train_acc_metric.reset_state()
        val_acc_metric.reset_state()
        
        # --- 更新 UI ---
        progress = (epoch + 1) / epochs
        st_progress_bar.progress(progress)
        
        elapsed = time.time() - start_time
        st_status_text.text(f"Epoch {epoch+1}/{epochs} | Loss: {curr_loss:.4f} (VoteWt: {vote_weight}) | Train Acc: {train_acc:.1%} | Val Acc: {val_acc:.1%}")
        
    st_status_text.text("✅ 投票增强训练完成！")
    return history
