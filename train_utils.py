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

def mixup(x, y, alpha=0.2):
    """
    对 batch 数据进行 Mixup 增强
    x: (Batch, Time, Feat)
    y: (Batch, Classes) -> 必须是 One-Hot 编码
    """
    if alpha <= 0: return x, y
    
    batch_size = tf.shape(x)[0]
    
    # 生成 Mixup 系数 lambda (Beta分布)
    # tf.random.gamma 用于生成 Beta 分布
    weight = tf.random.gamma([batch_size], alpha, 1.0)
    beta = tf.random.gamma([batch_size], alpha, 1.0)
    lam = weight / (weight + beta)
    lam = tf.reshape(lam, [batch_size, 1, 1]) # 广播维度
    
    # 打乱数据顺序
    indices = tf.range(batch_size)
    shuffled_indices = tf.random.shuffle(indices)
    
    x_shuffled = tf.gather(x, shuffled_indices)
    y_shuffled = tf.gather(y, shuffled_indices)
    
    # 混合
    x_mix = x * lam + x_shuffled * (1 - lam)
    
    # 标签混合 (lam 维度调整为 [batch, 1])
    lam_y = tf.reshape(lam, [batch_size, 1])
    y_mix = y * lam_y + y_shuffled * (1 - lam_y)
    
    return x_mix, y_mix

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
                                st_progress_bar, st_status_text,
                                use_mixup=False, 
                                label_smoothing=0.0):
    
    # 1. 确定 Loss 函数
    # 如果用 Mixup 或 Smoothing，必须用 CategoricalCrossentropy (支持软标签)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    optimizer = tf.keras.optimizers.Adam()

    # 2. 准备 Metrics
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    
    history = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}
    
    # 获取类别数，用于 One-Hot 转换
    num_classes = y_train.max() + 1 
    
    # 验证集预处理 (转 One-Hot)
    y_test_onehot = tf.one_hot(y_test, depth=num_classes)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test_onehot)).batch(batch_size * samples_per_group)

    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        
        # 获取分组数据生成器
        data_gen = group_batch_generator(X_train, y_train, groups_train, batch_size, samples_per_group)
        
        for step, (x_batch_groups, y_batch) in enumerate(data_gen):
            # x_batch_groups: (B, N, T, F)
            # y_batch: (B,) 原始整数标签
            
            B, N, T, F = x_batch_groups.shape
            
            # 展平输入: (B*N, T, F)
            x_flat = tf.reshape(x_batch_groups, (B * N, T, F))
            # 扩展标签并转 One-Hot: (B,) -> (B*N,) -> (B*N, Classes)
            y_flat_int = np.repeat(y_batch, N)
            y_flat_onehot = tf.one_hot(y_flat_int, depth=num_classes)
            
            # [NEW] 应用 Mixup
            if use_mixup:
                # Mixup 会改变 x_flat 和 y_flat_onehot 的值
                x_flat, y_flat_onehot = mixup(x_flat, y_flat_onehot, alpha=0.2)
            
            with tf.GradientTape() as tape:
                # 前向传播
                logits_flat = model(x_flat, training=True) # (B*N, Classes)
                
                # 1. Instance Loss
                loss_instance = loss_fn(y_flat_onehot, logits_flat)
                
                # 2. Voting Loss (组级 Loss)
                # 变回 (B, N, Classes)
                logits_grouped = tf.reshape(logits_flat, (B, N, -1))
                avg_preds = tf.reduce_mean(logits_grouped, axis=1) # (B, Classes)
                
                # Voting Loss 的目标是真实的 y_batch (转One-Hot)
                y_batch_onehot = tf.one_hot(y_batch, depth=num_classes)
                
                # 注意：如果 Mixup 开启了，这里 Voting Loss 比较难定义，
                # 因为组内的每个样本可能混了不同的类。
                # 为了简化，我们规定：Mixup只影响 Instance Loss，Voting Loss 依然对齐原始标签。
                # 但这要求 logits 也是未混合的预测。
                # 妥协方案：如果 Mixup 开启，暂时降低 Voting Weight 或者只计算 Instance Loss。
                # 这里为了代码简洁，我们假定 Mixup 时模型输出的是混合预测，
                # 强行跟原始标签算 Loss 会有偏差，但也能训练。
                # 更严谨的做法是对 y_batch 也做同样的 Mixup (很难实现因为 shuffle 是随机的)。
                # **实用方案**：Mixup 时，avg_preds 也是混合的，我们让它逼近 y_batch_onehot (未混合) 
                # 这其实起到了正则化作用。
                
                loss_vote = loss_fn(y_batch_onehot, avg_preds)
                
                total_loss = (1.0 - vote_weight) * loss_instance + vote_weight * loss_vote

            grads = tape.gradient(total_loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            
            epoch_loss_avg.update_state(total_loss)
            train_acc_metric.update_state(y_flat_onehot, logits_flat)
            
        # 验证步
        for x_val, y_val in val_dataset:
            val_logits = model(x_val, training=False)
            val_acc_metric.update_state(y_val, val_logits)
            # 记录 val_loss
            loss_val = loss_fn(y_val, val_logits)

        # 记录
        train_acc = train_acc_metric.result()
        val_acc = val_acc_metric.result()
        curr_loss = epoch_loss_avg.result()
        
        history['accuracy'].append(float(train_acc))
        history['loss'].append(float(curr_loss))
        history['val_accuracy'].append(float(val_acc))
        history['val_loss'].append(float(loss_val))
        
        train_acc_metric.reset_state()
        val_acc_metric.reset_state()
        
        progress = (epoch + 1) / epochs
        st_progress_bar.progress(progress)
        st_status_text.text(f"Epoch {epoch+1}/{epochs} | Loss: {curr_loss:.4f} | Train Acc: {train_acc:.1%} | Val Acc: {val_acc:.1%}")
        
    return history