import time
import os
import numpy as np
import streamlit as st
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import GroupShuffleSplit

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
        elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed_time))
        
        loss = logs.get('loss', 0)
        acc = logs.get('accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        val_acc = logs.get('val_accuracy', 0)
        
        status_msg = (
            f"Epoch {current_epoch}/{self.total_epochs} | "
            f"⏳ 剩余: {eta_str} (已用: {elapsed_str}) | "
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
    
    # --- 策略 1: 混合大乱炖 ---
    if strategy == "混合切分 (看到所有天/人)":
        for f in unique_files:
            f_indices = indices[groups == f]
            split_point = int(len(f_indices) * (1 - test_size))
            train_idx.extend(f_indices[:split_point])
            test_idx.extend(f_indices[split_point:])
            
    # --- 策略 2: 严格留一文件 ---
    elif strategy == "留文件验证 (同天/同人)":
        if manual_target:
            is_test = np.array([os.path.basename(g.split('_seg')[0]) == manual_target for g in groups])
            test_idx = indices[is_test]
            train_idx = indices[~is_test]
        else:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
            train_i, test_i = next(gss.split(X, y, groups=groups))
            train_idx, test_idx = indices[train_i], indices[test_i]

    # --- 策略 3: 严格留一日期/对象 ---
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
    """
    为每个类别提取固定数量的样本用于微调
    """
    train_idx = []
    test_idx = []
    unique_labels = np.unique(y)
    
    for label in unique_labels:
        label_indices = np.where(y == label)[0]
        np.random.shuffle(label_indices)
        train_idx.extend(label_indices[:n_samples_per_class])
        test_idx.extend(label_indices[n_samples_per_class:])
        
    return np.array(train_idx), np.array(test_idx)