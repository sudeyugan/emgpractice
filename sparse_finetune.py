import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import glob
import copy
import argparse
from sklearn.metrics import classification_report

# 导入你的模型定义和配置
import sparse_model 
from sparse_auto_train import CONFIG, parse_filename_info, process_files, augment_dataset_in_memory
# ================= 配置区域 =================
# 1. 基础设置
DEFAULT_MODEL_PATH = "sparse_train_logs_pytorch_tcn/best_model.pth"# 你的 .pth 路径
FINETUNE_LR = 0.0001           # 微调通常使用更小的学习率
EPOCHS = 30                    # 微调轮数
BATCH_SIZE = 32

AUGMENT_CONFIG = {
    'enable_rest': True,       # 确保包含静息数据
    'multiplier': 20,          # ⚡ 增强 20 倍
    'enable_scaling': True,    # 开启幅度缩放
    'enable_noise': True,      # 开启高斯噪声
    'enable_warp': False,      # 关闭时间扭曲 (计算量大且微调通常不需要)
    'enable_shift': False,     
    'enable_mask': False       
}

# 2. 微调策略
# True: 全量微调 (所有层都参与更新)
# False: 只训练分类头 (冻结卷积层，适合数据极少的情况)
UNFREEZE_ALL = True            

# 3. 新的数据集 (用于微调的目标)
TARGET_SUBJECTS = ["fred"] # 或者是你想测试的新用户
TARGET_LABELS = [5, 6, 8]   # 动作必须与原模型一致
SHOTS_PER_CLASS = 2            # Few-shot: 每个类别只用 2 个样本训练

# ===========================================

def load_pretrained_model(path, device, num_classes, input_channels):
    print(f"🔄 正在加载基模型: {path}")
    
    # 1. 实例化新模型 (使用当前的 num_classes，例如 4)
    model = sparse_model.TCNModel(
        input_channels=input_channels, 
        num_classes=num_classes
    )
    
    # 2. 加载旧模型的权重字典
    if os.path.exists(path):
        pretrained_dict = torch.load(path, map_location=device)
    else:
        print(f"❌ 找不到权重文件: {path}")
        exit()

    # 3. 获取新模型的权重字典
    model_dict = model.state_dict()
    
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}
    
    # 4. 更新权重
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    
    print(f"✅ 成功加载预训练权重 (已丢弃旧分类头 {len(pretrained_dict)} -> {len(filtered_dict)} 层)")
    
    return model

def prepare_few_shot_data(X, y, shots=5):
    """从数据集中每个类别随机抽取 k 个样本作为训练集，其余作为测试集"""
    train_indices = []
    test_indices = []
    
    unique_labels = np.unique(y)
    for label in unique_labels:
        # 找到该类别的所有索引
        idx = np.where(y == label)[0]
        np.random.shuffle(idx)
        
        if len(idx) >= shots:
            train_indices.extend(idx[:shots])
            test_indices.extend(idx[shots:])
        else:
            # 如果样本不够，全放入训练集（或者报错）
            train_indices.extend(idx)
            
    return train_indices, test_indices

def run_finetuning(base_model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 查找并加载新数据
    search_pattern = os.path.join("data", "*", "*", "RAW_EMG*.csv")
    all_files = glob.glob(search_pattern)
    target_files = []
    for f in all_files:
        s, d, l, _ = parse_filename_info(f)
        if s in TARGET_SUBJECTS and l in TARGET_LABELS:
            target_files.append(f)
            
    if not target_files:
        print(f"❌ 未找到 {TARGET_SUBJECTS} 的数据，无法微调。")
        return

    # 复用 sparse_auto_train 的数据处理函数
    # 注意：微调时通常关闭增强，或者只做轻微增强
    dummy_aug = {'enable_rest': True, 'multiplier': 1}
    
    # ✅ 1. 捕获 groups (增强函数需要用到)
    X_all, y_all, groups_all = process_files(target_files, CONFIG, dummy_aug)
    
    unique_labels = sorted(TARGET_LABELS)
    
    # 检查数据中是否有静息类 (0)，如果有且没包含在目标里，手动加上
    # 这样做能保证 0 (Rest) 映射为 0，5->1, 6->2 ... 与训练时的逻辑对齐
    if 0 in y_all and 0 not in unique_labels:
        unique_labels = [0] + unique_labels
        
    label_map = {orig: new for new, orig in enumerate(unique_labels)}
    
    y_mapped = np.array([label_map[y] for y in y_all])
    
    # 2. 划分 Few-shot 数据集
    train_idx, test_idx = prepare_few_shot_data(X_all, y_mapped, shots=SHOTS_PER_CLASS)
    
    # 提取原始 numpy 数据
    X_train_raw = X_all[train_idx]
    y_train_raw = y_all[train_idx] # 注意：这是原始标签还是映射后的？这里应使用原始y_all对应的值，或者注意增强函数对y的处理
    # 修正：augment_dataset_in_memory 需要原始 y 还是映射后的 y 都可以，但为了保险，我们用映射前的逻辑，或者直接对 mapped 后的做增强（只要它是 array）
    # 简单做法：直接对 split 出来的 numpy array 做增强
    
    groups_train = groups_all[train_idx]
    
    # ✅ 2. 执行内存增强 (仅对训练集)
    print(f"🚀 正在对训练集进行 20 倍增强...")
    # 注意：这里的 y_mapped[train_idx] 已经是映射好的 0-4 标签，可以直接增强
    X_train_aug, y_train_aug, _ = augment_dataset_in_memory(
        X_train_raw, 
        y_mapped[train_idx], # 传入映射后的标签
        groups_train, 
        AUGMENT_CONFIG
    )
    
    print(f"📈 增强后训练样本数: {len(X_train_aug)}")

    # ✅ 3. 转 Tensor (注意用增强后的数据)
    X_train = torch.FloatTensor(X_train_aug).permute(0, 2, 1) 
    y_train = torch.LongTensor(y_train_aug)
    
    # 测试集保持原样
    X_test = torch.FloatTensor(X_all[test_idx]).permute(0, 2, 1)
    y_test = torch.LongTensor(y_mapped[test_idx])
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
    
    print(f"📊 微调数据: 训练集 {len(train_idx)} (每个类 {SHOTS_PER_CLASS} 个), 测试集 {len(test_idx)}")
    
    # 3. 加载模型
    input_channels = X_train.shape[1] # 11 (IMU) or 8 (EMG)
    num_classes = len(unique_labels)
    model = load_pretrained_model(base_model_path, device, num_classes, input_channels)
    model.to(device)
    
    # 4. 冻结/解冻策略
    if not UNFREEZE_ALL:
        print("❄️ 冻结特征提取层，仅训练分类头...")
        for name, param in model.named_parameters():
            # 假设最后一层叫 'fc' 或 'linear'，其他都冻结
            if 'fc' not in name and 'classifier' not in name:
                param.requires_grad = False
    else:
        print("🔥 全量微调: 所有参数参与更新")

    # 5. 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 6. 微调循环
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "finetuned_best.pth")
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Val Acc: {acc:.4f}")
            
    print(f"✅ 微调完成！最佳准确率: {best_acc:.4f}")
    print("模型已保存为 finetuned_best.pth")

if __name__ == "__main__":
    # ✅ 添加参数解析
    parser = argparse.ArgumentParser(description="Few-shot Finetuning Script")
    parser.add_argument(
        '--model_path', 
        type=str, 
        default=DEFAULT_MODEL_PATH,
        help='Path to the pretrained base model weights (.pth)'
    )
    
    args = parser.parse_args()
    
    # ✅ 将解析到的路径传给函数
    run_finetuning(base_model_path=args.model_path)
