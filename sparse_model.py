import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    """ 模拟 Keras 的 padding='causal' """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=self.padding, dilation=dilation)
    
    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            return x[:, :, :-self.padding] # 移除右侧多余的 padding
        return x

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(0.2)
        
        # 处理残差连接的维度不匹配 (类似 Keras 中的 1x1 Conv)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.dropout2(out)
        
        return F.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        # 初始卷积
        self.init_conv = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        
        # 堆叠 TCN 块
        self.block1 = TCNBlock(64, 64, dilation=1)
        self.block2 = TCNBlock(64, 64, dilation=2)
        self.block3 = TCNBlock(64, 128, dilation=4)
        self.block4 = TCNBlock(128, 128, dilation=8)
        self.block5 = TCNBlock(128, 256, dilation=16)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x shape: [batch, input_channels, seq_len]
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x