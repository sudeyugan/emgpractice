import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

# ================= 辅助模块：SE-Block (注意力机制) =================
def se_block(input_tensor, ratio=16):
    """ SE-Attention 模块 (保持你之前的代码或新增) """
    channel_axis = -1
    filters = input_tensor.shape[channel_axis]
    se = layers.GlobalAveragePooling1D()(input_tensor)
    se = layers.Reshape((1, filters))(se)
    se = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = layers.Multiply()([input_tensor, se])
    return x

def residual_block(x, filters, kernel_size=3, stride=1):
    """ 残差块 (ResNet Block) """
    shortcut = x
    # 如果维度不匹配 (stride>1 或 filters变化)，对 shortcut 做 1x1 卷积调整
    if stride > 1 or x.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same')(x)
        shortcut = layers.BatchNormalization()(shortcut)

    # 主路径
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv1D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # 核心：Add
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def tcn_block(x, filters, kernel_size=3, dilation_rate=1):
    """ TCN 块 (膨胀卷积) """
    prev_x = x
    if x.shape[-1] != filters:
        prev_x = layers.Conv1D(filters, 1, padding='same')(x)
        
    # 膨胀卷积 1
    x = layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # 膨胀卷积 2
    x = layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # 残差连接
    x = layers.Add()([x, prev_x])
    return x

# ================= 1. 轻量级模型 (保持不变) =================
def build_simple_cnn(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv1D(filters=32, kernel_size=5, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(filters=64, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Simple_CNN")
    return model

# ================= 2. 重量级模型 (已升级 SE-Attention) =================
def build_advanced_crnn(input_shape, num_classes):
    """
    进阶 Multi-Scale CRNN + SE Attention
    """
    inputs = layers.Input(shape=input_shape)
    
    # --- 多尺度分支 (Inception Style) ---
    # 分支1: 小感受野
    b1 = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    b1 = layers.MaxPooling1D(2)(b1)
    
    # 分支2: 中感受野
    b2 = layers.Conv1D(32, kernel_size=7, padding='same', activation='relu')(inputs)
    b2 = layers.MaxPooling1D(2)(b2)

    # 分支3: 大感受野
    b3 = layers.Conv1D(32, kernel_size=11, padding='same', activation='relu')(inputs)
    b3 = layers.MaxPooling1D(2)(b3)
    
    # 融合特征
    x = layers.Concatenate()([b1, b2, b3])
    
    # [NEW] 插入注意力机制！让模型自动给好用的特征加权
    x = se_block(x, ratio=8) 
    
    x = layers.Dropout(0.3)(x)
    
    # 整合特征
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    
    # 时序建模 (LSTM)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.4)(x)
    
    # 分类头
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Advanced_CRNN_SE")
    return model

def build_resnet_model(input_shape, num_classes):
    """ 1. ResNet-1D 模型 """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)
    
    # 堆叠残差块
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs, name="ResNet1D")

def build_tcn_model(input_shape, num_classes):
    """ 2. TCN 模型 """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 3, padding='same')(inputs)
    
    # 堆叠膨胀卷积，dilation 指数级增长: 1, 2, 4, 8, 16
    x = tcn_block(x, 64, dilation_rate=1)
    x = tcn_block(x, 64, dilation_rate=2)
    x = tcn_block(x, 128, dilation_rate=4)
    x = tcn_block(x, 128, dilation_rate=8)
    x = tcn_block(x, 256, dilation_rate=16)
    
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs, name="TCN")

def build_dual_stream_model(input_shape, num_classes):
    """ 3. 双流网络 (时域 + 频域融合) """
    inputs = layers.Input(shape=input_shape)
    
    # --- 分支 A: 时域 (Time Domain) ---
    # 使用轻量级 CNN 处理原始波形
    t = layers.Conv1D(32, 5, padding='same', activation='relu')(inputs)
    t = layers.MaxPooling1D(2)(t)
    t = layers.Conv1D(64, 3, padding='same', activation='relu')(t)
    t = layers.GlobalAveragePooling1D()(t) # (Batch, 64)
    
    # --- 分支 B: 频域 (Freq Domain) ---
    # 在模型内部动态计算 STFT，不需要修改 Data Loader
    # input shape: (Batch, Time, Channels) -> permute to calculate STFT per channel
    # 为了简化，我们先对所有通道求平均，或者把通道视为 batch 维度的一部分
    # 这里采用一种简单策略：对每个通道分别做 STFT 然后 2D 卷积
    
    # 1. 调整维度适应 STFT: (Batch, Time, Channels) -> (Batch, Time) (如果是多通道，这里做一个简单融合或者分别处理)
    # 简单起见，这里先用 1x1 卷积把多通道融合为单通道信号，再做 STFT
    merged_channel = layers.Conv1D(1, 1)(inputs) # (B, T, 1)
    merged_channel = layers.Reshape((-1,))(merged_channel) # (B, T)
    
    # 2. 计算 STFT
    # frame_length=64, frame_step=32 对应约 64ms 窗口
    stft = layers.Lambda(lambda x: tf.signal.stft(x, frame_length=256, frame_step=32))(merged_channel)
    stft = layers.Lambda(lambda x: tf.abs(x))(stft) # 取幅值 (B, T_frames, Freq_bins)
    stft = layers.Lambda(lambda x: tf.expand_dims(x, -1))(stft) # 增加通道维变成图片 (B, H, W, 1)
    
    # 3. 2D CNN 处理频谱图
    f = layers.Conv2D(32, (3,3), activation='relu', padding='same')(stft)
    f = layers.MaxPooling2D((2,2))(f)
    f = layers.Conv2D(64, (3,3), activation='relu', padding='same')(f)
    f = layers.GlobalAveragePooling2D()(f) # (Batch, 64)
    
    # --- 融合 ---
    merged = layers.Concatenate()([t, f]) # (Batch, 128)
    merged = layers.Dropout(0.5)(merged)
    merged = layers.Dense(64, activation='relu')(merged)
    outputs = layers.Dense(num_classes, activation='softmax')(merged)
    
    return models.Model(inputs, outputs, name="DualStream_TimeFreq")