import tensorflow as tf
from tensorflow.keras import layers, models

# ================= 1. 轻量级模型 (适合单人/小数据) =================
def build_simple_cnn(input_shape, num_classes):
    """
    经典 Simple CNN：结构简单，参数少，不容易在小数据上过拟合。
    """
    inputs = layers.Input(shape=input_shape)
    
    # Layer 1
    x = layers.Conv1D(filters=32, kernel_size=5, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)
    
    # Layer 2
    x = layers.Conv1D(filters=64, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)
    
    # Flatten & Dense
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Simple_CNN")
    return model

# ================= 2. 重量级模型 (适合多人/跨天/大数据) =================
def build_advanced_crnn(input_shape, num_classes):
    """
    进阶 Multi-Scale CRNN：多尺度卷积 + LSTM，捕捉复杂时序特征。
    """
    inputs = layers.Input(shape=input_shape)
    
    # --- 多尺度分支 ---
    b1 = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    b1 = layers.MaxPooling1D(2)(b1)
    
    b2 = layers.Conv1D(32, kernel_size=7, padding='same', activation='relu')(inputs)
    b2 = layers.MaxPooling1D(2)(b2)

    b3 = layers.Conv1D(32, kernel_size=11, padding='same', activation='relu')(inputs)
    b3 = layers.MaxPooling1D(2)(b3)
    
    # 融合
    x = layers.Concatenate()([b1, b2, b3])
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
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Advanced_CRNN")
    return model