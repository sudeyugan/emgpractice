import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(input_shape, num_classes):
    """
    input_shape: (256, 5)
    num_classes: 动作种类数
    """
    inputs = layers.Input(shape=input_shape)
    
    # 第一层卷积
    x = layers.Conv1D(filters=32, kernel_size=5, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x) # 防止过拟合
    
    # 第二层卷积
    x = layers.Conv1D(filters=64, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)
    
    # 特征扁平化
    x = layers.GlobalAveragePooling1D()(x) # 或者 Flatten()
    
    # 全连接层
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model