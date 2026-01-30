import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

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
