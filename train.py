import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# 引用刚才写的模型
from model import build_cnn_model # 假设上面的文件名叫 model_definition.py

# 1. 加载处理好的数据 (秒级加载，无需等待预处理)
print("Loading data...")
X = np.load('processed_data/X.npy')
y = np.load('processed_data/y.npy')

# 2. 划分训练/测试集
# 注意：这里简单的随机划分会有数据泄露风险，最好是按人或按文件划分
# 为了演示先用这个
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 构建模型
num_classes = len(np.unique(y))
model = build_cnn_model(input_shape=(X.shape[1], X.shape[2]), num_classes=num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 4. 训练
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[
        EarlyStopping(patience=10, restore_best_weights=True)
    ]
)

# 5. 画图看结果
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.show()

# 6. 保存模型
model.save('emg_cnn_model.h5')