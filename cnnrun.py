import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import os

# 数据加载函数
def load_data(base_dir, letters):
    X = []
    y = []
    for label, letter in enumerate(letters):
        letter_dir = os.path.join(base_dir, letter)
        for file in os.listdir(letter_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(letter_dir, file)
                df = pd.read_csv(file_path, header=None)
                if df.shape[0] == 3000:  # 确保数据完整
                    # 数据标准化（按通道）
                    data = df.values.astype(np.float32)
                    for channel in range(4):
                        data[:, channel] = (data[:, channel] - np.mean(data[:, channel])) / np.std(data[:, channel])
                    X.append(data)
                    y.append(label)
    return np.array(X), np.array(y)

# 加载数据
base_dir = "D:/develop/pythonSample/EMGGNN"
letters = ['i', 'b', 'h', 'e']
X, y = load_data(base_dir, letters)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 转换为One-hot编码
y_train = tf.keras.utils.to_categorical(y_train, 4)
y_test = tf.keras.utils.to_categorical(y_test, 4)

# 构建CNN模型
model = Sequential([
    Conv1D(32, 5, activation='relu', input_shape=(3000, 4)),
    BatchNormalization(),
    MaxPooling1D(2),
    Conv1D(64, 5, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

# 编译模型
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型
history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"测试准确率: {accuracy:.4f}")