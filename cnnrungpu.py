import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import os

# 配置 GPU 显存动态增长
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 数据加载函数（优化为 tf.data）
def load_data(base_dir, letters, batch_size=32):
    X = []
    y = []
    for label, letter in enumerate(letters):
        letter_dir = os.path.join(base_dir, letter)
        for file in os.listdir(letter_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(letter_dir, file)
                df = pd.read_csv(file_path, header=None)
                if df.shape[0] == 3000:
                    data = df.values.astype(np.float32)
                    for channel in range(4):
                        data[:, channel] = (data[:, channel] - np.mean(data[:, channel])) / np.std(data[:, channel])
                    X.append(data)
                    y.append(label)
    X = np.array(X)
    y = np.array(y)
    y = tf.keras.utils.to_categorical(y, len(letters))  # 转换为 One-hot 编码
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# 加载数据为 Dataset
base_dir = "D:/develop/pythonSample/EMGGNN"
letters = ['i', 'b', 'h', 'e']
dataset = load_data(base_dir, letters, batch_size=32)

# 划分训练集和测试集
num_samples = len(dataset) * 32  # 计算总样本数量
train_size = int(0.8 * num_samples)
train_dataset = dataset.take(train_size // 32)  # 按批次划分
test_dataset = dataset.skip(train_size // 32)

# 模型构建
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

# 编译与训练
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_dataset, epochs=25, validation_data=test_dataset)

# 评估
loss, accuracy = model.evaluate(test_dataset)
print(f"测试准确率: {accuracy:.4f}")

# 或保存为 HDF5 格式
model.save("emg_gesture_model.h5")