import tensorflow as tf
import numpy as np
import pandas as pd

class EMGGestureClassifier:
    def __init__(self, model_path):
        """
        加载预训练模型
        :param model_path: 模型文件路径（.h5 或 SavedModel 目录）
        """
        # 抑制TensorFlow的冗余日志
        tf.get_logger().setLevel('ERROR')
        self.model = tf.keras.models.load_model("D:\develop\pythonSample\EMGGNN\emg_gesture_model")
        self.label_map = {0: 'i', 1: 'b', 2: 'h', 3: 'e'}  # 与训练时letters顺序一致

    def preprocess(self, raw_data):
        """
        数据预处理（与训练时完全一致）
        :param raw_data: 原始肌电数据，形状需为 (3000, 4) 的numpy数组
        :return: 标准化后的数据，形状 (1, 3000, 4)
        """
        if raw_data.shape != (3000, 4):
            raise ValueError(f"输入数据形状需为 (3000, 4)，当前形状: {raw_data.shape}")

        data = raw_data.astype(np.float32)
        # 按通道标准化
        for channel in range(4):
            data[:, channel] = (data[:, channel] - np.mean(data[:, channel])) / np.std(data[:, channel])
        return np.expand_dims(data, axis=0)  # 添加batch维度

    def predict(self, data):
        """
        执行预测
        :param data: 预处理后的数据（形状 (1, 3000, 4)）
        :return: 预测结果字典
        """
        if data.shape != (1, 3000, 4):
            raise ValueError(f"输入数据形状需为 (1, 3000, 4)，当前形状: {data.shape}")

        probabilities = self.model.predict(data, verbose=0)[0]
        pred_class = np.argmax(probabilities)
        return {
            "label": self.label_map[pred_class],
            "confidence": float(probabilities[pred_class]),
            "probabilities": {
                label: float(prob) for label, prob in zip(self.label_map.values(), probabilities)
            }
        }

    def predict_from_csv(self, csv_path):
        """
        直接从CSV文件预测
        :param csv_path: CSV文件路径
        """
        df = pd.read_csv(csv_path, header=None)
        raw_data = df.values
        processed = self.preprocess(raw_data)
        return self.predict(processed)

