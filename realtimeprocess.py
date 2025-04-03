from EMGGestureClassifier import EMGGestureClassifier
import numpy as np

# 初始化分类器
classifier = EMGGestureClassifier("emg_gesture_model.h5")  # 或使用SavedModel路径

# 模拟实时数据（3000行 x 4列）
sample_data = np.random.randn(3000, 4)  # 替换为实际采集数据

# 预处理 + 预测
try:
    processed = classifier.preprocess(sample_data)
    result = classifier.predict(processed)
    print(f"手势识别结果: {result['label']} (置信度: {result['confidence']:.2%})")
    print("各类别概率:", result['probabilities'])
except ValueError as e:
    print(f"输入数据错误: {str(e)}")

