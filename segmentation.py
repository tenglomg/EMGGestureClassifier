import pandas as pd
import numpy as np
import os

# 参数设置
sampling_rate = 2000
window_length = 1.5  # 秒
window_step = 0.25   # 秒
total_duration = 64  # 秒

# 计算窗口对应的样本点数
window_size = int(window_length * sampling_rate)  # 3000
step_size = int(window_step * sampling_rate)      # 500

# 输入输出路径配置
base_dir = "D:/develop/pythonSample/EMGGNN"
letters = ['i', 'b', 'h', 'e']

for letter in letters:
    # 输入文件路径
    input_file = os.path.join(base_dir, f"{letter}.csv")
    # 输出目录路径
    output_dir = os.path.join(base_dir, letter)
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    data = pd.read_csv(input_file, header=None)
    num_samples = data.shape[0]
    
    # 生成滑动窗口
    start_idx = 0
    file_count = 0
    while start_idx + window_size <= num_samples:
        end_idx = start_idx + window_size
        window_data = data.iloc[start_idx:end_idx, :]
        # 保存窗口数据
        output_path = os.path.join(output_dir, f"{letter}_{file_count:04d}.csv")
        window_data.to_csv(output_path, index=False, header=False)
        # 更新索引和计数器
        start_idx += step_size
        file_count += 1