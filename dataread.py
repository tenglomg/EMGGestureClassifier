import pandas as pd

# 定义文件路径
file_paths = {
    'BackInside': r"C:\Users\Tenglong\OneDrive\桌面\手势肌电\BackInside_1.csv",
    'BackOutside': r"C:\Users\Tenglong\OneDrive\桌面\手势肌电\BackOutside_1.csv",
    'FrontInside': r"C:\Users\Tenglong\OneDrive\桌面\手势肌电\FrontInside_1.csv",
    'FrontOutside': r"C:\Users\Tenglong\OneDrive\桌面\手势肌电\FrontOutside_1.csv"
}

# 读取四个通道的数据，假设每个文件只有一列数据
channels = {}
for name, path in file_paths.items():
    channels[name] = pd.read_csv(path, header=None, names=[name])

# 生成每个字母的时间段
def get_intervals(letter):
    intervals = []
    for k in range(32):  # k从0到31
        if letter == 'i':
            start = 0 + 16 * k
            end = 2 + 16 * k
        elif letter == 'b':
            start = 4 + 16 * k
            end = 6 + 16 * k
        elif letter == 'h':
            start = 8 + 16 * k
            end = 10 + 16 * k
        elif letter == 'e':
            start = 12 + 16 * k
            end = 14 + 16 * k
        # 确保时间不超过512秒
        if end <= 512:
            intervals.append((start, end))
    return intervals

# 处理每个字母
for letter in ['i', 'b', 'h', 'e']:
    intervals = get_intervals(letter)
    all_data = pd.DataFrame()
    
    for start_sec, end_sec in intervals:
        start_row = int(start_sec * 2000)
        end_row = int(end_sec * 2000)
        
        # 提取各通道数据并合并
        segments = []
        for channel in channels.values():
            segment = channel.iloc[start_row:end_row].reset_index(drop=True)
            segments.append(segment)
        combined = pd.concat(segments, axis=1)
        
        all_data = pd.concat([all_data, combined], ignore_index=True)
    
    # 保存为CSV文件
    output_path = f'{letter}.csv'
    all_data.to_csv(output_path, index=False, header=False)
    print(f'字母 {letter} 的数据已保存到 {output_path}')