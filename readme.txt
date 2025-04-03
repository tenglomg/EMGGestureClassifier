dataread.py  segmentation 这两个文件是将采集到的原始数据读取且分段的，读者可以忽略
"EMGGNN\i.csv"（b,h,e同理）是采集到的原始数据的原文件(Excel文件)
"EMGGNN\i"（b,h,e同理）是经过预处理并分割成功的文件夹(包含了分段后的csv文件)
cnnrnn.py是模型训练过程   cnnrnngpu.py是GPU加速的模型训练过程
"D:\develop\pythonSample\EMGGNN\emg_gesture_model"是加载的训练成功的模型参数
EMGGestureClassifier.py 是将训练成功的模型封装写成的一个肌电图分类器
realtimeprocess是实时用肌电图分类器对输入数据进行处理的程序

输入: 4通道的1.5秒(3000点)肌电数据   (通道1: BackInside 通道2:BackOutside 通道3: FrontInside 通道4: FrontOutside)     Back:手臂背面 Inside:手臂内侧  Front     Outsidet同理
输出:  
格式举例:
手势识别结果: h (置信度: 33.65%)
各类别概率: {'i': 0.2003810852766037, 'b': 0.1444631963968277, 'h': 0.3365476727485657, 'e': 0.3186081051826477}


