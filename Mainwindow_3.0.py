import sys
import numpy as np
import pyqtgraph as pg
import os
import datetime
from PyQt6.QtCore import Qt, QTimer, QSettings, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QSplitter,
    QLabel,
    QPushButton,
    QFileDialog,
    QComboBox,
    QSpinBox,
    QLineEdit,
    QTextEdit,
    QGroupBox,
    QMenuBar,
    QMenu,
    QDialog,
    QCheckBox,
)
from PyQt6.QtGui import QAction
from nidaqmx import Task
from nidaqmx.constants import AcquisitionType

# 新增识别控制类
class GestureRecognitionController(QThread):
    recognition_success = pyqtSignal(str)  # 识别成功信号

    def __init__(self, task, sample_rate=1000, buffer_size=200):
        super().__init__()
        self.task = task
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.running = False
        self.recognizer = self.GestureRecognition()

    def run(self):
        """持续采集并识别数据"""
        self.running = True
        while self.running:
            data = self.task.read(number_of_samples_per_channel=self.buffer_size)
            gesture = self.recognizer.process_data(np.array(data))
            if gesture != "Unknown":
                self.recognition_success.emit(gesture)
                break  # 识别成功时退出循环

    def stop(self):
        self.running = False
    def GestureRecognition(self):
        return

class MultiChannelWaveformDisplay(QWidget):
    def __init__(self, num_channels=4):
        super().__init__()
        self.num_channels = num_channels
        self.init_ui()
        self.selected_channel = None  # 当前选中的通道

    def init_ui(self):
        """初始化波形显示界面"""
        self.layout = QVBoxLayout(self)

        # 创建四个子绘图区域
        self.plots = []
        self.curves = []
        for i in range(self.num_channels):
            plot = pg.PlotWidget(title=f"Channel {i+1}")
            plot.setLabel('left', 'Amplitude', 'V')
            plot.setLabel('bottom', 'Time', 's')
            plot.setMouseEnabled(x=False, y=False)  # 禁用默认缩放
            plot.setMenuEnabled(False)  # 禁用右键菜单
            plot.setYRange(-1, 1)  # 初始Y轴范围
            plot.setXRange(0, 1)   # 初始X轴范围

            # 绑定点击事件
            plot.scene().sigMouseClicked.connect(
                lambda event, idx=i: self.on_plot_clicked(event, idx)
            )

            curve = plot.plot(pen=pg.mkPen(color=(i, self.num_channels), width=2))
            self.plots.append(plot)
            self.curves.append(curve)
            self.layout.addWidget(plot)

    def on_plot_clicked(self, event, channel_idx):
        """处理绘图区域点击事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected_channel = channel_idx
            self.highlight_selected_channel()

    def highlight_selected_channel(self):
        """高亮选中的通道"""
        for i, plot in enumerate(self.plots):
            if i == self.selected_channel:
                plot.setTitle(f"Channel {i+1} (Selected)", color="r")
                plot.setYRange(-2, 2)  # 放大Y轴范围
            else:
                plot.setTitle(f"Channel {i+1}", color="k")
                plot.setYRange(-1, 1)  # 恢复默认范围

    
    def update_waveforms(self, time_axis, data):
        """更新波形数据，支持动态时间轴"""
        if data.size == 0:
            return

        # 更新每条曲线
        for i, curve in enumerate(self.curves):
            curve.setData(time_axis, data[i])

            # 动态调整Y轴范围（可选）
            y_min, y_max = np.min(data[i]), np.max(data[i])
            margin = 0.1 * (y_max - y_min) if y_max != y_min else 0.5
            self.plots[i].setYRange(y_min - margin, y_max + margin)

        # 自动滚动X轴（关键！）
        for plot in self.plots:
            plot.setXRange(time_axis[0], time_axis[-1], padding=0)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.auto_save_counter = 1
        self.init_settings()
        self.init_ui()
        self.init_data()
        self.init_gesture_control()

    def init_gesture_control(self):
        """初始化手势识别控制器"""
        self.gesture_task = Task()
        self.gesture_task.ai_channels.add_ai_voltage_chan("Dev1/ai0:3")
        self.gesture_task.timing.cfg_samp_clk_timing(
            rate=1000,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=200
        )
        self.recognition_controller = None

    def toggle_recognition(self):
        """切换识别状态"""
        if not self.recognition_controller or not self.recognition_controller.isRunning():
            self.start_recognition()
        else:
            self.stop_recognition()

    def start_recognition(self):
        """启动识别流程"""
        # 重置界面状态
        self.recognition_display.setText("识别中...")
        self.start_recog_btn.setText("停止识别")
        self.start_recog_btn.setStyleSheet("background-color: #f44336;")

        # 启动识别线程
        self.recognition_controller = GestureRecognitionController(self.gesture_task)
        self.recognition_controller.recognition_success.connect(self.on_recognition_success)
        self.recognition_controller.start()

    def stop_recognition(self):
        """停止识别流程"""
        if self.recognition_controller:
            self.recognition_controller.stop()
            self.recognition_controller.wait()
        self.recognition_display.setText("已停止")
        self.start_recog_btn.setText("开始识别")
        self.start_recog_btn.setStyleSheet("background-color: #4CAF50;")

    def on_recognition_success(self, gesture):
        """识别成功处理"""
        self.recognition_display.setText(f"识别成功：{gesture}")
        self.stop_recognition()

    def closeEvent(self, event):
        """窗口关闭时释放资源"""
        self.stop_recognition()
        
        # 安全关闭数据采集任务
        if hasattr(self, 'acq_task') and self.acq_task:
            try:
                self.acq_task.stop()
                self.acq_task.close()
                self.acq_task = None  # 清除引用
            except Exception as e:
                print(f"关闭任务时出错: {e}")
        
        # 关闭手势识别任务
        if hasattr(self, 'gesture_task') and self.gesture_task:
            self.gesture_task.close()
        
        super().closeEvent(event)
    
    def init_settings(self):
        """加载保存配置"""
        self.settings = QSettings("MyCompany", "EMG_Recorder")
        self.save_path = self.settings.value("save_path", os.path.expanduser("~"))
        self.max_file_size = int(self.settings.value("max_file_size", 100))  # 单位MB
        self.auto_save_enabled = self.settings.value("auto_save_enabled", "false") == "true"  # 自动保存状态
        self.auto_save_interval = int(self.settings.value("auto_save_interval", 5))  # 自动保存间隔（秒）

    def init_ui(self):
        """初始化主界面"""
        self.setWindowTitle("EMG Data Acquisition System")
        self.setGeometry(100, 100, 1200, 600)
        """界面新增识别控制按钮"""
        # 在右侧识别区域添加
        self.start_recog_btn = QPushButton("开始识别", self)
        self.start_recog_btn.clicked.connect(self.toggle_recognition)
        self.start_recog_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 12px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #45a049; }
        """)

        # === 菜单栏 ===
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件")

        # 保存波形数据
        save_action = QAction("保存波形数据", self)
        save_action.triggered.connect(self.save_data)
        file_menu.addAction(save_action)

        # 自动保存设置
        auto_save_action = QAction("自动保存设置", self)
        auto_save_action.triggered.connect(self.show_auto_save_dialog)
        file_menu.addAction(auto_save_action)

        # 自动保存开关
        self.auto_save_toggle_action = QAction("启用自动保存", self)
        self.auto_save_toggle_action.setCheckable(True)
        self.auto_save_toggle_action.setChecked(self.auto_save_enabled)
        self.auto_save_toggle_action.triggered.connect(self.toggle_auto_save)
        file_menu.addAction(self.auto_save_toggle_action)

        # 退出
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # === 主布局 ===
        main_widget = QWidget()
        self.layout = QHBoxLayout(main_widget)

        # 左侧波形显示区域（保持不变）
        self.waveform_display = MultiChannelWaveformDisplay(num_channels=4)
        self.layout.addWidget(self.waveform_display, 70)  # 70%宽度

        # 右侧控制面板（重新布局）
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.insertWidget(2, self.start_recog_btn)  # 插入到原有布局中

        # 1. 识别结果区域
        recognition_group = QGroupBox("手势识别结果")
        recognition_layout = QVBoxLayout(recognition_group)
        self.recognition_display = QLabel("当前手势：无")
        self.recognition_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recognition_display.setStyleSheet("font-size: 24px; color: #2c3e50;")
        recognition_layout.addWidget(self.recognition_display)
        right_layout.addWidget(recognition_group)

        # 2. 状态显示区域
        status_group = QGroupBox("系统状态")
        status_layout = QVBoxLayout(status_group)
        
        # 自动保存状态
        self.auto_save_status = QLabel("自动保存：关闭")
        self.auto_save_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 采集状态
        self.acquisition_status = QLabel("采集状态：运行中")
        self.acquisition_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        status_layout.addWidget(self.auto_save_status)
        status_layout.addWidget(self.acquisition_status)
        right_layout.addWidget(status_group)

        # 3. 控制按钮区域
        control_group = QGroupBox("控制面板")
        control_layout = QVBoxLayout(control_group)
        
        # 暂停/恢复按钮
        self.pause_btn = QPushButton("暂停采集")
        self.pause_btn.setStyleSheet(
            "QPushButton { background-color: #e74c3c; color: white; border-radius: 5px; padding: 8px; }"
            "QPushButton:hover { background-color: #c0392b; }"
        )
        self.pause_btn.clicked.connect(self.toggle_pause)
        
        # 紧急停止按钮（示例）
        emergency_stop_btn = QPushButton("紧急停止")
        emergency_stop_btn.setStyleSheet(
            "QPushButton { background-color: #34495e; color: white; border-radius: 5px; padding: 8px; }"
            "QPushButton:hover { background-color: #2c3e50; }"
        )
        
        control_layout.addWidget(self.pause_btn)
        control_layout.addWidget(emergency_stop_btn)
        right_layout.addWidget(control_group)

        # 设置右侧面板比例
        right_layout.setStretch(0, 3)  # 识别结果区域占3份
        right_layout.setStretch(1, 2)  # 状态区域占2份
        right_layout.setStretch(2, 1)  # 控制区域占1份

        self.layout.addWidget(right_panel, 30)  # 右侧占30%宽度
        self.setCentralWidget(main_widget)

        # 初始化状态显示
        self.update_auto_save_status()

    def toggle_pause(self):
        """切换暂停状态"""
        self.waveform_display.toggle_pause()
        if self.waveform_display.is_paused:
            self.pause_btn.setText("恢复采集")
            self.pause_btn.setStyleSheet("background-color: #27ae60; color: white;")
            self.acquisition_status.setText("采集状态：已暂停")
        else:
            self.pause_btn.setText("暂停采集")
            self.pause_btn.setStyleSheet("background-color: #e74c3c; color: white;")
            self.acquisition_status.setText("采集状态：运行中")
  

    def toggle_auto_save(self):
        """切换自动保存状态"""
        self.auto_save_enabled = not self.auto_save_enabled
        self.settings.setValue("auto_save_enabled", "true" if self.auto_save_enabled else "false")

        if self.auto_save_enabled:
            self.auto_save_timer.start(self.auto_save_interval * 1000)
        else:
            self.auto_save_timer.stop()

        self.update_auto_save_status()

    def update_auto_save_status(self):
        """更新自动保存状态显示"""
        status = "开启" if self.auto_save_enabled else "关闭"
        color = "#27ae60" if self.auto_save_enabled else "#e74c3c"
        self.auto_save_status.setText(
            f"自动保存：<span style='color: {color}; font-weight: bold;'>{status}</span>"
        )
        self.auto_save_status.setToolTip(
            f"间隔：{self.auto_save_interval}秒 | 路径：{self.save_path}"
        )

    def update_auto_save_status(self):
        """更新自动保存状态显示"""
        status = "开启" if self.auto_save_enabled else "关闭"
        color = "green" if self.auto_save_enabled else "red"
        self.auto_save_status.setText(f"自动保存状态：<font color='{color}'>{status}</font>")
        self.auto_save_toggle_action.setChecked(self.auto_save_enabled)

    def show_auto_save_dialog(self):
        """显示自动保存设置对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("自动保存设置")
        layout = QVBoxLayout(dialog)

        # 启用复选框
        enable_check = QCheckBox("启用自动保存")
        enable_check.setChecked(self.auto_save_enabled)
        layout.addWidget(enable_check)

        # 保存间隔设置
        interval_spin = QSpinBox()
        interval_spin.setRange(1, 3600)
        interval_spin.setValue(self.auto_save_interval)
        layout.addWidget(QLabel("保存间隔（秒）:"))
        layout.addWidget(interval_spin)

        # 保存路径设置
        path_edit = QLineEdit(self.save_path)
        path_btn = QPushButton("浏览...")
        path_btn.clicked.connect(lambda: self.set_save_path(path_edit))
        layout.addWidget(QLabel("保存路径:"))
        layout.addWidget(path_edit)
        layout.addWidget(path_btn)

        # 文件分割大小设置
        file_split_spin = QSpinBox()
        file_split_spin.setRange(1, 1024)
        file_split_spin.setValue(self.max_file_size)
        layout.addWidget(QLabel("文件分割大小 (MB):"))
        layout.addWidget(file_split_spin)

        # 确认按钮
        confirm_btn = QPushButton("确认")
        confirm_btn.clicked.connect(lambda: self.update_auto_save_settings(
            enable_check.isChecked(),
            interval_spin.value(),
            path_edit.text(),
            file_split_spin.value()
        ))
        confirm_btn.clicked.connect(dialog.close)
        layout.addWidget(confirm_btn)

        dialog.exec()

    def update_auto_save_settings(self, enabled, interval, path, max_size):
        """更新自动保存配置"""
        self.auto_save_enabled = enabled
        self.auto_save_interval = interval
        self.save_path = path
        self.max_file_size = max_size

        # 保存设置
        self.settings.setValue("auto_save_enabled", "true" if enabled else "false")
        self.settings.setValue("auto_save_interval", interval)
        self.settings.setValue("save_path", path)
        self.settings.setValue("max_file_size", max_size)

        # 重启定时器
        if enabled:
            self.auto_save_timer.start(interval * 1000)
        else:
            self.auto_save_timer.stop()

        self.update_auto_save_status()

    def init_data(self):
        # 初始化参数
        self.sampling_rate = 1000  # NI-6009 采样率（根据实际配置调整）
        self.update_interval_ms = 200  # 显示刷新间隔（毫秒）
        self.samples_per_read = 200   # 每次读取的样本数
        
        # 初始化数据存储
        self.time_counter = 0
        self.history_data = np.zeros((4, 0))
        self.max_display_points = 1000  # 显示1秒的数据（1000点）

        # 配置NI-DAQmx任务
        self.acq_task = Task()
        self.acq_task.ai_channels.add_ai_voltage_chan("Dev1/ai0:3")
        self.acq_task.timing.cfg_samp_clk_timing(
            rate=self.sampling_rate,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=1000
        )
        self.acq_task.start()

        # 定时器设置
        self.timer = QTimer()
        self.timer.timeout.connect(self.read_real_data)
        self.timer.start(self.update_interval_ms)

    def auto_save_data(self):
        """自动保存数据"""
        if not self.waveform_display.is_paused:
            try:
                filename = self.generate_filename()
                data = self.waveform_display.get_waveform_data()
                if data is not None:
                    np.save(filename, data)
                    self.auto_save_status_label.setText(
                        f"最后自动保存：{datetime.datetime.now().strftime('%H:%M:%S')}"
                    )
            except Exception as e:
                print(f"自动保存失败：{str(e)}")
    
    def save_data(self):
        """保存波形数据"""
        data = self.waveform_display.get_waveform_data()
        if data is not None:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存数据", self.save_path,
                "CSV Files (*.csv);;TXT Files (*.txt);;NPY Files (*.npy)"
            )
            if file_path:
                try:
                    if file_path.endswith(".csv"):
                        np.savetxt(file_path, data, delimiter=',',
                                   header="Time,Channel1,Channel2,Channel3,Channel4",
                                   comments="")
                    elif file_path.endswith(".txt"):
                        np.savetxt(file_path, data, delimiter='\t',
                                   header="Time\tChannel1\tChannel2\tChannel3\tChannel4",
                                   comments="")
                    elif file_path.endswith(".npy"):
                        np.save(file_path, data)
                    self.recognition_display.setText(f"数据已保存: {os.path.basename(file_path)}")
                except Exception as e:
                    self.recognition_display.setText(f"保存失败: {str(e)}")


    # def generate_data(self):
    #     """生成模拟数据"""
    #     t = np.linspace(0, 1, 1000)
    #     data = np.array([
    #         np.sin(2 * np.pi * t),  # 通道1
    #         np.cos(2 * np.pi * t),   # 通道2
    #         np.sin(4 * np.pi * t),   # 通道3
    #         np.cos(4 * np.pi * t),   # 通道4
    #     ])
    #     self.waveform_display.update_waveforms(data)

    def read_real_data(self):
        try:
            # 读取数据
            new_data = self.acq_task.read(number_of_samples_per_channel=self.samples_per_read)
            new_data = np.array(new_data)  # 形状: (4, samples_per_read)
            
            # 计算时间轴
            time_per_update = self.samples_per_read / self.sampling_rate
            new_time = self.time_counter + np.arange(self.samples_per_read) / self.sampling_rate
            self.time_counter = new_time[-1]  # 更新全局时间戳
            
            # 累积数据并截取最近部分
            self.history_data = np.hstack((self.history_data, new_data))[:, -self.max_display_points:]
            time_axis = np.linspace(
                max(0, self.time_counter - self.max_display_points/self.sampling_rate),
                self.time_counter,
                self.history_data.shape[1]
            )
            
            # 更新显示
            self.waveform_display.update_waveforms(time_axis, self.history_data)
        except Exception as e:
            print(f"数据读取失败: {str(e)}")
            
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())