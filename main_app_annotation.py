import sys
import os
import subprocess
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QMainWindow, QDialog, QWidget, QPushButton,
                             QVBoxLayout, QLabel, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QPalette, QFont
import webbrowser
import tkinter as tk

root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()  # Close the hidden window

# hard_path = 'E:/home/mqxwd68/Downloads/Standard_Seg_Pipeline/EUS'
# hard_path = '/home/mqxwd68/Downloads/Standard_Seg_Pipeline/EUS'
class WaitingDialog(QDialog):
    def __init__(self, process, parent=None):
        super().__init__(parent)
        self.process = process
        self.init_ui()
        self.start_timers()

    def init_ui(self):
        self.setWindowTitle("Initializing")
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setFixedSize(400, 100)

        self.label = QLabel("Waiting for initialization")
        layout = QVBoxLayout()
        layout.addWidget(self.label, alignment=Qt.AlignCenter)
        self.setLayout(layout)

        # 动态点定时器
        self.dot_count = 0
        self.dot_timer = QTimer(self)
        self.dot_timer.timeout.connect(self.update_dots)

        # 自动关闭定时器
        self.close_timer = QTimer(self)
        self.close_timer.timeout.connect(self.close)

    def start_timers(self):
        self.dot_timer.start(500)  # 每500ms更新一次
        self.close_timer.start(10000)  # 3秒后自动关闭

    def update_dots(self):
        self.dot_count = (self.dot_count + 1) % 4
        dots = '.' * self.dot_count
        self.label.setText(f"Waiting for initialization{dots}\n(Press Q to terminate)")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.process.terminate()
            self.close()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 窗口设置
        self.setWindowTitle("EUS Test System")
        # self.setFixedSize(screen_width, screen_height)
        # self.setFixedSize(800, 494)
        self.resize(800, 494)  # 初始大小，允许最大化
        self.setMinimumSize(800, 494)  # 保持最小尺寸

        # 设置背景
        self.background = QLabel(self)
        self.set_background(
            # f"{hard_path}/Yolo-Series/GUI/bg.png")  # 替换你的背景图路径
            f"imgs/bg.png")  # 替换你的背景图路径

        # 主控件
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)

        # 布局设置
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(0, 130, 0, 70)  # 上下边距
        layout.setSpacing(20)

        # 创建按钮
        buttons = [
            ("REAL TIME TEST", self.run_realtime),
            ("LOCAL TEST", self.run_local),
            ("SMART ANNOTATION", self.smart_annotation),
            ("EXIT", self.close)
        ]

        # 按钮样式
        btn_style = """
        QPushButton {
            background-color: rgba(50, 50, 100, 150);
            color: white;
            border-radius: 15px;
            padding: 20px;
            font-size: 18px;
        }
        QPushButton:hover {
            background-color: rgba(70, 70, 150, 200);
        }
        """

        # 添加按钮
        for text, handler in buttons:
            btn = QPushButton(text)
            btn.clicked.connect(handler)
            btn.setStyleSheet(btn_style)
            btn.setFixedSize(QSize(300, 60))
            layout.addWidget(btn, alignment=Qt.AlignHCenter)

    def set_background(self, path):
        pixmap = QPixmap(path)
        self.background.setPixmap(pixmap.scaled(
            self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
        self.background.setGeometry(0, 0, self.width(), self.height())

    def run_realtime(self):
        process = subprocess.Popen([
            # "/home/mqxwd68/anaconda3/envs/sam2/bin/python",
            # f"{hard_path}/Yolo-Series/GUI/A001_Real_time_test.py"
            sys.executable,
            f"A001_Real_time_test.py"
        ])
        self.show_waiting_dialog(process, "Real time test")

    # def run_local(self):
    #     process = subprocess.Popen([
    #         "python3",
    #         f"{hard_path}/Yolo-Series/GUI/A001_Local_test.py"
    #     ])
    #     self.show_waiting_dialog(process, "Local test")
    def run_local(self):
        # 弹出文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )
        if file_path:
            # 传递视频路径作为参数
            process = subprocess.Popen([
                # "/home/mqxwd68/anaconda3/envs/sam2/bin/python",
                sys.executable,
                f"A002_Local_test_panel_rectangle_1.py",
                file_path
            ])
            self.show_waiting_dialog(process, "Local test")

    def smart_annotation(self):
        # Open folder dialog to select image folder
        folder_path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder_path:
            return

        # Verify the annotation script exists
        script_path = os.path.join(os.path.dirname(__file__), "A003_Smart_annotation_1.py")
        if not os.path.exists(script_path):
            QMessageBox.critical(self, "Error", "Annotation script not found!")
            return

        # Pass the folder path to A003_Smart_annotation.py and execute it
        try:
            process = subprocess.Popen([
                sys.executable,  # Use the same Python interpreter
                script_path,
                folder_path  # Pass folder path as argument
            ])
            self.show_waiting_dialog(process, "Loading...")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start annotation tool: {str(e)}")


    def show_waiting_dialog(self, process, test_type):
        dialog = WaitingDialog(process, self)
        dialog.setWindowTitle(f"Initializing {test_type}")
        dialog.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())