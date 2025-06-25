import json
import os, math
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFileDialog, QFrame, QScrollArea, QInputDialog, QMessageBox,
    QSizePolicy, QPushButton, QMenu, QSlider, QProgressDialog, QStatusBar, QLineEdit, QGroupBox, QProgressBar, QDialog, QTextEdit, QDesktopWidget, QComboBox
)
from PyQt5.QtCore import Qt, QPoint, QEvent, QTimer, QPropertyAnimation, QThread, pyqtSignal, QSize, QProcess
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QPen, QBrush, QCursor, QFont, QMovie
from tqdm import tqdm
from PIL import Image, ImageFilter, ImageDraw
import numpy as np

from scipy.ndimage import binary_erosion, binary_dilation, label
# import cv2

# 设置环境变量
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Select computation device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

# Initialize SAM2 predictor
try:
    from sam2.build_sam import build_sam2_video_predictor

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # sam2_checkpoint_t = os.path.join(script_dir, "SAM_model/checkpoint-20250612.pt")
    # model_cfg_t = "../sam2/configs/sam2.1/sam2.1_hiera_t.yaml"
    sam2_checkpoint_t = os.path.join(script_dir, "SAM_model/sam2.1_hiera_large.pt")
    model_cfg_t = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    ########################################################################################
    # From MedSAM2
    # sam2_checkpoint_t = os.path.join(script_dir, "MedSAM2_models/MedSAM2_US_Heart.pt")
    # model_cfg_t = "../sam2/configs/sam2.1/sam2.1_hiera_t.yaml"

    try:
        predictor = build_sam2_video_predictor(model_cfg_t, sam2_checkpoint_t, device=device)
        print("SAM2 predictor initialized successfully")
    except Exception as e:
        print(f"Error initializing SAM2 predictor: {str(e)}")
        predictor = None
except ImportError:
    print("Warning: SAM2 module not found. Running in simulation mode.")
    predictor = None

try:
    from scipy.ndimage import binary_erosion, binary_dilation, label
    has_scipy = True
except ImportError:
    has_scipy = False
    print("Warning: SciPy not installed. Some morphology operations may not work as expected.")

file_png2yolo = "A000_generate_yolo_from_png.py"
class_map_t = {
    'Common hepatic artery': '1',
    'Proper hepatic artery': '2',
    'Gastroduodenal artery': '3',
    'Left gastric artery': '4',
    'Right gastric artery': '5',
    'Left gastric vein': '6',
    'Right gastric vein': '7',
    'Pancreas': '8',
    'Duodenal stump': '9',
    'Liver': '10',
    'Gallbladder': '11', }  # 强制对应原图中的顺序调整

class_map_i = {
    'Curved grasper': '12',
    'Straight grasper': '13',
    'Irrigation tube': '14',
    'Harmonic scalpel': '15',
    'Stapler': '16',
    'Hem-o-lok': '17',
    'Gauze': '18',
    'Nndoscopic scissor': '19',
    'Needle holder': '20',
    'Needle': '21',
    'Suture': '22'
}

# 合并两个映射表
class_map_all = {**class_map_t, **class_map_i}
num_classes = len(class_map_all)


class InitThread(QThread):
    finished = pyqtSignal(object)  # 传递初始化状态
    error = pyqtSignal(str)  # 传递错误信息

    def __init__(self, frame_dir):
        super().__init__()
        self.frame_dir = frame_dir

    def run(self):
        """在后台线程中初始化SAM2"""
        try:
            # 初始化预测器
            inference_state = predictor.init_state(video_path=self.frame_dir)
            self.finished.emit(inference_state)
        except Exception as e:
            self.error.emit(str(e))


class LoadingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)  # 关键修改
        self.setWindowTitle("Loading SAM2")
        self.setWindowModality(Qt.ApplicationModal)
        self.setFixedSize(500, 200)  # 更大尺寸显示更多信息

        # 创建透明背景效果
        self.setAttribute(Qt.WA_TranslucentBackground)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)  # 添加边距

        # 半透明背景框架
        frame = QFrame()
        frame.setStyleSheet("background-color: rgba(255, 255, 255, 230); border-radius: 10px;")
        layout.addWidget(frame)
        frame_layout = QVBoxLayout(frame)

        # 添加标题
        title = QLabel("Frame Loading")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #333;")
        title.setAlignment(Qt.AlignCenter)
        frame_layout.addWidget(title)

        # 添加加载图标
        self.loading_label = QLabel()
        # movie = QMovie(":/icons/loading.gif")  # 如果有动画图标
        # if movie.isValid():
        #     movie.setScaledSize(QSize(32, 32))
        #     self.loading_label.setMovie(movie)
        #     print("we have movie")
        #     movie.start()
        # else:
        #     print("we no have movie")
        #     self.loading_label.setText("⏳")
        #     self.loading_label.setFont(QFont("Arial", 24))
        #     self.loading_label.setAlignment(Qt.AlignCenter)
        # frame_layout.addWidget(self.loading_label, 0, Qt.AlignCenter)

        # 进度标签
        self.label = QLabel("Initializing SAM2...")
        self.label.setStyleSheet("font-size: 11pt; color: #555;")
        self.label.setAlignment(Qt.AlignCenter)
        frame_layout.addWidget(self.label)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 不确定进度
        self.progress_bar.setFixedHeight(10)
        frame_layout.addWidget(self.progress_bar)

        # 日志区域
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(80)
        self.log_text.setStyleSheet(
            "font-family: 'Courier New'; font-size: 9pt; background-color: rgba(240, 240, 240, 180);")
        frame_layout.addWidget(self.log_text)

        # 添加取消按钮
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedWidth(100)
        cancel_btn.clicked.connect(self.reject)
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(cancel_btn)
        button_layout.addStretch(1)
        frame_layout.addLayout(button_layout)

        # 重定向sys.stdout
        sys.stdout = self

        # 确保显示在顶层
        self.raise_()
        self.activateWindow()

    def showEvent(self, event):
        """窗口显示时居中定位"""
        super().showEvent(event)
        self.center_dialog()

    def center_dialog(self):
        """将对话框居中显示"""
        frame_rect = self.frameGeometry()
        screen = QDesktopWidget().availableGeometry()
        frame_rect.moveCenter(screen.center())
        self.move(frame_rect.topLeft())

    def write(self, text):
        """捕获打印输出并显示在日志中"""
        if text.strip():  # 只添加非空行
            self.log_text.append(text.strip())
            # 滚动到底部
            cursor = self.log_text.textCursor()
            cursor.movePosition(cursor.End)
            self.log_text.setTextCursor(cursor)

    def flush(self):
        """标准输出需要实现flush方法"""
        pass

class ObjectButton(QWidget):
    def __init__(self, main_window, obj_id, name, color, active=False, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.obj_id = obj_id
        self.layout = QHBoxLayout()
        self.button = QPushButton(name)
        self.obj_color = color  # 保存对象颜色
        self.is_active = active  # 标识按钮是否被激活（是否有mask）

        # 设置按钮颜色 - 初始为灰色（未激活）
        self.update_button_style(active=active)

        self.button.setCheckable(True)
        self.layout.addWidget(self.button)
        self.setLayout(self.layout)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

        # 连接按钮点击事件
        self.button.clicked.connect(self.on_button_clicked)

        # 连接鼠标双击事件
        self.button.setMouseTracking(True)
        self.button.installEventFilter(self)

    def update_button_style(self, active=True):
        """根据活动状态更新按钮样式 - 增强视觉反馈"""
        # 使用对象颜色或灰色
        if active:
            color_hex = "#{:02x}{:02x}{:02x}".format(*self.obj_color)
            darker_color = "#{:02x}{:02x}{:02x}".format(
                max(0, self.obj_color[0] - 20),
                max(0, self.obj_color[1] - 20),
                max(0, self.obj_color[2] - 20)
            )
            hover_color = "#{:02x}{:02x}{:02x}".format(
                min(255, self.obj_color[0] + 20),
                min(255, self.obj_color[1] + 20),
                min(255, self.obj_color[2] + 20)
            )
        else:
            color_hex = "#f0f0f0"
            darker_color = "#d0d0d0"
            hover_color = "#ffffff"

        # 增强的视觉反馈样式
        style = f"""
                QPushButton {{
                    background-color: {color_hex};
                    color: {'white' if active else '#555'};
                    font-weight: bold;
                    border: 1px solid #666;
                    border-radius: 4px;
                    padding: 5px;
                    min-height: 30px;

                }}

                QPushButton:hover {{
                    background-color: {hover_color};
                    border: 2px solid #000;

                }}

                QPushButton:pressed {{
                    background-color: {darker_color};
                    border: 2px solid #000;

                }}

                QPushButton:checked {{
                    border: 3px solid {'#ff0000' if active else '#00ff00'};
                    background-color: {hover_color};

                }}
            """

        # 添加动画定义（仅在选中状态下有动画效果）
        if active:
            style += """
                     @keyframes pulse {  // 注释掉这一行
                        0% { box-shadow: 0 0 0 0px rgba(255, 255, 255, 0.7); }
                        70% { box-shadow: 0 0 0 8px rgba(255, 255, 255, 0); }
                        100% { box-shadow: 0 0 0 0px rgba(255, 255, 255, 0); }
                    } 
                """

        self.button.setStyleSheet(style)
        self.is_active = active

    def on_button_clicked(self):
        """处理按钮点击事件，添加视觉反馈"""
        # 创建点击动画效果
        # self.animate_click()

        # 选中逻辑
        if not self.button.isChecked():
            self.button.setChecked(True)
        self.main_window.select_object(self)

        # 保持刷子模式（如果已激活）
        if hasattr(self.main_window.image_label, 'brush_mode') and self.main_window.image_label.brush_mode:
            self.main_window.image_label.set_brush_mode(True, self.main_window.brush_size)

    def show_context_menu(self, pos):
        """显示右键菜单（删除对象）"""
        delete_menu = QMenu(self)
        delete_action = delete_menu.addAction("Delete Object")

        global_pos = self.mapToGlobal(pos)
        action = delete_menu.exec(global_pos)

        if action == delete_action:
            if self.main_window.delete_object(self.obj_id):
                self.deleteLater()

    def eventFilter(self, source, event):
        """事件过滤器处理鼠标双击"""
        if event.type() == QEvent.MouseButtonDblClick:
            if event.button() == Qt.LeftButton:
                # 触发闪烁效果
                self.main_window.highlight_mask(self.obj_id)

                # 选择此对象
                self.main_window.select_object(self)

                # 新增：重置连通分量为1
                self.main_window.connect_components = 1
                self.main_window.connect_value_label.setText(f"{self.main_window.connect_components}")
                print(f"Reset connected components to 1 for object {self.obj_id}")
                # 切换刷子模式
                # self.main_window.toggle_brush_mode()
                return True  # 事件已处理

        return super().eventFilter(source, event)


class ImageLabel(QLabel):
    def __init__(self, smart_annotation_tool, parent=None):
        super().__init__(parent)
        self.smart_annotation_tool = smart_annotation_tool
        self.setMouseTracking(True)
        # 添加历史记录相关属性
        self.history = []  # 存储历史状态
        self.current_index = 0  # 当前状态索引
        self.max_history = 15  # 最大历史记录数量

        # 存储点和标签，按对象ID分组
        self.object_points = {}  # {obj_id: [[x,y], ...]}
        self.object_labels = {}  # {obj_id: [label, ...]}
        self.object_boxes = {}  # {obj_id: [x1, y1, x2, y2]}

        # 边界框状态变量
        self.dragging_box = False
        self.box_start_point = None
        self.current_box = None  # 当前正在绘制的框
        self.current_obj_box = None  # 当前选中对象的框

        self.current_obj_id = None
        self.original_image = None
        self.display_image = None  # 用于显示的图像副本
        self.masks = {}  # {obj_id: mask_array}
        self.obj_mask_visible = self.smart_annotation_tool.obj_mask_visible  # 存储每个对象的mask可见性状态 {obj_id: bool}
        self.masks_visible = True  # 控制分割结果是否可见
        self.points_visible = True  # 控制标记点是否可见
        self.boxes_visible = True  # 控制边界框是否可见

        # 刷子工具相关变量
        self.brush_mode = False
        self.brush_size = 10
        self.brush_position = None
        self.is_brushing = False
        self.brush_positive = True
        self.brush_temp_mask = None  # 临时存储刷子修改
        self.original_mask = None  # 刷子操作前的原始掩码

        # 尺寸策略确保适应原始图像尺寸
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setAlignment(Qt.AlignCenter)
        self.setFocusPolicy(Qt.StrongFocus)  # 确保可以接收键盘事件

    def apply_gaussian_blur(self, kernel_size):
        """对当前对象的mask应用高斯模糊 - 使用PIL实现"""
        if self.current_obj_id not in self.masks:
            return False

        mask = self.masks[self.current_obj_id]
        mask_uint8 = (mask * 255).astype(np.uint8)

        # 将数组转为PIL图像
        mask_img = Image.fromarray(mask_uint8)

        # 应用高斯模糊
        # 注意：PIL的高斯模糊需要半径，我们使用核大小的1/2作为近似
        radius = max(1, kernel_size // 2)
        blurred_img = mask_img.filter(ImageFilter.GaussianBlur(radius=radius))

        # 转回数组并二值化
        blurred_array = np.array(blurred_img)
        binary_array = blurred_array > 128

        self.masks[self.current_obj_id] = binary_array
        self.update_display()
        self.smart_annotation_tool.mask_modified = True
        return True

    def apply_morph_operations(self, kernel_size):
        """对当前对象的mask应用形态学开闭操作 - 使用NumPy实现"""
        if self.current_obj_id not in self.masks:
            return False

        mask = self.masks[self.current_obj_id]

        # 创建结构元素
        if kernel_size > 1:
            structure = np.ones((kernel_size, kernel_size), dtype=bool)
        else:
            structure = np.ones((3, 3), dtype=bool)

        # 应用开操作（先腐蚀后膨胀）
        opened = binary_erosion(mask, structure=structure)
        opened = binary_dilation(opened, structure=structure)

        # 应用闭操作（先膨胀后腐蚀）
        closed = binary_dilation(opened, structure=structure)
        closed = binary_erosion(closed, structure=structure)

        self.masks[self.current_obj_id] = closed
        self.update_display()
        self.smart_annotation_tool.mask_modified = True
        return True

    def keep_top_connected_components(self, n_components):
        """保留面积最大的前n个连通分量 - 使用SciPy实现"""
        if self.current_obj_id not in self.masks:
            return False

        mask = self.masks[self.current_obj_id]

        # 标记连通分量
        labeled_mask, num_features = label(mask)

        # 如果没有连通分量，直接返回
        if num_features == 0:
            return False

        # 计算每个连通分量的面积
        component_sizes = [np.sum(labeled_mask == i) for i in range(1, num_features + 1)]

        # 按面积排序
        sorted_indices = np.argsort(component_sizes)[::-1]
        top_indices = sorted_indices[:min(n_components, num_features)]

        # 创建新掩码，只保留前n个分量
        new_mask = np.zeros_like(mask, dtype=bool)
        for i in top_indices:
            new_mask |= (labeled_mask == i + 1)

        self.masks[self.current_obj_id] = new_mask
        self.update_display()
        self.smart_annotation_tool.mask_modified = True
        return True

    def set_image(self, image_path):
        try:
            self.original_image = Image.open(image_path).convert("RGB")
            # 保持透明度通道用于绘制
            self.display_image = self.original_image.copy().convert("RGBA")
            # 添加初始空状态到历史记录
            # self.save_current_state()
            print(f"Image loaded: {image_path}")
        except Exception as e:
            print(f"Error loading image: {str(e)}")

    def set_brush_mode(self, active, brush_size=None):
        """设置刷子模式"""
        self.brush_mode = active
        if brush_size is not None:
            self.brush_size = brush_size

        if active:
            # 创建圆形刷子光标
            cursor_size = max(20, self.brush_size * 3)  # 确保光标大小足够
            pixmap = QPixmap(cursor_size, cursor_size)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setPen(QPen(Qt.white, 2))
            painter.setBrush(QBrush(Qt.NoBrush))
            radius = self.brush_size // 2
            center_x = cursor_size // 2
            center_y = cursor_size // 2

            # 绘制圆圈轮廓
            painter.drawEllipse(center_x - radius, center_y - radius, self.brush_size, self.brush_size)
            painter.end()

            # 设置自定义光标 - 修正参数类型为整数
            hot_x = int(center_x)  # 确保热点X坐标为整数
            hot_y = int(center_y)  # 确保热点Y坐标为整数
            self.setCursor(QCursor(pixmap, hot_x, hot_y))
        else:
            self.setCursor(Qt.ArrowCursor)

    def save_current_state(self):
        """将当前状态保存到历史记录"""
        if len(self.history) >= self.max_history:
            self.history.pop(0)  # 移除最旧的状态

        # 保存当前状态
        state = self.get_state()
        self.history.append(state)
        print(f"Saved state to history (total: {len(self.history)}/{self.max_history})")

    def undo_last_action(self):
        """回退到上一次状态"""
        if len(self.history) > 1:  # 至少要有2个状态才能回退
            # 移除当前状态
            self.history.pop()
            # 获取前一个状态
            previous_state = self.history[-1]
            # 恢复状态
            self.restore_state(previous_state)
            self.update_display()
            print(f"Undo completed. History states left: {len(self.history)}")
        else:
            print("Cannot undo - no previous state available")

    def reset_state(self):
        """重置当前frame的状态"""
        self.object_points = {}
        self.object_labels = {}
        self.object_boxes = {}
        self.masks = {}
        self.current_box = None
        self.current_obj_box = None
        self.update_display()

    def restore_state(self, state):
        """从状态字典恢复状态"""
        if state:
            self.object_points = state.get('object_points', {}).copy()
            self.object_labels = state.get('object_labels', {}).copy()
            self.object_boxes = state.get('object_boxes', {}).copy()
            self.masks = state.get('masks', {}).copy()
            self.current_box = state.get('current_box', None)
            self.current_obj_box = state.get('current_obj_box', None)
            self.current_obj_id = state.get('current_obj_id', None)
            self.update_display()

    def get_state(self):
        """获取当前frame的状态字典"""
        return {
            'object_points': self.object_points.copy(),
            'object_labels': self.object_labels.copy(),
            'object_boxes': self.object_boxes.copy(),
            'masks': self.masks.copy(),
            'current_box': self.current_box,
            'current_obj_box': self.current_obj_box,
            'current_obj_id': self.current_obj_id
        }

    # 修改update_display方法
    def update_display(self):
        if self.original_image is None:
            return

        # 创建新图像（不包含点和掩码）
        img = self.display_image.copy().convert("RGB")
        width, height = img.size

        # 创建带透明通道的合成图层
        composite = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(composite)

        # 处理掩码显示（考虑刷子模式和可见性）
        should_draw_masks = self.masks_visible
        # if self.current_obj_id is not None:
            # 有选中对象时，使用该对象的可见性状态
            # should_draw_masks = self.obj_mask_visible.get(self.current_obj_id, True)
            # print("should_draw_masks: ", should_draw_masks)

        # 如果需要在刷子模式中特殊处理当前对象的掩码
        if self.is_brushing and self.current_obj_id in self.masks:
            # 1. 先绘制其他对象的掩码（如果它们的mask可见）
            for obj_id, mask in self.masks.items():
                if obj_id == self.current_obj_id or mask is None or mask.size == 0:
                    continue

                # 检查该对象的可见性
                if self.current_obj_id is not None and not self.obj_mask_visible.get(obj_id, True):
                    continue

                # 获取颜色
                color = self.smart_annotation_tool.obj_colors.get(obj_id, (255, 0, 0, 128))

                # 绘制掩码
                try:
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    mask_img = Image.fromarray(mask_uint8, 'L')

                    # 创建彩色层
                    color_layer = Image.new('RGBA', img.size, color)
                    composite.paste(color_layer, (0, 0), mask_img)
                except Exception as e:
                    print(f"Error processing mask (obj_id={obj_id}): {str(e)}")

            # 2. 然后绘制当前对象的临时掩码（黄色高亮），编辑时总是可见
            if self.brush_temp_mask is not None:
                current_obj_id = self.current_obj_id
                highlight_color = (255, 255, 0, 180)  # 黄色高亮

                mask_uint8 = (self.brush_temp_mask * 255).astype(np.uint8)
                mask_img = Image.fromarray(mask_uint8, 'L')

                # 创建彩色层
                color_layer = Image.new('RGBA', img.size, highlight_color)
                composite.paste(color_layer, (0, 0), mask_img)
        elif should_draw_masks:  # 非刷子模式，只有全局或选中对象可见时才绘制
        # else:
            # 正常绘制所有掩码（如果它们可见）
            for obj_id, mask in self.masks.items():
                if mask is None or mask.size == 0:
                    print(f"Warning: Invalid mask (obj_id={obj_id}) - skipping")
                    continue

                # 如果没有选中对象或该对象可见
                if self.current_obj_id is None or self.obj_mask_visible.get(obj_id, True):
                    # print(f"No brushTool ID (obj_id={obj_id}) is visible")
                    # 获取颜色
                    color = self.smart_annotation_tool.obj_colors.get(obj_id, (255, 0, 0, 128))

                    try:
                        mask_uint8 = (mask * 255).astype(np.uint8)
                        mask_img = Image.fromarray(mask_uint8, 'L')

                        # 创建彩色层
                        color_layer = Image.new('RGBA', img.size, color)
                        composite.paste(color_layer, (0, 0), mask_img)
                    except Exception as e:
                        print(f"Error processing mask (obj_id={obj_id}): {str(e)}")

        # 如果边界框可见，绘制所有边界框
        if self.boxes_visible:
            for obj_id, box in self.object_boxes.items():
                if obj_id == self.current_obj_id:
                    # 当前选中的对象框用白色
                    draw.rectangle([box[0], box[1], box[2], box[3]], outline="white", width=2)

                # 获取颜色
                color = self.smart_annotation_tool.obj_colors.get(obj_id, (255, 0, 0))
                draw.rectangle([box[0], box[1], box[2], box[3]], outline=tuple(color[:3]), width=2)

        # 绘制当前正在拖拽的边界框
        if self.current_box is not None:
            draw.rectangle([self.current_box[0], self.current_box[1],
                            self.current_box[2], self.current_box[3]],
                           outline="cyan", width=2)

        # 如果点可见，添加所有对象的点
        if self.points_visible:
            for obj_id, points in self.object_points.items():
                labels = self.object_labels.get(obj_id, [])
                if not points or not labels:
                    continue

                # 获取点的颜色（对象颜色）
                color = self.smart_annotation_tool.obj_colors.get(obj_id, (0, 255, 0, 255))

                # 转换RGBA元组为PIL颜色字符串
                point_color = "#{:02x}{:02x}{:02x}".format(*color[:3])

                # 为每个点绘制标记
                for point, label in zip(points, labels):
                    x, y = point

                    # 根据标签类型绘制形状
                    if label == 1:  # 正样本点（绿色+号）
                        # 绘制+号
                        draw.line([(x - 6, y), (x + 6, y)], fill=point_color, width=3)
                        draw.line([(x, y - 6), (x, y + 6)], fill=point_color, width=3)
                    else:  # 负样本点（红色-号）
                        # 绘制-号
                        draw.line([(x - 6, y), (x + 6, y)], fill='red', width=3)

        # 合成最终图像
        final_img = Image.alpha_composite(img.convert('RGBA'), composite)

        # 转换为QPixmap显示
        final_img = final_img.convert("RGB")
        qimage = QImage(final_img.tobytes(), final_img.width, final_img.height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        # 设置图像并保持原始尺寸
        self.setPixmap(pixmap)
        self.setFixedSize(width, height)
        self.repaint()  # 强制立即重绘

    def draw_mask_to_composite(self, composite, mask_array, color):
        """将掩码绘制到合成图层"""
        if mask_array is None or mask_array.size == 0:
            return

        # 创建掩码图像
        mask_img = Image.fromarray((mask_array * 255).astype(np.uint8), 'L')

        # 创建彩色层
        r, g, b, a = color
        color_layer = Image.new('RGBA', composite.size, (r, g, b, a))

        # 添加掩码
        composite.alpha_composite(color_layer, (0, 0), mask_img)

    def mousePressEvent(self, event):
        if self.original_image is None or self.current_obj_id is None:
            return

        self.setFocus()

        if self.brush_mode and self.current_obj_id:
            # 刷子模式下开始绘制
            if event.button() == Qt.LeftButton:
                self.brush_positive = True
                self.start_brushing(event.pos())
            elif event.button() == Qt.RightButton:
                self.brush_positive = False
                self.start_brushing(event.pos())
            return

        else:  # not brush mode
            # 获取点击位置（相对于QLabel）
            x = event.pos().x()
            y = event.pos().y()

            # 检查是否在图像范围内
            if not (0 <= x < self.display_image.width and 0 <= y < self.display_image.height):
                return
            # 确定标签（左键=1，右键=0）
            label_val = 1
            # 右键点击：删除点/框
            if event.button() == Qt.RightButton:
                if self.current_obj_id is None:
                    return

                # 清除当前对象的框
                self.clear_box_for_current_object()
                label_val = 0
                # return

            # 左键按下：开始选择点或绘制边界框
            if self.current_obj_id is None:
                return

            # 如果按下Ctrl键，开始绘制边界框
            if QApplication.keyboardModifiers() == Qt.ControlModifier:
                self.dragging_box = True
                self.box_start_point = QPoint(x, y)
                self.current_box = [x, y, x, y]  # 初始化为点
            else:  # the promt of click
                # 添加点
                # 确定标签（左键=1，右键=0）
                self.add_point(x, y, label_val)
                self.run_prediction()
            # 确保刷子模式关闭
            if self.brush_mode:
                self.set_brush_mode(False)
        # self.save_current_state()  # 鼠标操作后保存状态

    def mouseMoveEvent(self, event):

        if self.brush_mode and self.is_brushing and self.current_obj_id:
            # 刷子模式下继续绘制
            self.continue_brushing(event.pos())
            return

        if not self.dragging_box or self.original_image is None or self.box_start_point is None:
            return

        # 更新当前框的结束位置
        current_point = event.pos()

        # 确保坐标在图像范围内
        width, height = self.display_image.size
        x1 = min(max(0, self.box_start_point.x()), width - 1)
        y1 = min(max(0, self.box_start_point.y()), height - 1)
        x2 = min(max(0, current_point.x()), width - 1)
        y2 = min(max(0, current_point.y()), height - 1)

        # 确保是有效的矩形（非零面积）
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            return

        # 转换为左上角+右下角格式
        self.current_box = [
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2)
        ]

        # 更新显示
        self.update_display()

    def mouseReleaseEvent(self, event):

        if self.brush_mode and self.is_brushing and self.current_obj_id:
            # 刷子模式下结束绘制
            self.finish_brushing()
            return

        if not self.dragging_box or self.original_image is None or self.box_start_point is None:
            return

        # 完成绘制边界框
        self.dragging_box = False

        # 确保我们有有效的框
        if self.current_box and self.current_obj_id:
            x1, y1, x2, y2 = self.current_box

            # 确保是有效的矩形（非零面积）
            if abs(x2 - x1) >= 5 and abs(y2 - y1) >= 5:
                # 设置当前对象的边界框
                self.object_boxes[self.current_obj_id] = self.current_box

                # 清除当前临时框
                self.current_box = None

                # 运行预测
                self.run_prediction()

        # self.save_current_state()  # 框选完成后保存状态

    def start_brushing(self, pos):
        """开始刷子操作"""
        self.is_brushing = True
        self.brush_position = (pos.x(), pos.y())

        # 保存当前掩码状态，用于回退
        obj_id = self.current_obj_id
        if obj_id in self.masks:
            self.original_mask = self.masks[obj_id].copy()
        else:
            # 如果没有掩码，创建一个空掩码
            w, h = self.original_image.size
            self.original_mask = np.zeros((h, w), dtype=bool)
            self.masks[obj_id] = self.original_mask

        self.brush_temp_mask = self.original_mask.copy()

        # 在当前位置绘制初始笔触
        self.draw_brush_stroke(pos.x(), pos.y())
        self.update_display()

    def continue_brushing(self, pos):
        """继续刷子操作"""
        last_x, last_y = self.brush_position
        new_x, new_y = pos.x(), pos.y()

        # 在移动点之间绘制线段
        self.draw_brush_line(last_x, last_y, new_x, new_y)
        self.brush_position = (new_x, new_y)
        self.update_display()

    def finish_brushing(self):
        """结束刷子操作"""
        self.is_brushing = False
        obj_id = self.current_obj_id
        self.masks[obj_id] = self.brush_temp_mask
        self.brush_temp_mask = None
        self.original_mask = None
        self.save_current_state()  # 保存状态到历史记录

    def draw_brush_stroke(self, x, y):
        """在指定位置绘制笔触"""
        if self.brush_temp_mask is None:
            return

        # 创建圆形笔触
        radius = self.brush_size // 2
        x1, y1 = x - radius, y - radius
        x2, y2 = x + radius, y + radius

        # 获取图像尺寸
        height, width = self.brush_temp_mask.shape

        # 确保笔触在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)

        # 创建圆形区域的掩码
        xx, yy = np.mgrid[y1:y2 + 1, x1:x2 + 1]  # 注意这里y在前x在后
        circle = (xx - y) ** 2 + (yy - x) ** 2 <= radius ** 2

        # 应用笔触
        if self.brush_positive:
            self.brush_temp_mask[y1:y2 + 1, x1:x2 + 1][circle] = True  # 添加像素
        else:
            self.brush_temp_mask[y1:y2 + 1, x1:x2 + 1][circle] = False  # 移除像素

    def draw_brush_line(self, x1, y1, x2, y2):
        """在两点之间绘制连续笔触"""
        distance = max(1, int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)))
        steps = max(2, distance)

        for i in range(steps):
            t = i / (steps - 1)
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            self.draw_brush_stroke(x, y)

    def paintEvent(self, event):
        """自定义绘制事件，添加刷子预览"""
        super().paintEvent(event)

        # 如果处于刷子模式且没有正在绘制，显示刷子预览
        if self.brush_mode and self.current_obj_id and not self.is_brushing:
            painter = QPainter(self)
            painter.setPen(QPen(QColor(255, 255, 0), 1, Qt.SolidLine))
            painter.setBrush(Qt.NoBrush)

            # 绘制刷子轮廓
            if self.brush_position:
                x, y = self.brush_position
                radius = self.brush_size // 2
                painter.drawEllipse(int(x - radius), int(y - radius),
                                    self.brush_size, self.brush_size)

    def add_point(self, x, y, label_val):
        # 为当前对象添加点
        if self.current_obj_id not in self.object_points:
            self.object_points[self.current_obj_id] = []
            self.object_labels[self.current_obj_id] = []

        self.object_points[self.current_obj_id].append([x, y])
        self.object_labels[self.current_obj_id].append(label_val)
        # self.save_current_state()  # 添加点后保存状态

    def clear_box_for_current_object(self):
        """清除当前对象的边界框"""
        if self.current_obj_id and self.current_obj_id in self.object_boxes:
            del self.object_boxes[self.current_obj_id]
            print(f"Current object {self.current_obj_id} has been removed")
            self.current_box = None
            self.run_prediction()
            self.update_display()

        self.save_current_state()  # 清除框后保存状态

    def run_prediction(self):
        """使用当前点和框运行SAM2预测"""
        if self.current_obj_id is None:
            return

        points = self.object_points.get(self.current_obj_id, [])
        labels = self.object_labels.get(self.current_obj_id, [])
        box = self.object_boxes.get(self.current_obj_id, None)

        # 调用SAM2预测（即时更新结果）
        self.smart_annotation_tool.run_sam_prediction(
            points,
            labels,
            self.current_obj_id,
            box=box  # 添加box参数
        )

        self.save_current_state()  # 运行预测后保存状态

    def keyPressEvent(self, event):
        # 按键'T'：切换点和掩码的可见性
        if event.key() == Qt.Key_T:
            if self.current_obj_id:
                # 切换当前对象的mask可见性
                self.obj_mask_visible[self.current_obj_id] = not self.obj_mask_visible.get(self.current_obj_id, True)
                print(f"Toggled mask visibility for object {self.current_obj_id} to "
                      f"{'visible' if self.obj_mask_visible[self.current_obj_id] else 'hidden'}")
                print(self.obj_mask_visible)
            else:
                # 切换所有掩码的全局可见性
                self.masks_visible = not self.masks_visible
                self.points_visible = not self.points_visible
                self.boxes_visible = not self.boxes_visible
                print(f"Points, masks and boxes are now {'visible' if self.masks_visible else 'hidden'}")
            self.update_display()
        # 按键'C'：清除当前对象的所有点和框
        elif event.key() == Qt.Key_R and self.current_obj_id:
            self.object_points[self.current_obj_id] = []
            self.object_labels[self.current_obj_id] = []
            self.obj_mask_visible[self.current_obj_id] = False
            self.clear_box_for_current_object()
            self.update_display()
            self.save_current_state()  # 保存状态到历史记录
            print(f"Cleared points and box for object {self.current_obj_id}")
        # 处理Ctrl+Z撤销操作
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_Z:
            self.undo_last_action()
            # 高斯平滑
        elif event.key() == Qt.Key_G and self.current_obj_id:
            if self.apply_gaussian_blur(self.smart_annotation_tool.gaussian_kernel_size):
                self.save_current_state()
                # self.smart_annotation_tool.highlight_operation("gaussian")
                print(f"Applied Gaussian blur (kernel={self.smart_annotation_tool.gaussian_kernel_size})")
        # 形态学操作
        elif event.key() == Qt.Key_F and self.current_obj_id:
            if self.apply_morph_operations(self.smart_annotation_tool.morph_kernel_size):
                self.save_current_state()
                # self.smart_annotation_tool.highlight_operation("morphology")
                print(f"Applied morphological operations (kernel={self.smart_annotation_tool.morph_kernel_size})")
        # 保留连通分量
        elif event.key() == Qt.Key_H and self.current_obj_id:
            if self.keep_top_connected_components(self.smart_annotation_tool.connect_components):
                self.save_current_state()
                # self.smart_annotation_tool.highlight_operation("components")
                print(f"Kept top {self.smart_annotation_tool.connect_components} connected components")
        # 保存操作
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_S:
            self.smart_annotation_tool.save_current_masks()

        # 新增：按E键清除选中目标的mask
        elif event.key() == Qt.Key_E and self.current_obj_id:
            self.clear_mask_for_current_object()
            self.save_current_state()  # 保存状态到历史记录
            self.update_display()
            print(f"Cleared mask for object {self.current_obj_id}")
        else:
            super().keyPressEvent(event)

    def clear_mask_for_current_object(self):
        """清除当前选中对象的所有点、框和传播结果"""
        obj_id = self.current_obj_id
        if obj_id is None:
            return

        # 清除当前对象的点和框
        if obj_id in self.object_points:
            self.object_points[obj_id] = []
        if obj_id in self.object_labels:
            self.object_labels[obj_id] = []
        if obj_id in self.object_boxes:
            self.object_boxes[obj_id] = None

        w, h = self.original_image.size
        # 清除当前帧的mask
        if obj_id in self.masks:
            self.masks[obj_id] = np.zeros(
                (h,w),
                dtype=np.uint8
            )

        # 清除传播结果 - 关键修改部分
        if hasattr(self, 'smart_annotation_tool') and self.smart_annotation_tool:
            smart_tool = self.smart_annotation_tool
            current_frame = smart_tool.current_frame_idx

            # 1. 从masks缓存中清除
            if current_frame in smart_tool.masks:
                if obj_id in smart_tool.masks[current_frame]:
                    del smart_tool.masks[current_frame][obj_id]

            # 2. 从generated_frames中清除
            if smart_tool.generated_frames:
                # 遍历所有缓存的帧
                for frame_idx, frame_data in smart_tool.generated_frames.items():
                    obj_ids = frame_data[0]
                    mask_logits = frame_data[1]

                    # 找到当前对象的位置
                    if obj_id in obj_ids:
                        idx = obj_ids.index(obj_id)

                        # 从对象ID列表中移除
                        new_obj_ids = list(obj_ids)
                        new_obj_ids.remove(obj_id)

                        # 从mask_logits中移除对应的掩码
                        new_mask_logits = list(mask_logits)
                        del new_mask_logits[idx]

                        # 更新帧数据
                        smart_tool.generated_frames[frame_idx] = {
                            'obj_ids': new_obj_ids,
                            'mask_logits': new_mask_logits
                        }

                        # 更新UI显示的masks缓存
                        if frame_idx in smart_tool.masks and obj_id in smart_tool.masks[frame_idx]:
                            del smart_tool.masks[frame_idx][obj_id]

                # 重新初始化传播生成器以确保一致性
                smart_tool.initialize_video_propagation()

                print(f"Removed object {obj_id} from propagation results")

    def add_mask(self, obj_id, mask_array):
        # 确保mask_array是有效的二维数组
        if mask_array is None or mask_array.size == 0:
            print(f"add_mask error: Invalid mask array (obj_id={obj_id})")
            return

        # 确保mask是二维数组
        if mask_array.ndim > 2:
            mask_array = mask_array.squeeze()

        if mask_array.ndim != 2:
            print(f"add_mask error: Invalid mask shape (obj_id={obj_id}, shape={mask_array.shape})")
            return

        # 更新对象掩码
        self.masks[obj_id] = mask_array

        # 强制更新显示
        self.update_display()

    def clear_object_points(self, obj_id):
        """清除特定对象的所有点"""
        if obj_id in self.object_points:
            del self.object_points[obj_id]

        if obj_id in self.object_labels:
            del self.object_labels[obj_id]

    def clear_current_points(self):
        """清除当前对象的所有点"""
        if self.current_obj_id:
            self.clear_object_points(self.current_obj_id)
            self.update_display()

    def remove_mask(self, obj_id):
        """移除特定对象的掩码"""
        if obj_id in self.masks:
            del self.masks[obj_id]
            self.update_display()

    def set_current_object(self, obj_id):
        """设置当前活动对象"""
        self.current_obj_id = obj_id
        print(f"Set current object to: {obj_id}")


class SmartAnnotationTool(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAM2 Interactive Annotation")
        self.setGeometry(100, 100, 1200, 800)

        self.frame_dir = None
        self.frame_names = []
        self.current_frame_idx = 0
        self.inference_state = None
        self.obj_counter = 1
        self.obj_buttons = {}
        self.obj_colors = {}
        self.obj_mask_visible = {}  # 存储每个对象的mask可见性状态 {obj_id: bool}
        # self.masks_visible = True  # 全局mask可见性状态（当没有选中对象时使用）

        # 添加状态字典来存储每个帧的状态
        self.frame_states = {}
        # 添加状态记录
        self.last_prompt_frame = None
        self.pending_prompts = {}  # 存储未导入的提示 {frame_idx: [obj1, obj2, ...]}

        # 添加必要的属性初始化
        self.masks = {}  # 存储所有帧的mask
        self.object_points_per_frame = {}  # 每帧的对象点
        self.object_boxes_per_frame = {}  # 每帧的对象框
        self.generated_frames = {}  # 存储生成的帧数据
        self.frame_generator = None  # 帧生成器
        self.propagation_started = False  # 传播是否已开始
        # 刷子工具相关变量
        self.brush_size = 20  # 默认刷子大小
        # 计算并设置统一宽度
        screen_width = QApplication.primaryScreen().size().width()
        unified_width = max(300, int(screen_width / 6))  # 使用1/4屏幕宽度，最小300px
        ################################################################################
        # navigation bar
        # 创建帧导航布局
        frame_nav_widget = QWidget()
        frame_nav_layout = QHBoxLayout(frame_nav_widget)
        frame_nav_widget.setFixedWidth(unified_width)
        # 帧导航标题
        frame_nav_layout.addWidget(QLabel("Frame Navigation:"))
        # 帧输入框
        self.frame_input = QLineEdit()
        self.frame_input.setPlaceholderText("Enter frame name or number")
        self.frame_input.returnPressed.connect(self.jump_to_frame)
        self.frame_input.setToolTip("Enter frame name without extension or frame number")
        frame_nav_layout.addWidget(self.frame_input)

        # 后缀标签
        self.frame_suffix_label = QLabel("")
        frame_nav_layout.addWidget(self.frame_suffix_label)

        # 跳转按钮
        jump_button = QPushButton("Go")
        jump_button.clicked.connect(self.jump_to_frame)
        frame_nav_layout.addWidget(jump_button)

        ################################################################################
        # post processign
        # 创建标签成员变量
        self.gauss_value_label = None
        self.morph_value_label = None
        self.connect_value_label = None
        # 添加新的控制面板布局
        self.gaussian_kernel_size = 17  # 默认高斯核大小
        self.morph_kernel_size = 17  # 默认形态学核大小
        self.connect_components = 1  # 默认连通分量数量
        ################################################################################
        # 创建新的操作面板
        operation_panel = QWidget()
        operation_layout = QHBoxLayout(operation_panel)
        ################################################################################
        # 高斯平滑控制
        gauss_group = QGroupBox("Gaussian")
        gauss_layout = QHBoxLayout(gauss_group)
        self.gauss_group = gauss_group

        gauss_label = QLabel("Kernel:")
        self.gauss_value_label = QLabel(f"{self.gaussian_kernel_size}")  # 保存为成员变量
        self.gauss_value_label.setFixedWidth(20)

        gauss_up_btn = QPushButton("↑")
        gauss_up_btn.setFixedWidth(25)
        gauss_up_btn.clicked.connect(lambda: self.adjust_kernel_size('gaussian', 2))

        gauss_down_btn = QPushButton("↓")
        gauss_down_btn.setFixedWidth(25)
        gauss_down_btn.clicked.connect(lambda: self.adjust_kernel_size('gaussian', -2))

        gauss_layout.addWidget(gauss_label)
        gauss_layout.addWidget(self.gauss_value_label)
        gauss_layout.addWidget(gauss_up_btn)
        gauss_layout.addWidget(gauss_down_btn)
        ###################################################################################
        # 形态学操作控制
        morph_group = QGroupBox("Morphology")
        morph_layout = QHBoxLayout(morph_group)
        self.morph_group = morph_group

        morph_label = QLabel("Kernel:")
        self.morph_value_label = QLabel(f"{self.morph_kernel_size}")  # 保存为成员变量
        self.morph_value_label.setFixedWidth(20)

        morph_up_btn = QPushButton("↑")
        morph_up_btn.setFixedWidth(25)
        morph_up_btn.clicked.connect(lambda: self.adjust_kernel_size('morph', 1))

        morph_down_btn = QPushButton("↓")
        morph_down_btn.setFixedWidth(25)
        morph_down_btn.clicked.connect(lambda: self.adjust_kernel_size('morph', -1))

        morph_layout.addWidget(morph_label)
        morph_layout.addWidget(self.morph_value_label)
        morph_layout.addWidget(morph_up_btn)
        morph_layout.addWidget(morph_down_btn)
        ###################################################################################
        # 连通分量控制
        connect_group = QGroupBox("Components")
        connect_layout = QHBoxLayout(connect_group)
        self.connect_group = connect_group

        connect_label = QLabel("Keep:")
        self.connect_value_label = QLabel(f"{self.connect_components}")  # 保存为成员变量
        self.connect_value_label.setFixedWidth(20)

        connect_up_btn = QPushButton("↑")
        connect_up_btn.setFixedWidth(25)
        connect_up_btn.clicked.connect(lambda: self.adjust_kernel_size('connect', 1))

        connect_down_btn = QPushButton("↓")
        connect_down_btn.setFixedWidth(25)
        connect_down_btn.clicked.connect(lambda: self.adjust_kernel_size('connect', -1))

        connect_layout.addWidget(connect_label)
        connect_layout.addWidget(self.connect_value_label)
        connect_layout.addWidget(connect_up_btn)
        connect_layout.addWidget(connect_down_btn)
        ###################################################################################
        # 保存按钮
        save_info = QLabel("S: Save Masks")
        save_info.setStyleSheet("font-weight: bold;")

        # 添加所有组件
        operation_layout.addWidget(gauss_group)
        operation_layout.addWidget(morph_group)
        operation_layout.addWidget(connect_group)
        operation_layout.addStretch(1)
        operation_layout.addWidget(save_info)

        # 添加保存状态变量
        self.save_yolo_path = None
        self.save_png_path = None
        self.mask_modified = False

        # UI设置
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 主内容区域布局
        content_layout = QHBoxLayout()
        content_layout.setSpacing(10)

        # 左侧面板
        left_panel = QFrame()
        left_panel.setFixedWidth(220)
        left_panel.setStyleSheet("background-color: #f8f8f8; border-radius: 5px;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(10)
        # 确保主窗口能接收键盘事件
        self.setFocusPolicy(Qt.StrongFocus)
        # 安装事件过滤器
        self.installEventFilter(self)
        # 添加对象按钮
        self.add_obj_btn = QPushButton("+ Add Object")
        self.add_obj_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.add_obj_btn.clicked.connect(self.add_new_object)
        left_layout.addWidget(self.add_obj_btn)

        # 添加导入提示按钮
        self.import_prompts_btn = QPushButton("Import Prompts")
        self.import_prompts_btn.setStyleSheet("""
            QPushButton {
                background-color: #FFA500;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #cc8400;
            }
        """)
        self.import_prompts_btn.clicked.connect(self.import_prompts)
        left_layout.addWidget(self.import_prompts_btn)

        # 对象滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("background-color: white; border: none;")
        self.obj_container = QWidget()
        self.obj_layout = QVBoxLayout(self.obj_container)
        self.obj_layout.setAlignment(Qt.AlignTop)
        self.obj_layout.setSpacing(5)
        scroll_area.setWidget(self.obj_container)
        left_layout.addWidget(scroll_area)

        content_layout.addWidget(left_panel)

        # 右侧滚动区域
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setStyleSheet("background-color: #2c2c2c;")
        self.image_label = ImageLabel(self)
        self.image_label.setStyleSheet("background-color: #3a3a3a;")
        right_scroll.setWidget(self.image_label)
        content_layout.addWidget(right_scroll, 1)  # 右边区域占主要空间

        main_layout.addLayout(content_layout)

        # 底部布局 - 帧导航和滚动条
        bottom_layout = QHBoxLayout()

        # 导航按钮
        self.prev_btn = QPushButton("⬅ Previous Frame")
        self.next_btn = QPushButton("Next Frame ➡")

        # 设置导航按钮样式
        nav_style = """
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:disabled {
                background-color: #b0bec5;
            }
        """
        self.prev_btn.setStyleSheet(nav_style)
        self.next_btn.setStyleSheet(nav_style)

        self.prev_btn.clicked.connect(self.prev_frame)
        self.next_btn.clicked.connect(self.next_frame)

        bottom_layout.addWidget(self.prev_btn)
        bottom_layout.addWidget(self.next_btn)

        # 滚动条
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #cccccc;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                width: 16px;
                height: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #2196F3;
                border-radius: 4px;
            }
        """)
        self.frame_slider.valueChanged.connect(self.slider_value_changed)
        bottom_layout.addWidget(self.frame_slider, 1)

        main_layout.addLayout(bottom_layout)
        # 在底部布局中创建帧数标签
        self.frame_position_label = QLabel()
        self.frame_position_label.setStyleSheet("""
                    QLabel {
                        color: #333;
                        font-weight: bold;
                        padding: 0 10px;
                    }
                """)
        bottom_layout.addWidget(self.frame_position_label)

        # UI中添加提示帧标签
        self.status_bar = QStatusBar()
        main_layout.addWidget(self.status_bar)
        self.update_prompt_status()

        # 创建刷子控制布局
        brush_control_widget = QWidget()
        brush_layout = QHBoxLayout(brush_control_widget)

        # 设置刷子控制部件的固定宽度（屏幕宽度的1/6）
        screen_width = QApplication.primaryScreen().size().width()
        brush_control_width = max(200, int(screen_width / 6))
        brush_control_widget.setFixedWidth(brush_control_width)
        # 在底部添加刷子大小控制
        # brush_layout = QHBoxLayout()
        brush_layout.addWidget(QLabel("Brush Size:"))

        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setMinimum(5)
        self.brush_slider.setMaximum(100)
        self.brush_slider.setValue(self.brush_size)
        self.brush_slider.valueChanged.connect(self.update_brush_size)
        brush_layout.addWidget(self.brush_slider)

        self.brush_size_label = QLabel(f"{self.brush_size}px")
        brush_layout.addWidget(self.brush_size_label)

        main_layout.addLayout(brush_layout)
        main_layout.addWidget(brush_control_widget)
        # 将帧导航添加到主布局
        main_layout.addWidget(frame_nav_widget)
        # 新的操作（高斯平滑，形态学开闭，连通分量）添加到主布局
        main_layout.addWidget(operation_panel)
        # 设置初始焦点
        self.image_label.setFocus()

    def highlight_operation(self, operation_name):
        """高亮显示操作面板"""
        # 创建半透明覆盖层
        overlay = QLabel(self)
        overlay.setStyleSheet("background-color: rgba(0, 255, 0, 150);")

        # 设置位置和大小
        if operation_name == "gaussian":
            rect = self.gauss_group.geometry()
        elif operation_name == "morphology":
            rect = self.morph_group.geometry()
        else:  # components
            rect = self.connect_group.geometry()

        # 扩大高亮区域
        margin = 10
        overlay.setGeometry(
            rect.x() - margin,
            rect.y() - margin,
            rect.width() + margin * 2,
            rect.height() + margin * 2
        )

        # 显示高亮
        overlay.show()

        # 创建淡出动画
        animation = QPropertyAnimation(overlay, b"windowOpacity")
        animation.setDuration(1000)  # 1秒
        animation.setStartValue(0.7)
        animation.setEndValue(0.0)
        animation.finished.connect(overlay.deleteLater)
        animation.start()

    def save_current_masks(self):
        """保存当前帧的mask为YOLO格式和PNG格式，只保存当前可见的mask"""
        # 检查是否有可见的mask（考虑全局可见性和对象自身可见性）
        has_visible_masks = False
        visible_mask_info = []  # 存储可见mask的信息（obj_id, mask, base_class_id）

        # 获取全局可见性
        masks_globally_visible = self.image_label.masks_visible
        obj_mask_visible = self.image_label.obj_mask_visible

        # 遍历所有对象，检查哪些mask是可见的
        for obj_id, mask in self.image_label.masks.items():
            # 全局可见性检查和对象自身可见性检查
            if masks_globally_visible and obj_mask_visible.get(obj_id, True):
                has_visible_masks = True

                # 获取基础类别ID
                base_class_id = obj_id // 1000 if obj_id > 999 else obj_id
                visible_mask_info.append((obj_id, mask, base_class_id))

        # 如果没有可见的mask，显示警告
        if not has_visible_masks:
            QMessageBox.warning(
                self, "No Visible Masks",
                "No visible masks to save! Ensure masks are globally visible and object visibility is enabled."
            )
            return False

        # 如果首次保存，获取保存路径
        if not self.save_yolo_path or not self.save_png_path:
            self.save_yolo_path = QFileDialog.getExistingDirectory(
                self, "Select Directory for YOLO Files"
            )
            if not self.save_yolo_path:
                return False

            self.save_png_path = QFileDialog.getExistingDirectory(
                self, "Select Directory for PNG Files"
            )
            if not self.save_png_path:
                return False

        # 创建合成mask（每个像素值代表其类别）
        w, h = self.image_label.original_image.size
        class_mask = np.zeros((h, w), dtype=np.uint8)

        # 遍历所有可见对象
        for obj_id, mask, base_class_id in visible_mask_info:
            class_mask[mask] = base_class_id
            print(f"Saving visible mask for object {obj_id} with class ID {base_class_id}")

        # 保存PNG文件（调色板模式）
        frame_name = os.path.basename(self.frame_names[self.current_frame_idx])
        frame_prefix = os.path.splitext(frame_name)[0]
        png_path = os.path.join(self.save_png_path, f"{frame_prefix}.png")

        # 创建调色板图像
        png_img = Image.fromarray(class_mask, mode='P')

        # 添加调色板（最多255个类别）
        palette = []
        for i in range(256):
            if i == 0:
                palette.extend([0, 0, 0])  # 背景为黑色
            else:
                # 为每个类别生成可区分的颜色
                hue = (i * 77) % 360
                # 使用饱和度85-100%，亮度55-65% 确保鲜艳明亮但不过曝
                s = 85 + (i % 15)  # 85% 到 100% 饱和度变化
                l = 55 + (i % 10)  # 55% 到 65% 亮度变化
                r, g, b = self.hsl_to_rgb(hue, s, l)
                palette.extend([r, g, b])

        png_img.putpalette(palette)
        png_img.save(png_path)
        print(f"Saved PNG mask: {png_path} (containing {len(visible_mask_info)} visible masks)")

        # 为YOLO格式准备文件路径
        txt_path = os.path.join(self.save_yolo_path, f"{frame_prefix}.txt")

        # 创建进程调用外部脚本
        process = QProcess(self)
        script_path = os.path.join(os.path.dirname(__file__), file_png2yolo)

        # 设置进程完成处理
        def on_process_finished(exit_code, exit_status):
            if exit_code == 0 and exit_status == QProcess.NormalExit:
                # print(f"Successfully generated YOLO annotations at {txt_path}")

                # 标记为已保存
                self.mask_modified = False

                # 频闪效果（闪白）
                self.flash_indicator()
            else:
                error = process.readAllStandardError().data().decode()
                QMessageBox.critical(
                    self, "YOLO Annotation Error",
                    f"Failed to generate YOLO annotations:\n{error}"
                )

                # 清理错误文件
                if os.path.exists(txt_path):
                    os.remove(txt_path)

        process.finished.connect(on_process_finished)

        # 添加处理标准输出和错误
        def on_ready_read_stdout():
            output = process.readAllStandardOutput().data().decode()
            if output:
                print("[YOLO Generation] " + output.strip())

        def on_ready_read_stderr():
            error = process.readAllStandardError().data().decode()
            if error:
                print("[YOLO Error] " + error.strip())

        process.readyReadStandardOutput.connect(on_ready_read_stdout)
        process.readyReadStandardError.connect(on_ready_read_stderr)

        # 启动进程
        process.start(sys.executable, [script_path, png_path, txt_path])

        # 等待进程启动
        if not process.waitForStarted():
            QMessageBox.critical(
                self, "Process Error",
                "Failed to start the YOLO generation process."
            )
            return False

        return True  # 返回成功，但实际异步进行

    def flash_indicator(self):
        """频闪效果指示保存成功"""
        # 创建白色覆盖层
        overlay = QLabel(self)
        overlay.setStyleSheet("background-color: white;")
        overlay.setGeometry(self.rect())
        overlay.show()

        # 创建淡出动画
        self.flash_animation = QPropertyAnimation(overlay, b"windowOpacity")
        self.flash_animation.setDuration(100)
        self.flash_animation.setStartValue(0.7)
        self.flash_animation.setEndValue(0.0)
        self.flash_animation.finished.connect(overlay.deleteLater)
        self.flash_animation.start()

    def check_for_unsaved_changes(self):
        """检查是否有未保存的修改，返回用户选择"""
        if not self.mask_modified:
            return QMessageBox.Yes  # 如果没有修改，直接继续

        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setText("You have unsaved mask modifications")
        msg_box.setInformativeText("Do you want to save your changes before leaving this frame?")
        msg_box.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        msg_box.setDefaultButton(QMessageBox.Save)

        ret = msg_box.exec_()
        if ret == QMessageBox.Save:
            return self.save_current_masks() and QMessageBox.Yes
        elif ret == QMessageBox.Discard:
            self.mask_modified = False
            return QMessageBox.Yes
        else:
            return QMessageBox.Cancel

    def adjust_kernel_size(self, kernel_type, delta):
        """调整不同操作的核大小"""
        if kernel_type == 'gaussian':
            new_size = max(1, min(21, self.gaussian_kernel_size + delta))
            if new_size % 2 == 0:  # 确保为奇数
                new_size += 1
            self.gaussian_kernel_size = new_size
            self.gauss_value_label.setText(f"{self.gaussian_kernel_size}")  # 直接更新成员变量

        elif kernel_type == 'morph':
            new_size = max(1, min(15, self.morph_kernel_size + delta))
            if new_size % 2 == 0:  # 确保为奇数
                new_size += 1
            self.morph_kernel_size = new_size
            self.morph_value_label.setText(f"{self.morph_kernel_size}")  # 直接更新成员变量

        elif kernel_type == 'connect':
            new_size = max(1, min(5, self.connect_components + delta))
            self.connect_components = new_size
            self.connect_value_label.setText(f"{self.connect_components}")  # 直接更新成员变量

    def update_frame_position_label(self):
        """更新帧位置标签"""
        if not self.frame_names or self.current_frame_idx < 0:
            return

        # 获取文件名和后缀
        frame_name = os.path.basename(self.frame_names[self.current_frame_idx])
        name_part, ext_part = os.path.splitext(frame_name)

        # 更新标签
        self.frame_position_label.setText(f"Frame {self.current_frame_idx + 1}/{len(self.frame_names)}: {name_part}")

        # 更新后缀标签
        self.frame_suffix_label.setText(ext_part)

    def jump_to_frame(self):
        # 检查未保存修改
        result = self.check_for_unsaved_changes()
        if result != QMessageBox.Yes:
            if result == QMessageBox.Cancel:
                return

        """跳转到指定帧"""
        input_text = self.frame_input.text().strip()
        if not input_text:
            return

        # 尝试转换为数字（帧索引）
        try:
            frame_num = int(input_text) - 1  # 用户输入1-based索引
            if 0 <= frame_num < len(self.frame_names):
                # 保存当前状态并跳转
                self.save_current_frame_state()
                self.current_frame_idx = frame_num
                self.load_frame()
                return
        except ValueError:
            pass

        # 尝试匹配文件名（不带后缀）
        input_lower = input_text.lower()
        for idx, frame_path in enumerate(self.frame_names):
            frame_name = os.path.basename(frame_path)
            name_part, ext_part = os.path.splitext(frame_name)

            # 检查是否匹配（不区分大小写）
            if name_part.lower() == input_lower:
                # 保存当前状态并跳转
                self.save_current_frame_state()
                self.current_frame_idx = idx
                self.load_frame()
                return

        # 如果都没找到，显示错误消息
        QMessageBox.warning(self, "Frame Not Found",
                            f"Frame '{input_text}' not found. Please enter a valid frame name or number.")

    def update_brush_size(self, size):
        """更新刷子大小"""
        self.brush_size = size
        self.brush_size_label.setText(f"{size}px")

        # 更新图像标签中的刷子大小
        if hasattr(self.image_label, 'brush_size'):
            self.image_label.brush_size = size

            # 如果已经处于刷子模式，更新光标
            if hasattr(self.image_label, 'brush_mode') and self.image_label.brush_mode:
                self.image_label.set_brush_mode(True, self.brush_size)

    def eventFilter(self, obj, event):
        """事件过滤器捕获键盘事件"""
        if event.type() == QEvent.KeyPress:
            # 空格键切换刷子模式
            if event.key() == Qt.Key_Space:
                self.toggle_brush_mode()
                print("Space toggle brush")
                return True

            # ESC键强制退出刷子模式
            elif event.key() == Qt.Key_Escape:
                if self.image_label.brush_mode:
                    self.toggle_brush_mode()
                    print("Esc toggle brush")
                    return True

                # 'c'键增加刷子大小
            elif event.key() == Qt.Key_C:
                new_size = min(100, self.brush_size + 2)
                self.brush_slider.setValue(new_size)
                print(f"Increased brush size to {new_size}")
                return True

                # 'x'键减小刷子大小
            elif event.key() == Qt.Key_X:
                new_size = max(5, self.brush_size - 2)
                self.brush_slider.setValue(new_size)
                print(f"Decreased brush size to {new_size}")
                return True

        return super().eventFilter(obj, event)

    def toggle_brush_mode(self):
        """切换刷子模式"""
        if self.image_label.current_obj_id is None:
            return

        new_mode = not self.image_label.brush_mode
        self.image_label.set_brush_mode(new_mode, self.brush_size)

        # if new_mode:
        # self.highlight_mask(self.image_label.current_obj_id)

        if not new_mode:
            # 确保光标恢复正常
            self.image_label.setCursor(Qt.ArrowCursor)

        self.update_prompt_status()

    def update_prompt_status(self):
        """更新状态栏提示帧信息"""
        current_frame = self.frame_names[self.current_frame_idx] if self.frame_names else "None"
        prompt_frame = self.frame_names[
            self.last_prompt_frame] if self.last_prompt_frame is not None and self.frame_names else "None"
        # self.status_bar.showMessage(f"Current frame: {current_frame} | Prompt frame: {prompt_frame}")
        # 添加刷子状态
        brush_status = ""
        if hasattr(self.image_label, 'brush_mode'):
            if self.image_label.brush_mode:
                brush_status = " | Brush Mode: ACTIVE"
            else:
                brush_status = " | Brush Mode: INACTIVE"

        message = f"Current frame: {os.path.basename(current_frame)} | " + \
                  f"Prompt frame: {os.path.basename(prompt_frame) if prompt_frame else 'None'}" + \
                  brush_status

        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage(message)

    def slider_value_changed(self, value):
        """当滚动条值改变时切换帧"""
        if not self.frame_names:
            return

        # 检查未保存修改
        result = self.check_for_unsaved_changes()
        if result != QMessageBox.Yes:
            if result == QMessageBox.Cancel:
                return

        # 保存当前帧状态
        self.save_current_frame_state()

        if value >= 0 and value < len(self.frame_names) and value != self.current_frame_idx:
            self.current_frame_idx = value
            self.load_frame()

    def add_new_object(self):
        """添加新对象，允许输入自定义类别或选择预定义类别"""
        # 初始化对象可见性

        # 创建自定义对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("Add New Object")

        # 创建布局
        layout = QVBoxLayout(dialog)

        # 创建输入框和标签
        input_label = QLabel("Object Class Name:")
        class_input = QLineEdit()
        layout.addWidget(input_label)
        layout.addWidget(class_input)

        # 创建下拉框和标签
        dropdown_label = QLabel("Or select from predefined classes:")
        class_dropdown = QComboBox()
        predefined_classes = list(class_map_all.keys())
        class_dropdown.addItems(predefined_classes)
        class_dropdown.setEditable(True)  # 可编辑的下拉框
        layout.addWidget(dropdown_label)
        layout.addWidget(class_dropdown)

        # 当下拉框选择改变时，更新输入框内容
        class_dropdown.currentIndexChanged.connect(
            lambda: class_input.setText(class_dropdown.currentText())
        )

        # 添加按钮布局
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        # 按钮连接
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)

        # 显示对话框并等待用户响应
        if dialog.exec_() != QDialog.Accepted:
            return  # 用户取消操作

        class_name = class_input.text().strip()

        # 验证输入
        if not class_name:
            QMessageBox.warning(self, "Invalid Input", "Class name cannot be empty.")
            return

        # 确定类别ID
        if class_name in class_map_all:
            # 如果输入的是预定义类名，使用其ID
            class_id = int(class_map_all[class_name])
        else:
            # 对于自定义类别：
            # 1. 首先检查是否存在已映射的ID
            if class_name in class_map_i:
                class_id = int(class_map_i[class_name])
            else:
                # 2. 获取最大的现有ID
                existing_ids = {int(id_val) for id_val in class_map_all.values()} | {int(id_val) for id_val in
                                                                                     class_map_i.values()}
                max_id = max(existing_ids) if existing_ids else 0
                new_base_id = max_id + 1

                # 3. 创建新的映射
                class_map_i[class_name] = new_base_id
                class_id = new_base_id

        # 检查是否有相同类别的对象存在
        existing_class_objs = [
            obj_id for obj_id, btn in self.obj_buttons.items()
            if (obj_id // 1000) == class_id
        ]

        # 计算序号后缀
        suffix = len(existing_class_objs)

        # 创建唯一的obj_id
        obj_id = class_id * 1000 + suffix

        # 确保obj_id唯一（处理边缘情况）
        while obj_id in self.obj_buttons:
            suffix += 1
            obj_id = class_id * 1000 + suffix

        self.obj_mask_visible[obj_id] = True

        # 如果存在相同类别的对象，在名称后添加后缀
        display_name = class_name
        if suffix > 0:
            display_name = f"{class_name}_{suffix}"

        # 为对象生成唯一颜色
        hue = (obj_id * 137.5) % 360  # 使用黄金角度分布
        r, g, b = self.hsl_to_rgb(hue, 65, 60)  # 增加饱和度和亮度
        obj_color = (r, g, b, 180)

        # 创建并添加对象按钮
        btn = ObjectButton(self, obj_id, display_name, obj_color[:3], active=False)
        self.obj_buttons[obj_id] = btn
        self.obj_layout.addWidget(btn)

        # 存储对象颜色
        self.obj_colors[obj_id] = obj_color

        print(f"Created object {obj_id} ({display_name}) with base ID {class_id} and color R:{r}, G:{g}, B:{b}")

        # # 初始化传播生成器以包含新对象
        # if hasattr(self, 'inference_state') and self.inference_state is not None:
        #     self.initialize_propagation_generator()

        print("Added new object for the incoming frames")

        # 选择新对象
        self.select_object(btn)

    def highlight_mask(self, obj_id):
        """高亮显示指定对象的掩码（闪烁效果）"""
        print(f"Highlighting mask for object {obj_id}")

        if obj_id not in self.obj_colors:
            print(f"Warning: No color defined for obj_id {obj_id}")
            return

        # 原始颜色
        original_color = self.obj_colors[obj_id]

        # 创建高亮颜色（黄色用于强调）
        highlight_color = (255, 255, 0, 255)  # 纯黄色

        # 临时设置高亮颜色
        self.obj_colors[obj_id] = highlight_color
        self.image_label.update_display()

        # 0.5秒后恢复原始颜色
        QTimer.singleShot(100, lambda: self.restore_mask_color(obj_id, original_color))

    def restore_mask_color(self, obj_id, original_color):
        """恢复原始mask颜色"""
        if obj_id in self.obj_colors:
            self.obj_colors[obj_id] = original_color
            self.image_label.update_display()
        print(f"Restored color for obj_id {obj_id}")

    def import_prompts(self, auto_process_current=True):
        """导入所有提示数据（Bbox）并更新所有相关帧的SAM2状态"""
        if not self.frame_names:
            QMessageBox.warning(self, "Error", "No frames loaded. Please load image folder first.")
            return

        # 选择提示文件夹
        prompts_dir = QFileDialog.getExistingDirectory(self, "Select Prompts Folder")
        if not prompts_dir:
            return

        print(f"Selected prompts directory: {prompts_dir}")

        # 确保核心数据结构存在
        self.ensure_data_structures_exist()

        # 获取所有JSON/TXT文件
        prompt_files = [f for f in os.listdir(prompts_dir) if f.endswith(('.json', '.txt'))]
        if not prompt_files:
            QMessageBox.warning(self, "Error", "No JSON or TXT files found in the selected folder.")
            return

        print(f"Found {len(prompt_files)} prompt files")

        # 获取不带后缀的帧名称
        frame_basenames = [os.path.splitext(f)[0] for f in self.frame_names]

        # 用于追踪已创建的对象和需要预测的帧
        frames_to_process = {}

        # 第0步：收集所有提示
        all_prompts = {}  # {frame_idx: [obj1, obj2, ...]}
        # 第一步：处理所有提示文件
        for prompt_file in prompt_files:
            try:
                # 获取不带后缀的文件名
                base_name = os.path.splitext(prompt_file)[0]

                # 查找对应帧的索引
                frame_idx = next(
                    (i for i, frame_name in enumerate(frame_basenames)
                     if frame_name in base_name),
                    None
                )

                if frame_idx is None:
                    print(f"Warning: Couldn't find frame for prompt {prompt_file}. Skipping.")
                    continue

                print(f"Processing prompt file: {prompt_file} for frame {frame_idx}")

                # 读取JSON文件
                with open(os.path.join(prompts_dir, prompt_file), 'r') as f:
                    prompt_data = json.load(f)

                # 确保文件包含所需的字段
                if 'objects' not in prompt_data or prompt_data['objects'] is []:
                    print(f"Warning: File {prompt_file} has invalid format. Missing 'objects' field.")
                    continue

                # to verifiy whether this prompt contains rectangle
                has_rectangle = False
                for obj in prompt_data['objects']:
                    if obj.get('geometryType') != 'rectangle':
                        continue # still no rec
                    else:
                        has_rectangle = True
                        break

                if not has_rectangle:
                    continue

                # 确保该帧的数据结构存在
                self.ensure_frame_structures_exist(frame_idx)

                # 初始化帧处理字典
                if frame_idx not in frames_to_process:
                    # frames_to_process[frame_idx] = []
                    all_prompts[frame_idx] = []

                # 添加到帧的提示列表
                all_prompts[frame_idx].extend(prompt_data['objects'])

            except Exception as e:
                print(f"Error processing prompt file {prompt_file}: {str(e)}")
                import traceback
                traceback.print_exc()

        print(f"current frame {self.current_frame_idx} / {all_prompts.keys()}")
        print(auto_process_current)
        # 第二步：只处理当前帧的提示（如果存在）
        if self.current_frame_idx in all_prompts.keys():  # and auto_process_current:
            print(f"current frame {self.current_frame_idx} is in {all_prompts}")
            self.process_prompts_for_frame(self.current_frame_idx, all_prompts[self.current_frame_idx])
            # 标记为已处理
            del all_prompts[self.current_frame_idx]

        # 第三步：存储其他帧的未处理提示
        self.pending_prompts.update(all_prompts)

        # 第四步：设置最近使用的提示帧（即使当前帧没有提示）
        self.last_prompt_frame = self.current_frame_idx

        # 更新状态栏
        self.update_prompt_status()

        # 如果其他帧有提示，只是存储不处理
        if all_prompts:
            print(f"Stored {sum(len(objs) for objs in all_prompts.values())} prompts for future frames")
        print("Prompt import completed")

    def update_predictor_state(self, frame_idx, obj_id, box):
        """更新预测器状态但不显示结果"""
        try:

            # 转换box格式
            box_np = np.array(box, dtype=np.float32).reshape(1, 4) if box else None

            # 通过SAM2处理
            predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=None,
                labels=None,
                box=box_np
            )
            print(f"Updated predictor state for obj_id {obj_id} at frame {frame_idx}")

        except Exception as e:
            print(f"Error updating predictor state: {str(e)}")
            import traceback
            traceback.print_exc()

    def initialize_video_propagation(self):
        """初始化视频传播预测器"""

        if not hasattr(self, 'inference_state'):
            print("Cannot initialize video propagation - inference state missing")
            return

        # 创建视频传播生成器
        self.frame_generator = predictor.propagate_in_video(self.inference_state)
        self.generated_frames = {}  # 存储生成的帧数据
        self.propagation_started = False

        print("Video propagation generator initialized")

    def generate_frame(self, frame_idx):
        """生成指定帧的预测结果（如果尚未生成）"""
        if not hasattr(self, 'frame_generator') or self.frame_generator is None:
            print("Frame generator not available")
            return False

        # 如果已经生成了该帧，直接返回
        if frame_idx in self.generated_frames:
            return True

        # 如果尚未开始传播，从第一帧开始
        if not self.propagation_started:
            self.propagation_started = True
            current_frame = 0
        else:
            # 获取最后一个已生成帧的索引
            if self.generated_frames:
                current_frame = max(self.generated_frames.keys()) + 1
            else:
                current_frame = 0

        # 继续生成直到达到目标帧
        try:
            for _ in range(frame_idx - current_frame + 1):
                a = 0
                frame_idx_gen, obj_ids, mask_logits = next(self.frame_generator)
                self.generated_frames[frame_idx_gen] = (obj_ids, mask_logits)
                print(f"Generated frame {frame_idx_gen}")

                # 如果这是我们需要的帧，停止生成
                if frame_idx_gen == frame_idx:
                    break

        except StopIteration:
            print("Reached end of video propagation")
            self.frame_generator = None
            return False

        return frame_idx in self.generated_frames

    def ensure_data_structures_exist(self):
        """确保核心数据结构存在"""
        if not hasattr(self, 'masks'):
            self.masks = {}
        if not hasattr(self, 'object_points_per_frame'):
            self.object_points_per_frame = {}
        if not hasattr(self, 'object_boxes_per_frame'):
            self.object_boxes_per_frame = {}
        if not hasattr(self, 'generated_frames'):
            self.generated_frames = {}
        if not hasattr(self, 'frame_generator'):
            self.frame_generator = None
        if not hasattr(self, 'propagation_started'):
            self.propagation_started = False

    def ensure_frame_structures_exist(self, frame_idx):
        """确保特定帧的数据结构存在"""
        if frame_idx not in self.masks:
            self.masks[frame_idx] = {}
        if frame_idx not in self.object_points_per_frame:
            self.object_points_per_frame[frame_idx] = {}
        if frame_idx not in self.object_boxes_per_frame:
            self.object_boxes_per_frame[frame_idx] = {}

    def hsl_to_rgb(self, h, s, l):
        """将HSL颜色转换为RGB颜色，使用更精确的算法"""
        h = h / 360.0
        s = s / 100.0
        l = l / 100.0

        if s == 0:
            r = g = b = l
        else:
            def hue_to_rgb(p, q, t):
                if t < 0: t += 1
                if t > 1: t -= 1
                if t < 1 / 6: return p + (q - p) * 6 * t
                if t < 1 / 2: return q
                if t < 2 / 3: return p + (q - p) * (2 / 3 - t) * 6
                return p

            q = l * (1 + s) if l < 0.5 else l + s - l * s
            p = 2 * l - q
            r = hue_to_rgb(p, q, h + 1 / 3)
            g = hue_to_rgb(p, q, h)
            b = hue_to_rgb(p, q, h - 1 / 3)

        return int(r * 255), int(g * 255), int(b * 255)

    def select_object(self, btn):
        # 取消选择其他按钮
        for obj_id, button_widget in self.obj_buttons.items():
            if button_widget != btn:
                button_widget.button.setChecked(False)

        # 选择点击的按钮
        btn.button.setChecked(True)
        self.image_label.set_current_object(btn.obj_id)

        # 确保图像标签获得焦点
        self.image_label.setFocus()
        print(f"Selected object: {btn.obj_id}")

    def delete_object(self, obj_id):
        if obj_id in self.obj_buttons:
            print(f"Deleting object: {obj_id}")

            # 清理图像标签中的掩码和点
            self.image_label.remove_mask(obj_id)
            self.image_label.clear_object_points(obj_id)

            # 移除对象按钮
            button_widget = self.obj_buttons.pop(obj_id)
            button_widget.deleteLater()

            # 移除对象颜色
            if obj_id in self.obj_colors:
                del self.obj_colors[obj_id]

            # 如果这是当前选择的对象，取消选择
            if self.image_label.current_obj_id == obj_id:
                self.image_label.set_current_object(None)
                print("Current object deselected")

            # 强制更新显示
            self.image_label.update_display()

            # 如果没有对象了，重置选择
            if not self.obj_buttons and self.obj_counter > 1:
                self.image_label.set_current_object(None)

            # 从所有帧状态中删除该对象
            for frame_idx in self.frame_states:
                state = self.frame_states[frame_idx]
                if 'masks' in state and obj_id in state['masks']:
                    del state['masks'][obj_id]
                if 'object_points' in state and obj_id in state['object_points']:
                    del state['object_points'][obj_id]
                if 'object_labels' in state and obj_id in state['object_labels']:
                    del state['object_labels'][obj_id]
                if 'object_boxes' in state and obj_id in state['object_boxes']:
                    del state['object_boxes'][obj_id]

            return True
        return False

    def update_nav_buttons(self):
        """更新导航按钮状态"""
        if self.frame_names:
            self.prev_btn.setEnabled(self.current_frame_idx > 0)
            self.next_btn.setEnabled(self.current_frame_idx < len(self.frame_names) - 1)
            # 更新滚动条范围和值
            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(len(self.frame_names) - 1)
            self.frame_slider.setValue(self.current_frame_idx)
        else:
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.frame_slider.setEnabled(False)
        self.update_frame_position_label()

    def update_frame_position_label(self):
        """更新帧位置标签显示"""
        text = f"{self.current_frame_idx + 1}/{len(self.frame_names)}"
        self.frame_position_label.setText(text)

    def save_current_frame_state(self):
        """保存当前帧的状态"""
        if self.current_frame_idx in self.frame_states:
            # 如果状态已存在，更新它
            self.frame_states[self.current_frame_idx] = self.image_label.get_state()
        else:
            # 否则添加新状态
            self.frame_states[self.current_frame_idx] = self.image_label.get_state()

        print(f"Saved state for frame {self.current_frame_idx}")
        print(f"Current frame states: {len(self.frame_states)} frames saved")

    def prev_frame(self):
        # 检查未保存修改
        result = self.check_for_unsaved_changes()
        if result != QMessageBox.Yes:
            if result == QMessageBox.Cancel:
                return

        # 保存当前帧状态
        self.save_current_frame_state()

        # 继续原逻辑
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.load_frame()

        self.update_prompt_status()

    def next_frame(self):
        # 检查未保存修改
        result = self.check_for_unsaved_changes()
        if result != QMessageBox.Yes:
            if result == QMessageBox.Cancel:
                return

        # 保存当前帧状态
        self.save_current_frame_state()

        # 继续原逻辑
        if self.current_frame_idx < len(self.frame_names) - 1:
            self.current_frame_idx += 1
            self.load_frame()

        self.update_prompt_status()

    def load_frame(self):
        """加载帧的方法（重写以使用传播预测结果）"""
        # 确保数据结构存在
        self.ensure_data_structures_exist()

        if not self.frame_names:
            return

        try:
            frame_path = os.path.join(self.frame_dir, self.frame_names[self.current_frame_idx])

            # 设置新图像 - 这一步总是需要执行
            self.image_label.set_image(frame_path)

            # 重置当前帧的状态（保留点/框等状态，但清除展示）
            self.image_label.reset_state()

            # 检查当前帧是否有未导入的提示
            if self.current_frame_idx in self.pending_prompts and len(self.pending_prompts[self.current_frame_idx]) > 0 :
                # 询问用户是否导入提示
                reply = QMessageBox.question(
                    self, "Import Prompts?",
                    f"Found {len(self.pending_prompts[self.current_frame_idx])} prompts for this frame. Import them?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                )

                if reply == QMessageBox.Yes:
                    self.process_prompts_for_frame(
                        self.current_frame_idx,
                        self.pending_prompts[self.current_frame_idx]
                    )
                    # 删除已处理的提示
                    del self.pending_prompts[self.current_frame_idx]

                # 更新状态栏
                self.update_prompt_status()
                return

            # 1. 首先尝试获取传播生成的掩码
            propagated_masks_available = False
            if hasattr(self, 'frame_generator') and self.frame_generator is not None:
                if self.generate_frame(self.current_frame_idx):
                    # 从传播生成器获取结果
                    obj_ids, mask_logits = self.generated_frames[self.current_frame_idx]

                    # 创建或更新当前帧的mask字典
                    if self.current_frame_idx not in self.masks:
                        self.masks[self.current_frame_idx] = {}

                    # 存储生成的掩码
                    for i, obj_id in enumerate(obj_ids):
                        mask = (mask_logits[i] > 0.0).cpu().numpy()
                        self.masks[self.current_frame_idx][obj_id] = mask

                        self.image_label.add_mask(obj_id, mask)
                        if obj_id in self.obj_buttons:
                            self.obj_buttons[obj_id].update_button_style(active=True)

                    # 更新UI显示
                    # self.image_label.object_masks = self.masks[self.current_frame_idx]

                    print(f"Loaded propagated masks for frame {self.current_frame_idx}")
                    propagated_masks_available = True
                    # 立即更新显示
                    self.image_label.update_display()

            # 2. 如果没有传播生成的掩码，尝试恢复保存的状态
            if not propagated_masks_available:
                if self.current_frame_idx in self.frame_states:
                    state = self.frame_states[self.current_frame_idx]
                    self.image_label.restore_state(state)
                    print(f"Restored saved state for frame {self.current_frame_idx}")
                else:
                    # 使用当前mask数据（如果有）
                    if self.current_frame_idx in self.masks and self.masks[self.current_frame_idx]:
                        self.image_label.object_masks = self.masks[self.current_frame_idx]
                        print(f"Using propagated masks for frame {self.current_frame_idx}")
                    else:
                        # 重置状态
                        self.image_label.reset_state()
                        print(f"No saved state for frame {self.current_frame_idx}, resetting")

            # 3. 如果有预测器，处理推理状态（但不覆盖现有展示）
            if predictor and not propagated_masks_available:
                # 初始化或重置推断状态
                if self.inference_state is None:
                    # self.inference_state = predictor.init_state(video_path=self.frame_dir)
                    self.show()
                    # 在显示主窗口后才显示加载对话框
                    self.loading_dialog = LoadingDialog(self)
                    self.loading_dialog.show()

                    # 初始化预测器（在后台线程执行）
                    self.init_thread = InitThread(self.frame_dir)
                    self.init_thread.finished.connect(self.on_init_completed)
                    self.init_thread.error.connect(self.on_init_error)
                    self.init_thread.start()
                    print("Initialized inference state")
                else:
                    # 重置状态（不覆盖UI显示）
                    predictor.reset_state(self.inference_state)
                    print("Reset inference state for propagation")

                    # 重放所有对象的点和框以重新生成掩码
                    for obj_id in self.image_label.object_points:
                        if obj_id in self.obj_buttons:
                            points = self.image_label.object_points.get(obj_id, [])
                            labels = self.image_label.object_labels.get(obj_id, [])
                            box = self.image_label.object_boxes.get(obj_id, None)
                            if points and labels or box is not None:
                                print(f"Replaying points and box for obj_id {obj_id}")
                                self.run_sam_prediction(points, labels, obj_id, box=box)

            # 更新按钮激活状态
            for obj_id, btn in self.obj_buttons.items():
                if self.image_label.masks.get(obj_id) is not None:
                    btn.update_button_style(active=True)
                else:
                    btn.update_button_style(active=False)

            # 重置当前对象ID
            if self.obj_buttons:
                # 尝试保留当前选中的对象
                if self.image_label.current_obj_id not in self.obj_buttons:
                    # 如果没有选中对象，选择第一个可用的对象
                    first_obj_id = next(iter(self.obj_buttons.keys()))
                    self.obj_buttons[first_obj_id].button.setChecked(True)
                    self.select_object(self.obj_buttons[first_obj_id])
            else:
                self.image_label.set_current_object(None)

            # 更新导航按钮
            self.update_nav_buttons()

            # 更新窗口标题
            self.setWindowTitle(
                f"SAM2 Interactive Annotation - Frame {self.current_frame_idx + 1}/{len(self.frame_names)}")

            # 确保图像标签获得焦点
            self.image_label.setFocus()

            # 强制更新显示
            self.image_label.update_display()

        except Exception as e:
            print(f"Error loading frame: {str(e)}")
            import traceback
            traceback.print_exc()

        self.update_frame_position_label()

    def on_init_completed(self, state):
        """初始化完成回调"""
        self.inference_state = state

        # 关闭加载对话框
        if self.loading_dialog:
            # 恢复标准输出
            sys.stdout = sys.__stdout__

            # 关闭对话框
            self.loading_dialog.accept()
            self.loading_dialog = None

        # 继续加载帧
        print("Initialized inference state")
        self.load_frame()

    def on_init_error(self, error):
        """初始化错误回调"""
        # 关闭加载对话框
        if self.loading_dialog:
            sys.stdout = sys.__stdout__
            self.loading_dialog.reject()
            self.loading_dialog = None

        # 显示错误消息
        QMessageBox.critical(self, "Initialization Error", f"Failed to initialize SAM2:\n{str(error)}")

    def process_prompts_for_frame(self, frame_idx, objects):
        """处理特定帧的提示对象"""
        print(f"Processing prompts for frame {frame_idx}")

        # 重置预测器状态
        if predictor:
            if self.inference_state is not None:
                predictor.reset_state(self.inference_state)
            else:
                self.inference_state = predictor.init_state(video_path=self.frame_dir)

        # 为相同classTitle的对象生成唯一obj_id
        class_counts = {}  # 跟踪每个类别的数量

        for obj in objects:
            if obj.get('geometryType') != 'rectangle':
                continue

            class_title = obj.get('classTitle', '')
            base_id = None

            # 查找映射表
            if class_title in class_map_t:
                base_id = int(class_map_t[class_title])
            elif class_title in class_map_i:
                base_id = int(class_map_i[class_title])

            # 生成唯一obj_id
            if base_id is not None:
                # 更新类别计数
                if base_id not in class_counts:
                    class_counts[base_id] = 1
                else:
                    class_counts[base_id] += 1

                # 创建唯一obj_id (base_id * 1000 + sequence)
                obj_id = base_id * 1000 + class_counts[base_id]
            else:
                # 生成新的唯一ID
                obj_id = self.obj_counter
                self.obj_counter += 1

            # 创建对象（如果不存在）
            if obj_id not in self.obj_buttons:
                name = class_title or f"Object {obj_id}"

                # 生成颜色
                hue = (obj_id * 137.5) % 360
                r, g, b = self.hsl_to_rgb(hue, 65, 60)
                obj_color = (r, g, b, 180)

                # 创建对象按钮
                btn = ObjectButton(self, obj_id, name, obj_color[:3], active=True)
                self.obj_buttons[obj_id] = btn
                self.obj_colors[obj_id] = obj_color
                self.obj_layout.addWidget(btn)

            # 解析边界框
            if 'points' not in obj or 'exterior' not in obj['points']:
                continue

            exterior = obj['points']['exterior']
            if len(exterior) < 2:
                continue

            # 转换为[x1, y1, x2, y2]格式
            x1 = min(exterior[0][0], exterior[1][0])
            y1 = min(exterior[0][1], exterior[1][1])
            x2 = max(exterior[0][0], exterior[1][0])
            y2 = max(exterior[0][1], exterior[1][1])
            bbox = [x1, y1, x2, y2]

            # 立即运行SAM2预测（只在当前帧处理）
            if frame_idx == self.current_frame_idx:
                self.run_sam_prediction(
                    points=[],
                    labels=[],
                    obj_id=obj_id,
                    box=bbox
                )
            else:
                # 更新预测器状态
                self.update_predictor_state(
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    box=bbox
                )

        # 设置最近使用的提示帧
        self.last_prompt_frame = frame_idx

        # 如果是处理当前帧，初始化传播
        if frame_idx == self.current_frame_idx:
            self.initialize_video_propagation()

        # 更新状态栏
        self.update_prompt_status()

    def run_sam_prediction(self, points, labels, obj_id, box=None):
        """运行SAM2预测并更新掩码，支持box参数"""
        print(f'Now the id {obj_id}, points {points}, labels {labels}')
        print(f"Running SAM prediction for obj {obj_id} with {len(points)} points and box {box}")

        if obj_id is None or not self.frame_names:
            print(f"Skip SAM prediction: invalid parameters (obj_id={obj_id})")
            return

        # 如果没有预测器，使用纯Python方法模拟结果
        if predictor is None:
            print("Predictor not available - simulating mask")
            if self.image_label.original_image:
                w, h = self.image_label.original_image.size
                mask = np.zeros((h, w), dtype=np.uint8)

                # 模拟框
                if box is not None:
                    x1, y1, x2, y2 = box
                    mask[y1:y2, x1:x2] = 1
                    print(f"Added box for obj_id {obj_id}: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

                # 使用纯Python方法创建圆形区域
                for i, point in enumerate(points):
                    center_x, center_y = int(point[0]), int(point[1])
                    label = labels[i] if i < len(labels) else 1
                    radius = 30

                    # 如果是正样本点，添加白色区域
                    # 如果是负样本点，移除区域
                    value = 1 if label == 1 else 0

                    # 计算圆内点
                    for y in range(max(0, center_y - radius), min(h, center_y + radius + 1)):
                        for x in range(max(0, center_x - radius), min(w, center_x + radius + 1)):
                            dx = x - center_x
                            dy = y - center_y
                            distance_squared = dx * dx + dy * dy
                            if distance_squared <= radius * radius:
                                if value == 1:
                                    mask[y, x] = 1
                                else:
                                    mask[y, x] = 0

                self.image_label.add_mask(obj_id, mask)
                print(f"Simulated mask for obj_id {obj_id}: {mask.shape}, {mask.sum()} non-zero pixels")
            return

        try:
            # 确保点是二维数组
            points_np = None
            if points:
                points_np = np.array(points, dtype=np.float32)
                if points_np.ndim == 1:
                    points_np = points_np.reshape(1, -1)

            # 确保标签是一维数组
            labels_np = None
            if labels:
                labels_np = np.array(labels, np.int32)
                if labels_np.ndim > 1:
                    labels_np = labels_np.flatten()

            # 转换box格式
            box_np = None
            if box:
                box_np = np.array(box, dtype=np.float32)

            # 通过SAM2处理
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=self.current_frame_idx,
                obj_id=obj_id,
                points=points_np,
                labels=labels_np,
                box=box_np
            )

            # 转换掩码到numpy格式
            print(f'out obj ids {out_obj_ids}')
            print(out_mask_logits.cpu().numpy().shape)
            if out_mask_logits.nelement() > 0:
                position = out_obj_ids.index(obj_id)
                mask = (out_mask_logits[position] > 0.0).cpu().numpy()
                # 添加掩码
                self.image_label.add_mask(obj_id, mask)
                # 激活按钮（变为对象颜色）
                if obj_id in self.obj_buttons:
                    self.obj_buttons[obj_id].update_button_style(active=True)
                print(f"Generated mask for obj_id {obj_id}: {mask.shape}, {mask.sum()} non-zero pixels")

        except Exception as e:
            print(f"Error running SAM2 prediction for object {obj_id}: {str(e)}")
            import traceback
            traceback.print_exc()

    def keyPressEvent(self, event):
        # 键盘事件：A键后退一帧，D键前进一帧
        if event.key() == Qt.Key_A:
            self.prev_frame()
        elif event.key() == Qt.Key_D:
            self.next_frame()
        else:
            # 传递其他键盘事件给图像标签
            self.image_label.keyPressEvent(event)


if __name__ == "__main__":
    # 检查是否传递了文件夹路径作为参数
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
        print(f"Received folder path: {folder_path}")

        if not os.path.isdir(folder_path):
            app = QApplication(sys.argv)
            QMessageBox.critical(None, "Error", f"Invalid folder path: {folder_path}")
            sys.exit(1)
    else:
        # 如果没有提供参数，使用文件夹对话框
        app = QApplication(sys.argv)
        folder_path = QFileDialog.getExistingDirectory(None, "Select Image Folder")
        if not folder_path:
            sys.exit(0)
        print(f"Selected folder path: {folder_path}")

    # 运行标注工具
    app = QApplication(sys.argv)
    window = SmartAnnotationTool()

    # 加载文件夹
    try:
        window.frame_dir = folder_path
        window.frame_names = [
            p for p in os.listdir(folder_path)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png", ".tiff"]
        ]


        # 按数字值排序文件名
        def extract_number(filename):
            base = os.path.splitext(filename)[0]
            if base.isdigit():
                return int(base)
            try:
                # 尝试提取文件名中的数字
                return int(''.join(filter(str.isdigit, base)))
            except:
                return 0


        window.frame_names.sort(key=extract_number)

        if not window.frame_names:
            QMessageBox.critical(None, "Error", "No image files found in the selected folder")
            sys.exit(1)

        print(f"Found {len(window.frame_names)} images")

        # 初始化
        window.current_frame_idx = 0
        window.load_frame()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        app = QApplication(sys.argv)
        QMessageBox.critical(None, "Error", f"Initialization failed: {str(e)}")
        sys.exit(1)