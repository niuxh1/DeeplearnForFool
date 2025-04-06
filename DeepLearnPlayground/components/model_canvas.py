#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型画布 - 用于构建神经网络模型的交互式画布
"""

import os
import json
import uuid
import pickle
from PyQt5.QtWidgets import (QGraphicsView, QGraphicsScene, QWidget, QVBoxLayout,
                           QGraphicsItem, QGraphicsRectItem, QGraphicsTextItem,
                           QGraphicsLineItem, QGraphicsEllipseItem, QMenu,
                           QAction, QInputDialog, QMessageBox, QDialog, QFormLayout,
                           QLineEdit, QDialogButtonBox, QComboBox, QSpinBox, QDoubleSpinBox,
                           QLabel, QGraphicsProxyWidget, QPushButton)
from PyQt5.QtCore import Qt, QRectF, QPointF, QLineF, pyqtSignal
from PyQt5.QtGui import QBrush, QPen, QColor, QFont, QPainter, QDrag, QPolygonF, QTransform

# 组件默认配置
DEFAULT_CONFIGS = {
    "conv2d": {
        "in_channels": 3,
        "out_channels": 16,
        "kernel_size": 3,
        "stride": 1,
        "padding": 0
    },
    "maxpool": {
        "kernel_size": 2,
        "stride": 2
    },
    "linear": {
        "in_features": 64,
        "out_features": 10
    },
    "dropout": {
        "p": 0.5
    },
    "batchnorm": {
        "num_features": 16
    },
    "relu": {
        "inplace": True
    },
    "input": {
        "shape": [3, 32, 32]
    },
    "output": {
        "num_classes": 10
    },
    "flatten": {},
    "lstm": {
        "input_size": 28,
        "hidden_size": 64,
        "num_layers": 1
    },
    "gru": {
        "input_size": 28,
        "hidden_size": 64,
        "num_layers": 1
    },
    "attention": {
        "embed_dim": 64,
        "num_heads": 4
    },
    "residual": {}
}

# 组件颜色
COMPONENT_COLORS = {
    "conv2d": "#3498db",     # 蓝色
    "maxpool": "#2ecc71",    # 绿色
    "linear": "#e74c3c",     # 红色
    "relu": "#f39c12",       # 橙色
    "dropout": "#9b59b6",    # 紫色
    "batchnorm": "#1abc9c",  # 青绿色
    "input": "#95a5a6",      # 灰色
    "output": "#34495e",     # 深灰色
    "flatten": "#f1c40f",    # 黄色
    "lstm": "#d35400",       # 棕色
    "gru": "#c0392b",        # 暗红色
    "attention": "#8e44ad",  # 深紫色
    "residual": "#16a085"    # 深青色
}

# 组件显示名称
COMPONENT_NAMES = {
    "conv2d": "卷积层",
    "maxpool": "池化层",
    "linear": "全连接层",
    "relu": "ReLU激活",
    "dropout": "Dropout",
    "batchnorm": "批归一化",
    "input": "输入层",
    "output": "输出层",
    "flatten": "展平层",
    "lstm": "LSTM层",
    "gru": "GRU层",
    "attention": "注意力层",
    "residual": "残差连接"
}

class ConnectionLine(QGraphicsLineItem):
    """表示神经网络中组件之间的连接线"""
    
    def __init__(self, start_component, end_component, parent=None):
        super().__init__(parent)
        self.start_component = start_component
        self.end_component = end_component
        self.setZValue(-1)  # 确保线条位于组件下方
        
        # 设置线条样式 - 使用更粗的线条和箭头
        pen = QPen(Qt.black, 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.setPen(pen)
        
        # 添加箭头标记
        self.arrow_size = 10
        
        # 更新线条位置
        self.update_position()
    
    def update_position(self):
        """更新连接线的位置"""
        if self.start_component and self.end_component:
            start_pos = self.start_component.get_output_pos()
            end_pos = self.end_component.get_input_pos()
            self.setLine(QLineF(start_pos, end_pos))
    
    def paint(self, painter, option, widget):
        """重写绘制方法，添加箭头"""
        super().paint(painter, option, widget)
        
        # 添加箭头
        line = self.line()
        if line.length() == 0:
            return
            
        angle = line.angle() - 180  # 箭头方向
        
        # 使用QTransform来旋转点，而不是使用不存在的rotated方法
        transform = QTransform()
        transform.rotate(angle)
        
        # 创建箭头点
        arrow_p1 = line.p2() - transform.map(QPointF(self.arrow_size * (1 + 0.5 * (angle > 90)), 0))
        
        # 创建另外两个箭头点
        transform30 = QTransform()
        transform30.rotate(angle + 30)
        arrow_p2 = line.p2() - transform30.map(QPointF(self.arrow_size * (1 - 0.5 * (angle > 90)), 0))
        
        transform_minus30 = QTransform()
        transform_minus30.rotate(angle - 30)
        arrow_p3 = line.p2() - transform_minus30.map(QPointF(self.arrow_size * (1 - 0.5 * (angle > 90)), 0))
        
        # 绘制填充箭头
        arrow = QPolygonF([line.p2(), arrow_p2, arrow_p3])
        painter.setBrush(Qt.black)
        painter.drawPolygon(arrow)

class ComponentItem(QGraphicsRectItem):
    """表示神经网络中的一个组件"""
    
    def __init__(self, component_type, x=0, y=0, parent=None):
        super().__init__(parent)
        self.component_type = component_type
        self.component_id = str(uuid.uuid4())
        self.connections_in = []
        self.connections_out = []
        
        # 初始化配置
        self.config = DEFAULT_CONFIGS.get(component_type, {}).copy()
        
        # 设置位置和大小 - 增加组件尺寸并优化
        self.setRect(0, 0, 160, 100)  # 增加高度解决遮挡问题
        self.setPos(x, y)
        
        # 设置样式
        color = QColor(COMPONENT_COLORS.get(component_type, "#cccccc"))
        self.setBrush(QBrush(color))
        self.setPen(QPen(Qt.black, 2))
        
        # 设置可选择和可移动
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        
        # 创建文本标签 - 调整位置和字体
        self.text_item = QGraphicsTextItem(self)
        self.text_item.setPos(15, 20)  # 垂直居中显示
        font = QFont()
        font.setPointSize(9)  # 增大字体
        self.text_item.setFont(font)
        self.update_text()
        
        # 创建输入输出连接点 - 更新连接点大小和位置
        self.input_point = QGraphicsEllipseItem(72, -10, 16, 16, self)  # 顶部中央
        self.input_point.setBrush(QBrush(Qt.white))
        self.input_point.setPen(QPen(Qt.black, 2))
        
        self.output_point = QGraphicsEllipseItem(72, 94, 16, 16, self)  # 底部中央(更新位置)
        self.output_point.setBrush(QBrush(Qt.white))
        self.output_point.setPen(QPen(Qt.black, 2))
        
        # 添加连接点标签 - 调整位置
        self.input_label = QGraphicsTextItem("输入", self)
        self.input_label.setPos(80, -25)
        self.input_label.setDefaultTextColor(Qt.darkGray)
        font = QFont()
        font.setPointSize(7)
        self.input_label.setFont(font)
        
        self.output_label = QGraphicsTextItem("输出", self)
        self.output_label.setPos(80, 110)  # 更新标签位置
        self.output_label.setDefaultTextColor(Qt.darkGray)
        self.output_label.setFont(font)
    
    def update_text(self):
        """更新组件显示的文本"""
        name = COMPONENT_NAMES.get(self.component_type, self.component_type)
        
        # 根据组件类型添加额外信息
        if self.component_type == "conv2d":
            extra = f"{self.config['in_channels']}→{self.config['out_channels']}, k={self.config['kernel_size']}"
        elif self.component_type == "linear":
            extra = f"{self.config['in_features']}→{self.config['out_features']}"
        elif self.component_type == "dropout":
            extra = f"p={self.config['p']}"
        elif self.component_type == "maxpool":
            extra = f"k={self.config['kernel_size']}"
        elif self.component_type == "input":
            shape_str = "x".join([str(dim) for dim in self.config['shape']])
            extra = f"{shape_str}"
        elif self.component_type == "output":
            extra = f"类别数={self.config['num_classes']}"
        elif self.component_type in ["lstm", "gru"]:
            extra = f"隐藏={self.config['hidden_size']}"
        else:
            extra = ""
        
        if extra:
            text = f"{name}\n{extra}"
        else:
            text = name
            
        self.text_item.setPlainText(text)
    
    def get_input_pos(self):
        """获取输入连接点的绝对位置"""
        return self.mapToScene(80, 0)  # 顶部中央
    
    def get_output_pos(self):
        """获取输出连接点的绝对位置"""
        return self.mapToScene(80, 100)  # 底部中央
    
    def itemChange(self, change, value):
        """当组件位置变化时更新连接线"""
        if change == QGraphicsItem.ItemPositionHasChanged:
            for conn in self.connections_in + self.connections_out:
                conn.update_position()
        return super().itemChange(change, value)
    
    def mousePressEvent(self, event):
        """处理鼠标按下事件"""
        if event.button() == Qt.RightButton:
            self.show_context_menu(event.screenPos())
        super().mousePressEvent(event)
    
    def mouseDoubleClickEvent(self, event):
        """处理鼠标双击事件"""
        self.edit_properties()
        super().mouseDoubleClickEvent(event)
    
    def show_context_menu(self, pos):
        """显示右键菜单"""
        menu = QMenu()
        
        # 编辑属性
        edit_action = QAction("编辑属性", menu)
        edit_action.triggered.connect(self.edit_properties)
        menu.addAction(edit_action)
        
        # 删除组件
        delete_action = QAction("删除组件", menu)
        delete_action.triggered.connect(self.remove_self)
        menu.addAction(delete_action)
        
        menu.exec_(pos)
    
    def remove_self(self):
        """从场景中移除自己"""
        # 移除所有连接
        for conn in self.connections_in + self.connections_out:
            if conn.scene():
                conn.scene().removeItem(conn)
                
            # 更新相连组件的连接列表
            if conn in self.connections_in:
                start_comp = conn.start_component
                if start_comp and conn in start_comp.connections_out:
                    start_comp.connections_out.remove(conn)
            
            if conn in self.connections_out:
                end_comp = conn.end_component
                if end_comp and conn in end_comp.connections_in:
                    end_comp.connections_in.remove(conn)
        
        # 清空连接列表
        self.connections_in = []
        self.connections_out = []
        
        # 从场景中移除
        if self.scene():
            self.scene().removeItem(self)
    
    def edit_properties(self):
        """编辑组件属性"""
        dialog = ComponentPropertiesDialog(self.component_type, self.config)
        if dialog.exec_() == QDialog.Accepted:
            self.config = dialog.get_config()
            self.update_text()
    
    def to_dict(self):
        """将组件转换为字典表示"""
        return {
            "id": self.component_id,
            "type": self.component_type,
            "x": self.x(),
            "y": self.y(),
            "config": self.config
        }

class ComponentPropertiesDialog(QDialog):
    """组件属性编辑对话框"""
    
    def __init__(self, component_type, config, parent=None):
        super().__init__(parent)
        self.component_type = component_type
        self.config = config.copy()
        
        self.init_ui()
    
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle(f"编辑{COMPONENT_NAMES.get(self.component_type, self.component_type)}属性")
        
        layout = QFormLayout(self)
        
        self.editors = {}
        
        # 根据组件类型添加不同的属性编辑器
        if self.component_type == "conv2d":
            self.add_spin_box(layout, "in_channels", "输入通道数", 1, 1000)
            self.add_spin_box(layout, "out_channels", "输出通道数", 1, 1000)
            self.add_spin_box(layout, "kernel_size", "卷积核大小", 1, 10)
            self.add_spin_box(layout, "stride", "步长", 1, 10)
            self.add_spin_box(layout, "padding", "填充", 0, 10)
            
        elif self.component_type == "maxpool":
            self.add_spin_box(layout, "kernel_size", "池化核大小", 1, 10)
            self.add_spin_box(layout, "stride", "步长", 1, 10)
            
        elif self.component_type == "linear":
            self.add_spin_box(layout, "in_features", "输入特征数", 1, 10000)
            self.add_spin_box(layout, "out_features", "输出特征数", 1, 10000)
            
        elif self.component_type == "dropout":
            self.add_double_spin_box(layout, "p", "丢弃概率", 0.0, 1.0, 0.1)
            
        elif self.component_type == "batchnorm":
            self.add_spin_box(layout, "num_features", "特征数", 1, 1000)
            
        elif self.component_type == "input":
            # 创建输入形状编辑器
            shape = self.config.get("shape", [3, 32, 32])
            
            self.add_spin_box(layout, "channels", "通道数", 1, 1000, shape[0])
            self.add_spin_box(layout, "height", "高度", 1, 1000, shape[1])
            self.add_spin_box(layout, "width", "宽度", 1, 1000, shape[2])
            
        elif self.component_type == "output":
            self.add_spin_box(layout, "num_classes", "类别数", 1, 1000)
            
        elif self.component_type in ["lstm", "gru"]:
            self.add_spin_box(layout, "input_size", "输入大小", 1, 1000)
            self.add_spin_box(layout, "hidden_size", "隐藏层大小", 1, 1000)
            self.add_spin_box(layout, "num_layers", "层数", 1, 10)
            
        elif self.component_type == "attention":
            self.add_spin_box(layout, "embed_dim", "嵌入维度", 1, 1000)
            self.add_spin_box(layout, "num_heads", "注意力头数", 1, 16)
        
        # 添加按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)
    
    def add_spin_box(self, layout, name, label, minimum, maximum, value=None):
        """添加一个整数输入框"""
        spin_box = QSpinBox()
        spin_box.setMinimum(minimum)
        spin_box.setMaximum(maximum)
        
        if value is not None:
            spin_box.setValue(value)
        elif name in self.config:
            spin_box.setValue(self.config[name])
            
        layout.addRow(label, spin_box)
        self.editors[name] = spin_box
    
    def add_double_spin_box(self, layout, name, label, minimum, maximum, step=0.1):
        """添加一个浮点数输入框"""
        spin_box = QDoubleSpinBox()
        spin_box.setMinimum(minimum)
        spin_box.setMaximum(maximum)
        spin_box.setSingleStep(step)
        
        if name in self.config:
            spin_box.setValue(self.config[name])
            
        layout.addRow(label, spin_box)
        self.editors[name] = spin_box
    
    def get_config(self):
        """获取编辑后的配置"""
        result = self.config.copy()
        
        for name, editor in self.editors.items():
            if self.component_type == "input" and name in ["channels", "height", "width"]:
                # 处理输入形状的特殊情况
                if "shape" not in result:
                    result["shape"] = [3, 32, 32]
                
                if name == "channels":
                    result["shape"][0] = editor.value()
                elif name == "height":
                    result["shape"][1] = editor.value()
                elif name == "width":
                    result["shape"][2] = editor.value()
            else:
                result[name] = editor.value()
                
        return result

class ModelCanvas(QGraphicsView):
    """模型画布类，用于构建神经网络模型"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # 设置画布属性
        self.setRenderHint(QPainter.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        
        # 增加画布初始大小，提供更多空间
        self.scene.setSceneRect(-500, -500, 2000, 2000)
        
        # 设置自动滚动和缩放功能
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        
        # 连接状态
        self.connecting = False
        self.connection_start = None
        self.temp_line = None
        
        # 添加连线提示
        self.connection_hint = QGraphicsTextItem("点击底部输出点并拖到另一个组件的顶部输入点完成连线")  # 更新提示文本
        self.connection_hint.setDefaultTextColor(Qt.gray)
        self.connection_hint.setPos(10, 10)
        self.scene.addItem(self.connection_hint)
        
        # 组件和连接线列表
        self.components = []
        self.connections = []
        
        # 当前加载的数据集
        self.current_dataset = None
        
        # 允许拖放
        self.setAcceptDrops(True)
        
        # 显示网格背景，帮助用户定位
        self.setBackgroundBrush(QBrush(QColor("#f5f5f5")))
        
        # 添加自动连线模式
        self.auto_connect_mode = True  # 默认开启自动连线
    
    def dragEnterEvent(self, event):
        """处理拖拽进入事件"""
        if event.mimeData().hasText():
            event.acceptProposedAction()
    
    def dragMoveEvent(self, event):
        """处理拖拽移动事件"""
        if event.mimeData().hasText():
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        """处理拖拽放置事件"""
        if event.mimeData().hasText():
            component_type = event.mimeData().text()
            
            # 获取放置位置
            pos = self.mapToScene(event.pos())
            
            # 创建新组件
            self.add_component(component_type, pos.x(), pos.y())
            
            event.acceptProposedAction()
    
    def mousePressEvent(self, event):
        """处理鼠标按下事件"""
        if event.button() == Qt.LeftButton:
            # 获取点击位置的项
            pos = self.mapToScene(event.pos())
            item = self.itemAt(event.pos())
            
            # 检查是否点击了组件的连接点
            if isinstance(item, QGraphicsEllipseItem) and item.parentItem():
                parent = item.parentItem()
                if isinstance(parent, ComponentItem):
                    # 如果是输出点，开始绘制连接线
                    if item == parent.output_point:
                        self.connecting = True
                        self.connection_start = parent
                        self.connection_hint.setPlainText("拖动到目标组件的输入点")
                        self.connection_hint.setDefaultTextColor(Qt.blue)
                        
                        # 创建临时线，更明显的虚线
                        start_pos = parent.get_output_pos()
                        self.temp_line = QGraphicsLineItem(QLineF(start_pos, pos))
                        # 使用更醒目的连线样式
                        pen = QPen(QColor("#3498db"), 2.5, Qt.DashLine)
                        self.temp_line.setPen(pen)
                        self.scene.addItem(self.temp_line)
                        
                        # 高亮显示所有可连接的输入点
                        self.highlight_valid_inputs()
                        
                        event.accept()
                        return
                    
                    # 如果是输入点，且正在连接，完成连接
                    elif item == parent.input_point and self.connecting:
                        self.finish_connection(parent)
                        event.accept()
                        return
            
            # 尝试自动连接模式：点击组件直接开始连线
            elif self.auto_connect_mode and isinstance(item, ComponentItem) or (item and isinstance(item.parentItem(), ComponentItem)):
                component = item if isinstance(item, ComponentItem) else item.parentItem()
                # 如果已有开始组件，则尝试连接到当前组件的输入
                if self.connecting and self.connection_start and self.connection_start != component:
                    self.finish_connection(component)
                    event.accept()
                    return
                # 否则以当前组件为起点开始连接
                else:
                    self.connecting = True
                    self.connection_start = component
                    self.connection_hint.setPlainText("拖动到目标组件完成连接")
                    self.connection_hint.setDefaultTextColor(Qt.blue)
                    
                    # 创建临时线
                    start_pos = component.get_output_pos()
                    self.temp_line = QGraphicsLineItem(QLineF(start_pos, pos))
                    pen = QPen(QColor("#3498db"), 2.5, Qt.DashLine)
                    self.temp_line.setPen(pen)
                    self.scene.addItem(self.temp_line)
                    
                    # 高亮显示所有可连接的输入点
                    self.highlight_valid_inputs()
                    
                    event.accept()
                    return
        
        super().mousePressEvent(event)
    
    def highlight_valid_inputs(self):
        """高亮显示所有可连接的输入点"""
        if not self.connection_start:
            return
            
        # 遍历所有组件，高亮除了开始组件外的所有输入点
        for component in self.components:
            if component != self.connection_start:
                # 使输入点更明显
                component.input_point.setBrush(QBrush(QColor("#2ecc71")))  # 绿色
                component.input_point.setScale(1.2)  # 略微放大
    
    def reset_highlights(self):
        """重置所有高亮效果"""
        for component in self.components:
            component.input_point.setBrush(QBrush(Qt.white))
            component.input_point.setScale(1.0)
    
    def mouseMoveEvent(self, event):
        """处理鼠标移动事件"""
        if self.connecting and self.temp_line:
            # 更新临时线的终点
            pos = self.mapToScene(event.pos())
            line = self.temp_line.line()
            line.setP2(pos)
            self.temp_line.setLine(line)
            
            # 自动吸附到最近的输入点
            closest_component = self.find_closest_input(pos)
            if closest_component:
                line.setP2(closest_component.get_input_pos())
                self.temp_line.setLine(line)
                
                # 更新高亮
                self.reset_highlights()
                closest_component.input_point.setBrush(QBrush(QColor("#3498db")))  # 蓝色表示当前目标
                closest_component.input_point.setScale(1.3)
            
            event.accept()
        else:
            super().mouseMoveEvent(event)
    
    def find_closest_input(self, pos, threshold=100):
        """查找距离指定位置最近的输入点"""
        closest_distance = float('inf')
        closest_component = None
        
        for component in self.components:
            if component != self.connection_start:
                input_pos = component.get_input_pos()
                distance = QLineF(pos, input_pos).length()
                
                if distance < closest_distance and distance < threshold:
                    closest_distance = distance
                    closest_component = component
        
        return closest_component
    
    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件"""
        if event.button() == Qt.LeftButton and self.connecting:
            # 获取释放位置的项
            pos = self.mapToScene(event.pos())
            
            # 查找最近的输入点
            closest_component = self.find_closest_input(pos)
            if closest_component:
                self.finish_connection(closest_component)
            else:
                # 检查点击的项
                item = self.itemAt(event.pos())
                if isinstance(item, ComponentItem) or (item and isinstance(item.parentItem(), ComponentItem)):
                    component = item if isinstance(item, ComponentItem) else item.parentItem()
                    if component != self.connection_start:
                        self.finish_connection(component)
                    else:
                        self.cancel_connection()
                else:
                    self.cancel_connection()
            
            event.accept()
            return
        
        super().mouseReleaseEvent(event)
    
    def finish_connection(self, end_component):
        """完成连接线的创建"""
        if self.connection_start and end_component and self.connection_start != end_component:
            # 创建新的连接线
            connection = ConnectionLine(self.connection_start, end_component)
            self.scene.addItem(connection)
            
            # 更新组件的连接列表
            self.connection_start.connections_out.append(connection)
            end_component.connections_in.append(connection)
            
            # 添加到连接线列表
            self.connections.append(connection)
            
            # 显示成功消息
            self.connection_hint.setPlainText(f"已连接: {COMPONENT_NAMES.get(self.connection_start.component_type)} → {COMPONENT_NAMES.get(end_component.component_type)}")
            self.connection_hint.setDefaultTextColor(Qt.darkGreen)
        
        # 清理
        self.cancel_connection()
    
    def cancel_connection(self):
        """取消当前的连接操作"""
        if self.temp_line and self.temp_line.scene():
            self.scene.removeItem(self.temp_line)
        
        self.connecting = False
        self.connection_start = None
        self.temp_line = None
        
        # 重置高亮
        self.reset_highlights()
        
        # 重置提示
        self.connection_hint.setPlainText("点击底部输出点并拖到另一个组件的顶部输入点完成连线")
        self.connection_hint.setDefaultTextColor(Qt.gray)
    
    def keyPressEvent(self, event):
        """处理键盘事件"""
        # 按ESC键取消连接
        if event.key() == Qt.Key_Escape and self.connecting:
            self.cancel_connection()
            event.accept()
            return
        
        # 按Delete键删除选中的组件
        if event.key() == Qt.Key_Delete:
            selected_items = self.scene.selectedItems()
            for item in selected_items:
                if isinstance(item, ComponentItem):
                    item.remove_self()
            event.accept()
            return
            
        super().keyPressEvent(event)
    
    def contextMenuEvent(self, event):
        """显示右键菜单"""
        menu = QMenu(self)
        
        # 添加自动连线模式切换选项
        toggle_auto_connect = QAction("自动连线模式" if not self.auto_connect_mode else "精确连线模式", self)
        toggle_auto_connect.triggered.connect(self.toggle_connect_mode)
        menu.addAction(toggle_auto_connect)
        
        # 添加清除所有连线选项
        clear_connections = QAction("清除所有连线", self)
        clear_connections.triggered.connect(self.clear_all_connections)
        menu.addAction(clear_connections)
        
        # 执行菜单
        menu.exec_(event.globalPos())
    
    def toggle_connect_mode(self):
        """切换连线模式"""
        self.auto_connect_mode = not self.auto_connect_mode
        self.connection_hint.setPlainText(
            "自动连线模式：点击组件即可开始连线" if self.auto_connect_mode 
            else "精确连线模式：需点击输出/输入点连线"
        )
    
    def clear_all_connections(self):
        """清除所有连线"""
        for connection in list(self.connections):
            if connection.scene():
                connection.scene().removeItem(connection)
                
            # 更新相连组件的连接列表 - 添加检查确保连接在列表中
            if connection.start_component:
                if connection in connection.start_component.connections_out:
                    connection.start_component.connections_out.remove(connection)
                else:
                    print(f"警告: 连接不在起始组件的connections_out列表中")
            
            if connection.end_component:
                if connection in connection.end_component.connections_in:
                    connection.end_component.connections_in.remove(connection)
                else:
                    print(f"警告: 连接不在目标组件的connections_in列表中")
        
        # 清空连接列表
        self.connections = []
        self.connection_hint.setPlainText("已清除所有连线")
    
    def add_component(self, component_type, x, y):
        """添加一个新组件到画布"""
        component = ComponentItem(component_type, x, y)
        self.scene.addItem(component)
        self.components.append(component)
        return component
    
    def clear_canvas(self):
        """清空画布"""
        self.components = []
        self.connections = []
        self.scene.clear()
        self.current_dataset = None
    
    def get_model_dict(self):
        """获取模型的字典表示"""
        model_dict = {
            "components": [comp.to_dict() for comp in self.components],
            "connections": []
        }
        
        # 添加连接信息
        for conn in self.connections:
            if conn.start_component and conn.end_component:
                model_dict["connections"].append({
                    "from": conn.start_component.component_id,
                    "to": conn.end_component.component_id
                })
        
        return model_dict
    
    def save_model(self, filename):
        """保存模型到文件"""
        model_dict = self.get_model_dict()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(model_dict, f, indent=2)
    
    def load_model(self, filename):
        """从文件加载模型"""
        with open(filename, 'r', encoding='utf-8') as f:
            model_dict = json.load(f)
        
        # 清空当前画布
        self.clear_canvas()
        
        # 创建组件
        component_map = {}  # 用于映射ID到组件对象
        
        for comp_data in model_dict.get("components", []):
            component = self.add_component(
                comp_data["type"],
                comp_data.get("x", 0),
                comp_data.get("y", 0)
            )
            
            # 设置ID和配置
            component.component_id = comp_data["id"]
            component.config = comp_data.get("config", {})
            component.update_text()
            
            component_map[component.component_id] = component
        
        # 创建连接
        for conn_data in model_dict.get("connections", []):
            from_id = conn_data["from"]
            to_id = conn_data["to"]
            
            if from_id in component_map and to_id in component_map:
                start_component = component_map[from_id]
                end_component = component_map[to_id]
                
                # 创建连接线
                connection = ConnectionLine(start_component, end_component)
                self.scene.addItem(connection)
                
                # 更新组件的连接列表
                start_component.connections_out.append(connection)
                end_component.connections_in.append(connection)
                
                # 添加到连接线列表
                self.connections.append(connection)
    
    def load_template(self, template_name):
        """加载预定义的模型模板"""
        # 这里应该根据模板名称加载预定义的模型结构
        # 实际实现时，可以从templates目录加载JSON文件
        pass
    
    def load_dataset(self, dataset_name):
        """加载示例数据集"""
        self.current_dataset = dataset_name
    
    def has_dataset(self):
        """检查是否已加载数据集"""
        return self.current_dataset is not None
    
    def validate_model(self):
        """验证模型结构是否有效"""
        # 检查是否有输入和输出
        has_input = any(comp.component_type == "input" for comp in self.components)
        has_output = any(comp.component_type == "output" for comp in self.components)
        
        if not has_input:
            return False, "模型缺少输入层"
        
        if not has_output:
            return False, "模型缺少输出层"
        
        # 检查输入和输出层是否有连接
        input_connected = False
        output_connected = False
        
        # 获取所有输入和输出组件
        input_components = [comp for comp in self.components if comp.component_type == "input"]
        output_components = [comp for comp in self.components if comp.component_type == "output"]
        
        # 检查所有输入组件
        for input_comp in input_components:
            if input_comp.connections_out:
                input_connected = True
                break
        
        # 检查所有输出组件
        for output_comp in output_components:
            if output_comp.connections_in:
                output_connected = True
                break
        
        if not input_connected:
            return False, "输入层未连接到任何组件"
        
        if not output_connected:
            return False, "没有任何组件连接到输出层"
        
        # 检查是否有未连接的非输入/输出组件
        for comp in self.components:
            if comp.component_type not in ["input", "output"]:
                if not comp.connections_in:
                    return False, f"{COMPONENT_NAMES.get(comp.component_type, comp.component_type)}未连接任何输入"
                
                if not comp.connections_out:
                    return False, f"{COMPONENT_NAMES.get(comp.component_type, comp.component_type)}未连接任何输出"
        
        # 检查连接的有效性：确保每个连接都有有效的起点和终点
        for conn in self.connections:
            if not conn.start_component or not conn.end_component:
                return False, "模型中存在无效连接"
            
            if conn.start_component not in self.components or conn.end_component not in self.components:
                return False, "模型中存在连接到已删除组件的连接线"
        
        return True, "模型结构验证通过"
    
    def run_model(self):
        """运行模型进行训练"""
        # 这里应该实现实际的模型训练逻辑
        # 需要生成PyTorch代码并执行
        # 对于初学者来说，这部分可以简化为可视化的训练过程
        pass
    
    def train_model(self, params, progress_callback):
        """训练模型
        
        Args:
            params: 训练参数字典
            progress_callback: 进度回调函数，格式为 callback(epoch, total_epochs, loss, accuracy, message)
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            import torchvision
            import torchvision.transforms as transforms
            import numpy as np
            import time
            from torch.utils.data import DataLoader
        except ImportError:
            progress_callback(0, 1, 0, 0, "错误: 缺少必要的库。请安装PyTorch和torchvision。")
            return
        
        # 设置设备（GPU/CPU）
        device = torch.device("cuda" if params["use_gpu"] and torch.cuda.is_available() else "cpu")
        
        # 准备数据集
        if params["dataset"] == "mnist":
            progress_callback(0, params["epochs"], 0, 0, "正在加载MNIST数据集...")
            
            # 数据增强设置
            transform_train = transforms.Compose([
                transforms.RandomRotation(10) if params['data_augmentation']['random_crop'] else transforms.Lambda(lambda x: x),
                transforms.RandomHorizontalFlip() if params['data_augmentation']['random_flip'] else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)) if params['data_augmentation']['normalize'] else transforms.Lambda(lambda x: x)
            ])
            
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)) if params['data_augmentation']['normalize'] else transforms.Lambda(lambda x: x)
            ])
            
            # 尝试加载MNIST数据集
            try:
                train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
                test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
                
                train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)
                
                input_shape = [1, 28, 28]  # MNIST形状
                num_classes = 10
                
            except Exception as e:
                progress_callback(0, 1, 0, 0, f"加载MNIST数据集出错: {str(e)}")
                return
                
        elif params["dataset"] == "cifar10":
            progress_callback(0, params["epochs"], 0, 0, "正在加载CIFAR-10数据集...")
            
            # 数据增强设置
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4) if params['data_augmentation']['random_crop'] else transforms.Lambda(lambda x: x),
                transforms.RandomHorizontalFlip() if params['data_augmentation']['random_flip'] else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if params['data_augmentation']['normalize'] else transforms.Lambda(lambda x: x)
            ])
            
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if params['data_augmentation']['normalize'] else transforms.Lambda(lambda x: x)
            ])
            
            # 尝试加载CIFAR-10数据集
            try:
                train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
                test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
                
                train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)
                
                input_shape = [3, 32, 32]  # CIFAR-10形状
                num_classes = 10
                
            except Exception as e:
                progress_callback(0, 1, 0, 0, f"加载CIFAR-10数据集出错: {str(e)}")
                return
        else:
            progress_callback(0, 1, 0, 0, f"不支持的数据集: {params['dataset']}")
            return
        
        # 获取模型信息
        model_dict = self.get_model_dict()
        
        # 创建简化版的模型执行函数
        def create_model():
            class ModelNetwork(nn.Module):
                def __init__(self):
                    super(ModelNetwork, self).__init__()
                    
                    # 根据可视化模型创建层
                    self.layers = nn.ModuleDict()
                    
                    # 创建每个组件的层
                    for comp in model_dict["components"]:
                        comp_id = comp["id"]
                        comp_type = comp["type"]
                        config = comp.get("config", {})
                        
                        if comp_type == "conv2d":
                            in_channels = config.get("in_channels", 3)
                            out_channels = config.get("out_channels", 16)
                            kernel_size = config.get("kernel_size", 3)
                            stride = config.get("stride", 1)
                            padding = config.get("padding", 0)
                            self.layers[f"conv_{comp_id}"] = nn.Conv2d(in_channels, out_channels, 
                                                                     kernel_size=kernel_size, 
                                                                     stride=stride, padding=padding)
                        
                        elif comp_type == "maxpool":
                            kernel_size = config.get("kernel_size", 2)
                            stride = config.get("stride", 2)
                            self.layers[f"pool_{comp_id}"] = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
                        
                        elif comp_type == "linear":
                            in_features = config.get("in_features", 64)
                            out_features = config.get("out_features", 10)
                            self.layers[f"fc_{comp_id}"] = nn.Linear(in_features, out_features)
                        
                        elif comp_type == "dropout":
                            p = config.get("p", 0.5)
                            self.layers[f"dropout_{comp_id}"] = nn.Dropout(p=p)
                        
                        elif comp_type == "batchnorm":
                            num_features = config.get("num_features", 16)
                            self.layers[f"bn_{comp_id}"] = nn.BatchNorm2d(num_features)
                        
                        elif comp_type == "flatten":
                            self.layers[f"flatten_{comp_id}"] = nn.Flatten()
                        
                        elif comp_type == "relu":
                            self.layers[f"relu_{comp_id}"] = nn.ReLU(inplace=True)
                
                def forward(self, x):
                    # 存储中间结果
                    activations = {"input": x}
                    
                    # 构建连接图
                    graph = {comp["id"]: [] for comp in model_dict["components"]}
                    # 反向图用于查找输入连接
                    reverse_graph = {comp["id"]: [] for comp in model_dict["components"]}
                    
                    for conn in model_dict["connections"]:
                        from_id = conn["from"]
                        to_id = conn["to"]
                        graph[from_id].append(to_id)
                        reverse_graph[to_id].append(from_id)
                    
                    # 找到输入组件
                    input_comp = next((comp for comp in model_dict["components"] if comp["type"] == "input"), None)
                    if not input_comp:
                        raise ValueError("模型缺少输入层")
                    
                    # 为输入组件设置初始激活值
                    activations[input_comp["id"]] = x
                    
                    # 查找所有需要处理的组件
                    all_components = {comp["id"]: comp for comp in model_dict["components"]}
                    
                    # 执行拓扑排序
                    processed = set([input_comp["id"]])  # 已处理的组件
                    queue = []
                    
                    # 首先将所有可以立即处理的组件加入队列（那些输入只来自输入层的组件）
                    for comp_id, comp in all_components.items():
                        if comp_id != input_comp["id"] and all(pred_id in processed for pred_id in reverse_graph.get(comp_id, [])):
                            queue.append(comp_id)
                    
                    # 处理所有组件
                    while queue:
                        current_id = queue.pop(0)
                        
                        # 如果已经处理过，跳过
                        if current_id in processed:
                            continue
                            
                        # 检查所有输入是否都已处理
                        if not all(pred_id in processed and pred_id in activations for pred_id in reverse_graph.get(current_id, [])):
                            # 将组件放回队列末尾
                            queue.append(current_id)
                            continue
                        
                        current_comp = all_components[current_id]
                        
                        # 处理特定类型的组件
                        if current_comp["type"] == "output":
                            # 输出层直接使用其输入的激活值
                            input_ids = reverse_graph[current_id]
                            if input_ids:
                                activations[current_id] = activations[input_ids[0]]
                        else:
                            # 对于其他组件，应用相应的层
                            input_ids = reverse_graph[current_id]
                            if input_ids:
                                # 确保所有输入已处理
                                if all(input_id in activations for input_id in input_ids):
                                    # 使用第一个输入（简化模型，实际上可能需要处理多输入）
                                    input_activations = activations[input_ids[0]]
                                    
                                    # 查找对应的层
                                    if current_comp["type"] == "relu":
                                        layer_key = f"relu_{current_id}"
                                    else:
                                        layer_key = f"{current_comp['type']}_{current_id}"
                                    
                                    # 尝试查找匹配的层名称
                                    if layer_key not in self.layers:
                                        layer_key = next((k for k in self.layers.keys() if current_id in k), None)
                                    
                                    # 应用层处理
                                    if layer_key and layer_key in self.layers:
                                        try:
                                            activations[current_id] = self.layers[layer_key](input_activations)
                                        except Exception as e:
                                            print(f"处理组件 {current_id} ({current_comp['type']}) 时出错: {str(e)}")
                                            # 防止错误导致整个前向传播失败，使用输入作为输出
                                            activations[current_id] = input_activations
                        
                        # 标记为已处理
                        processed.add(current_id)
                        
                        # 将所有输入已处理的后续组件加入队列
                        for next_id in graph.get(current_id, []):
                            # 检查所有输入是否都已处理
                            if next_id not in processed and all(pred_id in processed for pred_id in reverse_graph.get(next_id, [])):
                                queue.append(next_id)
                    
                    # 找到输出组件
                    output_comp = next((comp for comp in model_dict["components"] if comp["type"] == "output"), None)
                    if not output_comp:
                        raise ValueError("模型缺少输出层")
                    
                    # 确保输出组件有激活值
                    if output_comp["id"] not in activations:
                        # 尝试从连接到输出组件的所有节点获取激活值
                        input_to_output = reverse_graph[output_comp["id"]]
                        if not input_to_output:
                            raise ValueError("没有组件连接到输出层")
                        
                        # 检查所有输入到输出的组件ID是否都有激活值
                        valid_inputs = [input_id for input_id in input_to_output if input_id in activations]
                        if not valid_inputs:
                            # 打印诊断信息
                            print(f"警告：无法找到连接到输出层的组件激活值。已处理组件：{processed}")
                            print(f"所有激活值：{list(activations.keys())}")
                            print(f"输入到输出的组件：{input_to_output}")
                            
                            # 作为备选，使用任何可用的激活值
                            if activations:
                                # 选择最后处理的非输入激活值
                                for comp_id in reversed(list(processed)):
                                    if comp_id in activations and comp_id != input_comp["id"]:
                                        return activations[comp_id]
                                
                                # 如果没有找到，返回输入
                                return activations[input_comp["id"]]
                            
                            raise ValueError("无法找到任何有效的激活值返回")
                        
                        # 使用第一个有效的连接
                        return activations[valid_inputs[0]]
                    
                    # 返回输出组件的激活值
                    return activations[output_comp["id"]]
            
            return ModelNetwork()
        
        try:
            # 创建模型实例
            model = create_model()
            model = model.to(device)
            
            # 损失函数和优化器
            criterion = nn.CrossEntropyLoss()
            
            # 根据用户选择设置优化器
            if params["optimizer"] == "Adam":
                optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
            elif params["optimizer"] == "SGD":
                optimizer = optim.SGD(model.parameters(), lr=params["learning_rate"], momentum=0.9)
            elif params["optimizer"] == "RMSprop":
                optimizer = optim.RMSprop(model.parameters(), lr=params["learning_rate"])
            elif params["optimizer"] == "AdamW":
                optimizer = optim.AdamW(model.parameters(), lr=params["learning_rate"])
            else:
                optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
            
            # 初始化进度
            best_accuracy = 0
            
            # 开始训练
            for epoch in range(1, params["epochs"] + 1):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                
                progress_callback(epoch, params["epochs"], 0, 0, f"开始第 {epoch} 轮训练...")
                
                # 训练一个epoch
                for i, data in enumerate(train_loader, 0):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    if i % 10 == 9:  # 每10个batch更新一次进度
                        avg_loss = running_loss / 10
                        accuracy = 100 * correct / total
                        progress_callback(epoch, params["epochs"], avg_loss, accuracy, 
                                          f"批次 {i+1}/{len(train_loader)}")
                        running_loss = 0.0
                
                # 在测试集上评估
                model.eval()
                test_loss = 0
                correct = 0
                total = 0
                with torch.no_grad():
                    for data in test_loader:
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        
                        test_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                
                accuracy = 100 * correct / total
                
                # 更新最佳精度
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    if params["save_best"]:
                        torch.save(model.state_dict(), f'./best_model_{params["dataset"]}.pth')
                        progress_callback(epoch, params["epochs"], test_loss/len(test_loader), accuracy,
                                         f"保存最佳模型，准确率: {accuracy:.2f}%")
                
                # 报告进度
                progress_callback(epoch, params["epochs"], test_loss/len(test_loader), accuracy,
                                  f"测试集准确率: {accuracy:.2f}%")
            
            # 训练完成
            progress_callback(params["epochs"], params["epochs"], test_loss/len(test_loader), best_accuracy,
                              f"训练完成！最佳准确率: {best_accuracy:.2f}%")
            
            return model
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            progress_callback(0, params["epochs"], 0, 0, f"训练过程中出错: {str(e)}\n{error_msg}")
            return None
    
    def infer_model(self, params, progress_callback):
        """执行模型推理
        
        Args:
            params: 推理参数字典
            progress_callback: 进度回调函数，格式为 callback(progress_percent, results, message)
        """
        try:
            import torch
            import torch.nn as nn
            import torchvision
            import torchvision.transforms as transforms
            import numpy as np
            import matplotlib.pyplot as plt
            from torch.utils.data import DataLoader
        except ImportError:
            progress_callback(0, "错误: 缺少必要的库", "请安装PyTorch和torchvision")
            return
        
        # 设置设备（GPU/CPU）
        device = torch.device("cuda" if params["use_gpu"] and torch.cuda.is_available() else "cpu")
        
        # 准备数据集
        if params["dataset"] == "mnist":
            progress_callback(10, None, "正在加载MNIST数据集...")
            
            # 转换
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            # 尝试加载MNIST数据集
            try:
                test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
                test_loader = DataLoader(test_dataset, batch_size=64 if params["batch_inference"] else 1, shuffle=False)
                num_classes = 10
                
            except Exception as e:
                progress_callback(0, None, f"加载MNIST数据集出错: {str(e)}")
                return
                
        elif params["dataset"] == "cifar10":
            progress_callback(10, None, "正在加载CIFAR-10数据集...")
            
            # 转换
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            # 尝试加载CIFAR-10数据集
            try:
                test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
                test_loader = DataLoader(test_dataset, batch_size=64 if params["batch_inference"] else 1, shuffle=False)
                num_classes = 10
                
            except Exception as e:
                progress_callback(0, None, f"加载CIFAR-10数据集出错: {str(e)}")
                return
        else:
            progress_callback(0, None, f"不支持的数据集: {params['dataset']}")
            return
        
        # 获取类别名称
        if params["dataset"] == "cifar10":
            classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        else:  # MNIST
            classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        
        # 获取模型信息
        model_dict = self.get_model_dict()
        
        # 创建简化版的模型执行函数
        def create_model():
            class ModelNetwork(nn.Module):
                def __init__(self):
                    super(ModelNetwork, self).__init__()
                    
                    # 根据可视化模型创建层
                    self.layers = nn.ModuleDict()
                    
                    # 创建每个组件的层
                    for comp in model_dict["components"]:
                        comp_id = comp["id"]
                        comp_type = comp["type"]
                        config = comp.get("config", {})
                        
                        if comp_type == "conv2d":
                            in_channels = config.get("in_channels", 3)
                            out_channels = config.get("out_channels", 16)
                            kernel_size = config.get("kernel_size", 3)
                            stride = config.get("stride", 1)
                            padding = config.get("padding", 0)
                            self.layers[f"conv_{comp_id}"] = nn.Conv2d(in_channels, out_channels, 
                                                                     kernel_size=kernel_size, 
                                                                     stride=stride, padding=padding)
                        
                        elif comp_type == "maxpool":
                            kernel_size = config.get("kernel_size", 2)
                            stride = config.get("stride", 2)
                            self.layers[f"pool_{comp_id}"] = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
                        
                        elif comp_type == "linear":
                            in_features = config.get("in_features", 64)
                            out_features = config.get("out_features", 10)
                            self.layers[f"fc_{comp_id}"] = nn.Linear(in_features, out_features)
                        
                        elif comp_type == "dropout":
                            p = config.get("p", 0.5)
                            self.layers[f"dropout_{comp_id}"] = nn.Dropout(p=p)
                        
                        elif comp_type == "batchnorm":
                            num_features = config.get("num_features", 16)
                            self.layers[f"bn_{comp_id}"] = nn.BatchNorm2d(num_features)
                        
                        elif comp_type == "flatten":
                            self.layers[f"flatten_{comp_id}"] = nn.Flatten()
                        
                        elif comp_type == "relu":
                            self.layers[f"relu_{comp_id}"] = nn.ReLU(inplace=True)
                
                def forward(self, x):
                    # 存储中间结果
                    activations = {"input": x}
                    
                    # 构建连接图
                    graph = {comp["id"]: [] for comp in model_dict["components"]}
                    # 反向图用于查找输入连接
                    reverse_graph = {comp["id"]: [] for comp in model_dict["components"]}
                    
                    for conn in model_dict["connections"]:
                        from_id = conn["from"]
                        to_id = conn["to"]
                        graph[from_id].append(to_id)
                        reverse_graph[to_id].append(from_id)
                    
                    # 找到输入组件
                    input_comp = next((comp for comp in model_dict["components"] if comp["type"] == "input"), None)
                    if not input_comp:
                        raise ValueError("模型缺少输入层")
                    
                    # 为输入组件设置初始激活值
                    activations[input_comp["id"]] = x
                    
                    # 查找所有需要处理的组件
                    all_components = {comp["id"]: comp for comp in model_dict["components"]}
                    
                    # 执行拓扑排序
                    processed = set([input_comp["id"]])  # 已处理的组件
                    queue = []
                    
                    # 首先将所有可以立即处理的组件加入队列（那些输入只来自输入层的组件）
                    for comp_id, comp in all_components.items():
                        if comp_id != input_comp["id"] and all(pred_id in processed for pred_id in reverse_graph.get(comp_id, [])):
                            queue.append(comp_id)
                    
                    # 处理所有组件
                    while queue:
                        current_id = queue.pop(0)
                        
                        # 如果已经处理过，跳过
                        if current_id in processed:
                            continue
                            
                        # 检查所有输入是否都已处理
                        if not all(pred_id in processed and pred_id in activations for pred_id in reverse_graph.get(current_id, [])):
                            # 将组件放回队列末尾
                            queue.append(current_id)
                            continue
                        
                        current_comp = all_components[current_id]
                        
                        # 处理特定类型的组件
                        if current_comp["type"] == "output":
                            # 输出层直接使用其输入的激活值
                            input_ids = reverse_graph[current_id]
                            if input_ids:
                                activations[current_id] = activations[input_ids[0]]
                        else:
                            # 对于其他组件，应用相应的层
                            input_ids = reverse_graph[current_id]
                            if input_ids:
                                # 确保所有输入已处理
                                if all(input_id in activations for input_id in input_ids):
                                    # 使用第一个输入（简化模型，实际上可能需要处理多输入）
                                    input_activations = activations[input_ids[0]]
                                    
                                    # 查找对应的层
                                    if current_comp["type"] == "relu":
                                        layer_key = f"relu_{current_id}"
                                    else:
                                        layer_key = f"{current_comp['type']}_{current_id}"
                                    
                                    # 尝试查找匹配的层名称
                                    if layer_key not in self.layers:
                                        layer_key = next((k for k in self.layers.keys() if current_id in k), None)
                                    
                                    # 应用层处理
                                    if layer_key and layer_key in self.layers:
                                        try:
                                            activations[current_id] = self.layers[layer_key](input_activations)
                                        except Exception as e:
                                            print(f"处理组件 {current_id} ({current_comp['type']}) 时出错: {str(e)}")
                                            # 防止错误导致整个前向传播失败，使用输入作为输出
                                            activations[current_id] = input_activations
                        
                        # 标记为已处理
                        processed.add(current_id)
                        
                        # 将所有输入已处理的后续组件加入队列
                        for next_id in graph.get(current_id, []):
                            # 检查所有输入是否都已处理
                            if next_id not in processed and all(pred_id in processed for pred_id in reverse_graph.get(next_id, [])):
                                queue.append(next_id)
                    
                    # 找到输出组件
                    output_comp = next((comp for comp in model_dict["components"] if comp["type"] == "output"), None)
                    if not output_comp:
                        raise ValueError("模型缺少输出层")
                    
                    # 确保输出组件有激活值
                    if output_comp["id"] not in activations:
                        # 尝试从连接到输出组件的所有节点获取激活值
                        input_to_output = reverse_graph[output_comp["id"]]
                        if not input_to_output:
                            raise ValueError("没有组件连接到输出层")
                        
                        # 检查所有输入到输出的组件ID是否都有激活值
                        valid_inputs = [input_id for input_id in input_to_output if input_id in activations]
                        if not valid_inputs:
                            # 打印诊断信息
                            print(f"警告：无法找到连接到输出层的组件激活值。已处理组件：{processed}")
                            print(f"所有激活值：{list(activations.keys())}")
                            print(f"输入到输出的组件：{input_to_output}")
                            
                            # 作为备选，使用任何可用的激活值
                            if activations:
                                # 选择最后处理的非输入激活值
                                for comp_id in reversed(list(processed)):
                                    if comp_id in activations and comp_id != input_comp["id"]:
                                        return activations[comp_id]
                                
                                # 如果没有找到，返回输入
                                return activations[input_comp["id"]]
                            
                            raise ValueError("无法找到任何有效的激活值返回")
                        
                        # 使用第一个有效的连接
                        return activations[valid_inputs[0]]
                    
                    # 返回输出组件的激活值
                    return activations[output_comp["id"]]
            
            return ModelNetwork()
        
        try:
            # 创建模型实例
            model = create_model()
            
            # 尝试加载之前训练的最佳模型
            model_path = f'./best_model_{params["dataset"]}.pth'
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
                progress_callback(20, None, f"已加载预训练模型: {model_path}")
            else:
                progress_callback(20, None, "未找到预训练模型，将使用未训练的模型")
            
            model = model.to(device)
            model.eval()  # 设置为评估模式
            
            # 如果需要可视化结果
            if params["visualize"]:
                plt.figure(figsize=(12, 6))
            
            # 收集统计数据
            correct = 0
            total = 0
            class_correct = list(0. for i in range(num_classes))
            class_total = list(0. for i in range(num_classes))
            
            results_text = ""
            
            # 执行推理
            progress_step = 50 / len(test_loader)
            current_progress = 20
            
            # 使用数据集的一部分进行推理
            max_samples = 100
            sample_count = 0
            
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    if sample_count >= max_samples:
                        break
                        
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    sample_count += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    c = (predicted == labels).squeeze()
                    for j in range(labels.size(0)):
                        if j < len(c):  # 避免批量大小为1时的问题
                            label = labels[j]
                            class_correct[label] += c[j].item()
                            class_total[label] += 1
                    
                    # 可视化一些预测结果
                    if params["visualize"] and i < 8:
                        ax = plt.subplot(2, 4, i + 1)
                        img = images[0].cpu().numpy()
                        if img.shape[0] == 1:  # MNIST是单通道
                            img = img.reshape(img.shape[1], img.shape[2])
                            plt.imshow(img, cmap='gray')
                        else:  # CIFAR-10是彩色
                            img = np.transpose(img, (1, 2, 0))
                            # 反归一化
                            if params["dataset"] == "cifar10":
                                img = img * 0.5 + 0.5
                            plt.imshow(img)
                        
                        plt.title(f'预测: {classes[predicted[0]]}\n实际: {classes[labels[0]]}')
                        plt.axis('off')
                    
                    # 更新进度
                    current_progress += progress_step
                    progress_callback(min(70, int(current_progress)), None, f"已处理 {i+1}/{len(test_loader)} 批次")
            
            # 计算每个类别的准确率
            for i in range(num_classes):
                if class_total[i] > 0:
                    class_accuracy = 100 * class_correct[i] / class_total[i]
                    results_text += f"准确率 - {classes[i]}: {class_accuracy:.1f}%\n"
            
            # 计算总体准确率
            accuracy = 100 * correct / total
            results_text += f"\n总体准确率: {accuracy:.2f}%"
            
            # 保存可视化结果
            if params["visualize"]:
                plt.tight_layout()
                viz_path = f'./inference_results_{params["dataset"]}.png'
                plt.savefig(viz_path)
                plt.close()
                results_text += f"\n\n已保存可视化结果到: {viz_path}"
            
            # 完成推理
            progress_callback(100, results_text, "推理完成")
            
            return True
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            progress_callback(0, None, f"推理过程中出错: {str(e)}\n{error_msg}")
            return False