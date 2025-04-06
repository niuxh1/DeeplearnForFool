#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
组件调色板 - 提供可用的神经网络组件
"""

import sys
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QScrollArea,
                            QListWidget, QListWidgetItem, QAbstractItemView)
from PyQt5.QtCore import Qt, QSize, QMimeData, QByteArray
from PyQt5.QtGui import QDrag, QPixmap, QIcon

class ComponentPalette(QWidget):
    """组件调色板类，显示可拖拽的神经网络组件"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 标题
        title = QLabel("<h3>神经网络组件</h3>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 创建组件列表
        self.component_list = QListWidget()
        self.component_list.setDragEnabled(True)
        self.component_list.setViewMode(QListWidget.IconMode)
        self.component_list.setIconSize(QSize(64, 64))
        self.component_list.setSpacing(10)
        self.component_list.setAcceptDrops(False)
        self.component_list.setDropIndicatorShown(False)
        
        # 添加组件
        self.add_component("卷积层 (Conv2d)", "conv2d")
        self.add_component("池化层 (MaxPool2d)", "maxpool")
        self.add_component("全连接层 (Linear)", "linear")
        self.add_component("激活函数 (ReLU)", "relu")
        self.add_component("Dropout层", "dropout")
        self.add_component("批归一化 (BatchNorm)", "batchnorm")
        self.add_component("输入层", "input")
        self.add_component("输出层", "output")
        self.add_component("展平层 (Flatten)", "flatten")
        
        # 高级组件
        self.add_component("LSTM层", "lstm")
        self.add_component("GRU层", "gru")
        self.add_component("注意力层 (Attention)", "attention")
        self.add_component("残差连接 (Residual)", "residual")
        
        # 添加滚动区域
        scroll = QScrollArea()
        scroll.setWidget(self.component_list)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        # 帮助标签
        help_label = QLabel("拖拽组件到右侧画布创建神经网络")
        help_label.setWordWrap(True)
        help_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(help_label)
        
        # 连接信号
        self.component_list.itemPressed.connect(self.start_drag)
    
    def add_component(self, name, component_type):
        """添加一个组件到列表"""
        item = QListWidgetItem(name)
        item.setData(Qt.UserRole, component_type)
        # 这里应该有图标，但我们暂时用默认图标
        self.component_list.addItem(item)
    
    def start_drag(self, item):
        """开始拖拽操作"""
        # 获取组件类型
        component_type = item.data(Qt.UserRole)
        
        # 创建MIME数据
        mime_data = QMimeData()
        mime_data.setText(component_type)
        
        # 创建拖拽对象
        drag = QDrag(self)
        drag.setMimeData(mime_data)
        
        # 设置拖拽时的图标
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.white)
        drag.setPixmap(pixmap)
        
        # 开始拖拽
        drag.exec_(Qt.CopyAction | Qt.MoveAction)