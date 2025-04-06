#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主窗口类 - 用户界面的核心组件
"""

import os
import sys
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QAction, QToolBar, QDockWidget,
                             QListWidget, QSplitter, QFileDialog, QMessageBox,
                             QMenu, QStatusBar, QTabWidget, QComboBox, QFrame,
                             QProgressBar, QRadioButton, QButtonGroup, QSlider,
                             QGroupBox, QCheckBox, QStyleFactory, QApplication, QTextEdit, QFormLayout, QActionGroup)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QIcon, QDrag, QPixmap, QColor, QPalette, QFont

from components.component_palette import ComponentPalette
from components.model_canvas import ModelCanvas, ConnectionLine
from components.code_generator import CodeGenerator
from components.network_templates import TemplateManager


class MainWindow(QMainWindow):
    """主窗口类，提供整个应用程序的用户界面框架"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepLearn Playground - 深度学习探索乐园")
        self.setGeometry(100, 100, 1400, 900)  # 增大默认窗口尺寸
        
        # 设置应用主题和样式
        self.apply_modern_style()
        
        # 模型运行模式 (训练/推理)
        self.run_mode = "train"
        
        # GPU状态
        self.has_gpu = self.check_gpu_availability()
        self.use_gpu = self.has_gpu  # 默认使用GPU（如果可用）
        
        # 设置中心控件
        self.init_ui()
        
        # 添加菜单栏
        self.create_menubar()
        
        # 添加工具栏
        self.create_toolbar()
        
        # 添加状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        if self.has_gpu:
            self.statusBar.showMessage("就绪 | GPU加速模式已启用")
        else:
            self.statusBar.showMessage("就绪 | CPU模式")
        
        # 加载经典模型模板
        self.template_manager = TemplateManager()
        self.load_template_list()
    
    def apply_modern_style(self):
        """应用现代化样式"""
        # 设置Fusion风格，这是跨平台的现代UI风格
        QApplication.setStyle(QStyleFactory.create("Fusion"))
        
        # 创建自定义暗色主题调色板
        dark_palette = QPalette()
        
        # 设置背景色和文本色
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        
        # 应用暗色主题
        QApplication.setPalette(dark_palette)
        
        # 设置样式表
        style_sheet = """
        QToolTip { 
            color: #ffffff; 
            background-color: #2a82da; 
            border: 1px solid white;
            padding: 2px;
            border-radius: 3px;
            opacity: 200;
        }
        
        QTabWidget::pane {
            border: 1px solid #444;
            top: -1px;
            background: #353535;
        }
        
        QTabBar::tab {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, 
                                       stop: 0 #555, stop: 0.4 #444, 
                                       stop: 0.5 #333, stop: 1.0 #222);
            color: #ffffff;
            min-width: 8ex;
            padding: 8px 15px;
            border: 1px solid #333;
            border-bottom-color: #444;
        }
        
        QTabBar::tab:selected {
            background: #2a82da;
            border-color: #2a82da;
            border-bottom-color: #2a82da;
        }
        
        QPushButton {
            background-color: #2a82da;
            border-width: 0;
            border-radius: 4px;
            padding: 8px 16px;
            color: white;
            font-weight: bold;
        }
        
        QPushButton:hover {
            background-color: #3a92ea;
        }
        
        QPushButton:pressed {
            background-color: #1a72ca;
        }
        
        QComboBox {
            border: 1px solid #555;
            border-radius: 3px;
            padding: 5px;
            min-width: 6em;
        }
        
        QGroupBox {
            border: 1px solid #555;
            border-radius: 5px;
            margin-top: 1ex;
            padding-top: 10px;
            font-weight: bold;
        }
        
        QProgressBar {
            border: 1px solid #555;
            border-radius: 3px;
            text-align: center;
        }
        
        QProgressBar::chunk {
            background-color: #2a82da;
            width: 10px;
            margin: 0.5px;
        }
        """
        self.setStyleSheet(style_sheet)
    
    def check_gpu_availability(self):
        """检查是否有可用的GPU"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def load_template_list(self):
        """加载模板列表"""
        templates = self.template_manager.get_template_list()
        self.templates_list.clear()
        self.templates_list.addItems(templates)
    
    def init_ui(self):
        """初始化用户界面"""
        # 主布局
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # 添加运行配置面板（模式选择和GPU设置）
        run_config_panel = self.create_run_config_panel()
        main_layout.addWidget(run_config_panel)
        
        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧部件 - 组件面板
        self.component_palette = ComponentPalette()
        self.component_dock = QDockWidget("神经网络组件", self)
        self.component_dock.setWidget(self.component_palette)
        self.component_dock.setFeatures(QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.component_dock)
        
        # 中央部件 - 模型画布
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        
        # 添加画布标题
        canvas_title = QLabel("神经网络模型构建区")
        canvas_title.setAlignment(Qt.AlignCenter)
        canvas_title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")
        canvas_layout.addWidget(canvas_title)
        
        # 添加画布
        self.canvas = ModelCanvas()
        canvas_layout.addWidget(self.canvas)
        
        # 添加状态指示器
        self.canvas_status = QLabel("就绪")
        self.canvas_status.setAlignment(Qt.AlignRight)
        self.canvas_status.setStyleSheet("color: #2ecc71; padding: 5px;")
        canvas_layout.addWidget(self.canvas_status)
        
        splitter.addWidget(canvas_container)
        
        # 右侧控件 - 属性面板与代码生成器
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 属性与代码标签页
        self.right_tabs = QTabWidget()
        
        # 设置标签页
        self.settings_tab = self.create_settings_tab()
        self.right_tabs.addTab(self.settings_tab, "设置")
        
        # 模型运行标签页
        self.run_tab = self.create_run_tab()
        self.right_tabs.addTab(self.run_tab, "模型运行")
        
        # 代码生成面板
        self.code_generator = CodeGenerator()
        self.right_tabs.addTab(self.code_generator, "PyTorch 代码")
        
        # 模型模板面板
        templates_widget = QWidget()
        templates_layout = QVBoxLayout(templates_widget)
        
        templates_label = QLabel("选择预设模型模板:")
        templates_layout.addWidget(templates_label)
        
        self.templates_list = QListWidget()
        self.templates_list.itemDoubleClicked.connect(self.load_template)
        templates_layout.addWidget(self.templates_list)
        
        load_template_btn = QPushButton("加载选中模板")
        load_template_btn.clicked.connect(lambda: self.load_template(self.templates_list.currentItem()) 
                                         if self.templates_list.currentItem() else None)
        templates_layout.addWidget(load_template_btn)
        
        self.right_tabs.addTab(templates_widget, "经典模型")
        
        # 帮助面板
        help_widget = QWidget()
        help_layout = QVBoxLayout(help_widget)
        help_text = """
        <h3>欢迎使用深度学习探索乐园!</h3>
        <p>这是一款面向儿童和初学者的深度学习可视化教学软件</p>
        <p><b>基本操作:</b></p>
        <ul>
            <li>从左侧组件面板拖拽组件到中央画布</li>
            <li>双击组件可以编辑其属性</li>
            <li>连接组件创建神经网络</li>
            <li>在"模型运行"标签页中设置训练/推理模式</li>
            <li>点击"生成代码"按钮查看PyTorch代码</li>
            <li>在"经典模型"标签页可以加载预设的网络模型</li>
        </ul>
        """
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        help_layout.addWidget(help_label)
        self.right_tabs.addTab(help_widget, "使用帮助")
        
        right_layout.addWidget(self.right_tabs)
        
        # 操作按钮区域
        buttons_group = QGroupBox("操作")
        buttons_layout = QHBoxLayout(buttons_group)
        
        # 验证模型按钮
        validate_button = QPushButton("验证模型")
        validate_button.clicked.connect(self.validate_model)
        buttons_layout.addWidget(validate_button)
        
        # 生成代码按钮
        generate_button = QPushButton("生成代码")
        generate_button.clicked.connect(self.generate_code)
        buttons_layout.addWidget(generate_button)
        
        # 运行模型按钮
        run_button = QPushButton("运行模型")
        run_button.clicked.connect(self.run_model)
        run_button.setStyleSheet("background-color: #27ae60;") # 绿色按钮
        buttons_layout.addWidget(run_button)
        
        right_layout.addWidget(buttons_group)
        
        # 文件操作按钮
        file_group = QGroupBox("文件操作")
        file_layout = QHBoxLayout(file_group)
        
        save_button = QPushButton("保存模型")
        save_button.clicked.connect(self.save_model)
        file_layout.addWidget(save_button)
        
        load_button = QPushButton("加载模型")
        load_button.clicked.connect(self.load_model)
        file_layout.addWidget(load_button)
        
        export_button = QPushButton("导出代码")
        export_button.clicked.connect(self.export_code)
        file_layout.addWidget(export_button)
        
        right_layout.addWidget(file_group)
        
        splitter.addWidget(right_widget)
        
        # 设置分割器比例
        splitter.setSizes([700, 500])
        main_layout.addWidget(splitter)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        self.setCentralWidget(main_widget)
    
    def create_run_config_panel(self):
        """创建运行配置面板（含模式选择和GPU设置）"""
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setStyleSheet("background-color: #2c3e50; border-radius: 5px; padding: 5px;")
        
        layout = QHBoxLayout(panel)
        
        # 模式选择部分
        mode_group = QGroupBox("运行模式")
        mode_group.setStyleSheet("color: white;")
        mode_layout = QHBoxLayout(mode_group)
        
        self.train_mode_radio = QRadioButton("训练模式")
        self.train_mode_radio.setChecked(True)
        self.train_mode_radio.setStyleSheet("color: white;")
        self.train_mode_radio.toggled.connect(self.set_train_mode)
        
        self.inference_mode_radio = QRadioButton("推理模式")
        self.inference_mode_radio.setStyleSheet("color: white;")
        self.inference_mode_radio.toggled.connect(self.set_inference_mode)
        
        mode_layout.addWidget(self.train_mode_radio)
        mode_layout.addWidget(self.inference_mode_radio)
        
        layout.addWidget(mode_group)
        
        # GPU设置部分
        if self.has_gpu:
            gpu_group = QGroupBox("GPU加速")
            gpu_group.setStyleSheet("color: white;")
            gpu_layout = QHBoxLayout(gpu_group)
            
            self.use_gpu_checkbox = QCheckBox("启用GPU加速")
            self.use_gpu_checkbox.setChecked(self.use_gpu)
            self.use_gpu_checkbox.setStyleSheet("color: white;")
            self.use_gpu_checkbox.toggled.connect(self.toggle_gpu)
            
            gpu_layout.addWidget(self.use_gpu_checkbox)
            
            # 显示GPU信息
            import torch
            if torch.cuda.is_available():
                gpu_info = QLabel(f"GPU: {torch.cuda.get_device_name(0)}")
                gpu_info.setStyleSheet("color: #2ecc71;")  # 绿色文字
                gpu_layout.addWidget(gpu_info)
            
            layout.addWidget(gpu_group)
        else:
            gpu_info = QLabel("CPU模式 (未检测到可用GPU)")
            gpu_info.setStyleSheet("color: #e74c3c; background-color: #2c3e50; padding: 10px; border-radius: 5px;")
            layout.addWidget(gpu_info)
        
        # 数据集选择
        dataset_group = QGroupBox("数据集")
        dataset_group.setStyleSheet("color: white;")
        dataset_layout = QHBoxLayout(dataset_group)
        
        dataset_label = QLabel("选择数据集:")
        dataset_label.setStyleSheet("color: white;")
        dataset_layout.addWidget(dataset_label)
        
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["未选择", "MNIST", "CIFAR-10", "ImageNet"])
        self.dataset_combo.currentTextChanged.connect(self.on_dataset_changed)
        dataset_layout.addWidget(self.dataset_combo)
        
        layout.addWidget(dataset_group)
        
        return panel
    
    def create_settings_tab(self):
        """创建设置标签页"""
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        
        # 训练设置
        training_group = QGroupBox("训练设置")
        training_layout = QFormLayout(training_group)
        
        self.epochs_slider = QSlider(Qt.Horizontal)
        self.epochs_slider.setMinimum(1)
        self.epochs_slider.setMaximum(100)
        self.epochs_slider.setValue(10)
        self.epochs_label = QLabel("轮数(Epochs): 10")
        self.epochs_slider.valueChanged.connect(lambda v: self.epochs_label.setText(f"轮数(Epochs): {v}"))
        training_layout.addRow(self.epochs_label, self.epochs_slider)
        
        self.batch_size_combo = QComboBox()
        self.batch_size_combo.addItems(["16", "32", "64", "128", "256"])
        self.batch_size_combo.setCurrentIndex(2)  # 默认64
        training_layout.addRow("批次大小(Batch Size):", self.batch_size_combo)
        
        self.lr_combo = QComboBox()
        self.lr_combo.addItems(["0.0001", "0.001", "0.01", "0.1"])
        self.lr_combo.setCurrentIndex(1)  # 默认0.001
        training_layout.addRow("学习率(Learning Rate):", self.lr_combo)
        
        settings_layout.addWidget(training_group)
        
        # 优化器设置
        optimizer_group = QGroupBox("优化器")
        optimizer_layout = QFormLayout(optimizer_group)
        
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["SGD", "Adam", "RMSprop", "AdamW"])
        self.optimizer_combo.setCurrentIndex(1)  # 默认Adam
        optimizer_layout.addRow("选择优化器:", self.optimizer_combo)
        
        settings_layout.addWidget(optimizer_group)
        
        # 数据增强
        augment_group = QGroupBox("数据增强")
        augment_layout = QVBoxLayout(augment_group)
        
        self.random_flip_check = QCheckBox("随机翻转")
        self.random_flip_check.setChecked(True)
        augment_layout.addWidget(self.random_flip_check)
        
        self.random_crop_check = QCheckBox("随机裁剪")
        self.random_crop_check.setChecked(True)
        augment_layout.addWidget(self.random_crop_check)
        
        self.normalize_check = QCheckBox("标准化")
        self.normalize_check.setChecked(True)
        augment_layout.addWidget(self.normalize_check)
        
        settings_layout.addWidget(augment_group)
        
        # 添加空白区域
        settings_layout.addStretch()
        
        return settings_widget
    
    def create_run_tab(self):
        """创建模型运行标签页"""
        run_widget = QWidget()
        run_layout = QVBoxLayout(run_widget)
        
        # 训练/推理模式设置
        mode_group = QGroupBox("运行模式设置")
        mode_layout = QVBoxLayout(mode_group)
        
        # 训练模式选项
        self.train_options = QWidget()
        train_options_layout = QFormLayout(self.train_options)
        
        self.epochs_input = QComboBox()
        self.epochs_input.addItems(["1", "5", "10", "20", "50", "100"])
        self.epochs_input.setCurrentIndex(2)  # 默认10
        train_options_layout.addRow("训练轮数:", self.epochs_input)
        
        self.save_best_check = QCheckBox("保存最佳模型")
        self.save_best_check.setChecked(True)
        train_options_layout.addRow("", self.save_best_check)
        
        # 推理模式选项
        self.inference_options = QWidget()
        inference_options_layout = QFormLayout(self.inference_options)
        
        self.batch_inference_check = QCheckBox("批量推理")
        inference_options_layout.addRow("", self.batch_inference_check)
        
        self.visualize_check = QCheckBox("可视化预测结果")
        self.visualize_check.setChecked(True)
        inference_options_layout.addRow("", self.visualize_check)
        
        # 初始隐藏推理选项
        self.inference_options.setVisible(False)
        
        mode_layout.addWidget(self.train_options)
        mode_layout.addWidget(self.inference_options)
        
        run_layout.addWidget(mode_group)
        
        # 训练进度
        progress_group = QGroupBox("运行状态")
        progress_layout = QVBoxLayout(progress_group)
        
        self.run_status_label = QLabel("就绪")
        self.run_status_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.run_status_label)
        
        self.run_progress = QProgressBar()
        self.run_progress.setValue(0)
        progress_layout.addWidget(self.run_progress)
        
        self.cancel_button = QPushButton("取消运行")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_run)
        progress_layout.addWidget(self.cancel_button)
        
        run_layout.addWidget(progress_group)
        
        # 结果显示区域
        results_group = QGroupBox("结果")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("运行模型后，结果将显示在此处...")
        results_layout.addWidget(self.results_text)
        
        run_layout.addWidget(results_group)
        
        return run_widget
    
    def create_menubar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        
        # 新建模型
        new_action = QAction("新建模型", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_model)
        file_menu.addAction(new_action)
        
        # 打开模型
        open_action = QAction("打开模型", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.load_model)
        file_menu.addAction(open_action)
        
        # 保存模型
        save_action = QAction("保存模型", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_model)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # 导出代码
        export_action = QAction("导出PyTorch代码", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_code)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        # 退出
        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 模型菜单
        model_menu = menubar.addMenu("模型")
        
        # 验证模型
        validate_action = QAction("验证模型", self)
        validate_action.triggered.connect(self.validate_model)
        model_menu.addAction(validate_action)
        
        # 生成代码
        generate_action = QAction("生成代码", self)
        generate_action.triggered.connect(self.generate_code)
        model_menu.addAction(generate_action)
        
        # 运行模式
        model_menu.addSeparator()
        train_mode_action = QAction("训练模式", self)
        train_mode_action.setCheckable(True)
        train_mode_action.setChecked(self.run_mode == "train")
        train_mode_action.triggered.connect(lambda: self.train_mode_radio.setChecked(True))
        
        inference_mode_action = QAction("推理模式", self)
        inference_mode_action.setCheckable(True)
        inference_mode_action.setChecked(self.run_mode == "inference")
        inference_mode_action.triggered.connect(lambda: self.inference_mode_radio.setChecked(True))
        
        mode_group = QActionGroup(self)
        mode_group.addAction(train_mode_action)
        mode_group.addAction(inference_mode_action)
        mode_group.setExclusive(True)
        
        model_menu.addAction(train_mode_action)
        model_menu.addAction(inference_mode_action)
        
        # 数据集菜单
        dataset_menu = menubar.addMenu("数据集")
        
        # 加载示例数据集
        load_mnist_action = QAction("加载MNIST数据集", self)
        load_mnist_action.triggered.connect(lambda: self.load_dataset("mnist"))
        dataset_menu.addAction(load_mnist_action)
        
        load_cifar_action = QAction("加载CIFAR-10数据集", self)
        load_cifar_action.triggered.connect(lambda: self.load_dataset("cifar10"))
        dataset_menu.addAction(load_cifar_action)
        
        # 设置菜单
        settings_menu = menubar.addMenu("设置")
        
        # GPU设置
        if self.has_gpu:
            gpu_action = QAction("使用GPU加速", self)
            gpu_action.setCheckable(True)
            gpu_action.setChecked(self.use_gpu)
            gpu_action.triggered.connect(lambda checked: self.use_gpu_checkbox.setChecked(checked))
            settings_menu.addAction(gpu_action)
        
        # 主题设置
        theme_menu = settings_menu.addMenu("主题")
        dark_theme_action = QAction("暗色主题", self)
        dark_theme_action.setCheckable(True)
        dark_theme_action.setChecked(True)
        dark_theme_action.triggered.connect(self.apply_modern_style)
        theme_menu.addAction(dark_theme_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        
        # 关于
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        # 教程
        tutorial_action = QAction("教程", self)
        tutorial_action.triggered.connect(self.show_tutorial)
        help_menu.addAction(tutorial_action)
    
    def create_toolbar(self):
        """创建工具栏"""
        toolbar = QToolBar("主工具栏")
        toolbar.setIconSize(QSize(32, 32))
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.addToolBar(toolbar)
        
        # 新建模型按钮
        new_action = QAction("新建模型", self)
        new_action.triggered.connect(self.new_model)
        toolbar.addAction(new_action)
        
        # 打开模型按钮
        open_action = QAction("打开模型", self)
        open_action.triggered.connect(self.load_model)
        toolbar.addAction(open_action)
        
        # 保存模型按钮
        save_action = QAction("保存模型", self)
        save_action.triggered.connect(self.save_model)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        # 验证模型按钮
        validate_action = QAction("验证模型", self)
        validate_action.triggered.connect(self.validate_model)
        toolbar.addAction(validate_action)
        
        # 生成代码按钮
        generate_action = QAction("生成代码", self)
        generate_action.triggered.connect(self.generate_code)
        toolbar.addAction(generate_action)
        
        toolbar.addSeparator()
        
        # 切换模式按钮组
        mode_label = QLabel("模式:")
        toolbar.addWidget(mode_label)
        
        train_action = QAction("训练", self)
        train_action.setCheckable(True)
        train_action.setChecked(self.run_mode == "train")
        train_action.triggered.connect(lambda: self.train_mode_radio.setChecked(True))
        
        inference_action = QAction("推理", self)
        inference_action.setCheckable(True)
        inference_action.setChecked(self.run_mode == "inference")
        inference_action.triggered.connect(lambda: self.inference_mode_radio.setChecked(True))
        
        toolbar.addAction(train_action)
        toolbar.addAction(inference_action)
        
        # 添加操作组
        toolbar_action_group = QActionGroup(self)
        toolbar_action_group.addAction(train_action)
        toolbar_action_group.addAction(inference_action)
        toolbar_action_group.setExclusive(True)
        
        toolbar.addSeparator()
        
        # 运行模型按钮
        run_action = QAction("运行模型", self)
        run_action.triggered.connect(self.run_model)
        toolbar.addAction(run_action)
    
    def set_train_mode(self, checked):
        """设置为训练模式"""
        if checked:
            self.run_mode = "train"
            if hasattr(self, 'train_options') and hasattr(self, 'inference_options'):
                self.train_options.setVisible(True)
                self.inference_options.setVisible(False)
            self.statusBar.showMessage(f"已切换到训练模式" + (" | GPU加速已启用" if self.use_gpu and self.has_gpu else ""))
    
    def set_inference_mode(self, checked):
        """设置为推理模式"""
        if checked:
            self.run_mode = "inference"
            if hasattr(self, 'train_options') and hasattr(self, 'inference_options'):
                self.train_options.setVisible(False)
                self.inference_options.setVisible(True)
            self.statusBar.showMessage(f"已切换到推理模式" + (" | GPU加速已启用" if self.use_gpu and self.has_gpu else ""))
    
    def toggle_gpu(self, checked):
        """切换GPU使用状态"""
        self.use_gpu = checked
        status = "就绪 | "
        status += "GPU加速已启用" if self.use_gpu else "CPU模式"
        self.statusBar.showMessage(status)
    
    def on_dataset_changed(self, dataset_name):
        """数据集选择变更处理"""
        if dataset_name == "未选择":
            return
            
        # 将界面选择的数据集名称映射到内部使用的名称
        dataset_map = {
            "MNIST": "mnist",
            "CIFAR-10": "cifar10",
            "ImageNet": "imagenet"
        }
        
        if dataset_name in dataset_map:
            self.load_dataset(dataset_map[dataset_name])
    
    def cancel_run(self):
        """取消当前运行的模型"""
        # 这里应该中断模型训练/推理过程
        self.run_status_label.setText("已取消")
        self.run_progress.setValue(0)
        self.cancel_button.setEnabled(False)
        self.statusBar.showMessage("模型运行已取消")
        
    def generate_code(self):
        """生成PyTorch代码（只用于后台运行，不显示代码）"""
        # 确保获取最新的模型数据
        model_dict = self.canvas.get_model_dict()
        
        # 添加GPU使用标志
        model_dict["use_gpu"] = self.use_gpu and self.has_gpu
        
        # 添加运行模式
        model_dict["run_mode"] = self.run_mode
        
        # 生成代码但不显示
        code = self.code_generator.generate_code(model_dict)
        
        # 保存代码以便运行时使用，但不在界面上显示
        self.last_generated_code = code
        
        # 通知用户模型已准备好
        self.statusBar.showMessage("模型已准备就绪，可以运行")
        print(f"代码已生成: 组件数={len(model_dict['components'])}, 连接数={len(model_dict['connections'])}")
        
        return code
    
    def run_model(self):
        """运行模型（训练或推理）"""
        # 先检查数据集是否已加载
        if not self.canvas.has_dataset():
            # 使用正确的QMessageBox.StandardButtons而不是字符串
            reply = QMessageBox.question(
                self, "未加载数据集", 
                "您尚未加载数据集。要继续运行模型，请选择一个数据集:",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Yes:  # 对应MNIST
                self.load_dataset("mnist")
                self.dataset_combo.setCurrentText("MNIST")
            elif reply == QMessageBox.No:  # 对应CIFAR-10
                self.load_dataset("cifar10")
                self.dataset_combo.setCurrentText("CIFAR-10")
            else:
                return  # 用户取消
        
        # 验证模型结构
        is_valid, message = self.canvas.validate_model()
        if not is_valid:
            QMessageBox.warning(self, "验证失败", f"模型结构有问题，无法运行:\n{message}")
            return
        
        # 准备运行参数
        run_params = self.prepare_run_params()
        
        # 后台生成代码（不显示给用户）
        code = self.generate_code()
        
        # 确保代码生成成功
        if not code:
            QMessageBox.warning(self, "代码生成失败", "无法生成模型代码，请检查模型结构")
            return
        
        # 根据运行模式执行不同操作
        if self.run_mode == "train":
            self.run_training_mode(run_params)
        else:
            self.run_inference_mode(run_params)
    
    def prepare_run_params(self):
        """准备运行参数"""
        params = {
            "use_gpu": self.use_gpu and self.has_gpu,
            "mode": self.run_mode,
            "dataset": self.canvas.current_dataset
        }
        
        # 添加训练特定参数
        if self.run_mode == "train":
            params.update({
                "epochs": int(self.epochs_input.currentText()),
                "batch_size": int(self.batch_size_combo.currentText()),
                "learning_rate": float(self.lr_combo.currentText()),
                "optimizer": self.optimizer_combo.currentText(),
                "save_best": self.save_best_check.isChecked(),
                "data_augmentation": {
                    "random_flip": self.random_flip_check.isChecked(),
                    "random_crop": self.random_crop_check.isChecked(),
                    "normalize": self.normalize_check.isChecked()
                }
            })
        # 添加推理特定参数
        else:
            params.update({
                "batch_inference": self.batch_inference_check.isChecked(),
                "visualize": self.visualize_check.isChecked()
            })
        
        return params
    
    def run_training_mode(self, params):
        """运行训练模式"""
        # 显示训练开始信息
        self.run_status_label.setText("训练中...")
        self.run_progress.setValue(0)
        self.cancel_button.setEnabled(True)
        self.statusBar.showMessage(f"模型训练开始 | {'GPU' if params['use_gpu'] else 'CPU'} 模式 | {params['epochs']}轮训练")
        
        # 切换到运行标签页
        self.right_tabs.setCurrentWidget(self.run_tab)
        
        # 更新结果文本
        self.results_text.clear()
        self.results_text.append(f"开始训练模型...\n")
        self.results_text.append(f"使用数据集: {params['dataset']}")
        self.results_text.append(f"训练轮数: {params['epochs']}")
        self.results_text.append(f"批次大小: {params['batch_size']}")
        self.results_text.append(f"学习率: {params['learning_rate']}")
        self.results_text.append(f"优化器: {params['optimizer']}")
        self.results_text.append(f"设备: {'GPU' if params['use_gpu'] else 'CPU'}\n")
        
        # 通知Canvas执行训练
        self.canvas.train_model(params, self.update_training_progress)
    
    def run_inference_mode(self, params):
        """运行推理模式"""
        # 显示推理开始信息
        self.run_status_label.setText("推理中...")
        self.run_progress.setValue(0)
        self.cancel_button.setEnabled(True)
        self.statusBar.showMessage(f"模型推理开始 | {'GPU' if params['use_gpu'] else 'CPU'} 模式")
        
        # 切换到运行标签页
        self.right_tabs.setCurrentWidget(self.run_tab)
        
        # 更新结果文本
        self.results_text.clear()
        self.results_text.append(f"开始推理模型...\n")
        self.results_text.append(f"使用数据集: {params['dataset']}")
        self.results_text.append(f"批量推理: {'是' if params['batch_inference'] else '否'}")
        self.results_text.append(f"可视化结果: {'是' if params['visualize'] else '否'}")
        self.results_text.append(f"设备: {'GPU' if params['use_gpu'] else 'CPU'}\n")
        
        # 通知Canvas执行推理
        self.canvas.infer_model(params, self.update_inference_progress)
    
    def update_training_progress(self, epoch, total_epochs, loss, accuracy=None, message=None):
        """更新训练进度"""
        # 更新进度条
        progress = int((epoch / total_epochs) * 100)
        self.run_progress.setValue(progress)
        
        # 更新状态信息
        self.run_status_label.setText(f"训练中...第 {epoch}/{total_epochs} 轮")
        
        # 添加到结果文本
        status_text = f"轮次 {epoch}/{total_epochs}, 损失: {loss:.4f}"
        if accuracy is not None:
            status_text += f", 准确率: {accuracy:.2f}%"
        
        self.results_text.append(status_text)
        
        if message:
            self.results_text.append(message)
        
        # 如果训练完成
        if epoch >= total_epochs:
            self.run_status_label.setText("训练完成")
            self.cancel_button.setEnabled(False)
            self.statusBar.showMessage("模型训练已完成")
            self.results_text.append("\n训练完成！")
    
    def update_inference_progress(self, progress, results=None, message=None):
        """更新推理进度"""
        # 更新进度条
        self.run_progress.setValue(progress)
        
        # 更新状态信息
        if progress < 100:
            self.run_status_label.setText(f"推理中...{progress}%")
        else:
            self.run_status_label.setText("推理完成")
            self.cancel_button.setEnabled(False)
            self.statusBar.showMessage("模型推理已完成")
        
        # 添加到结果文本
        if results:
            self.results_text.append(results)
        
        if message:
            self.results_text.append(message)
        
        # 如果推理完成
        if progress >= 100:
            self.results_text.append("\n推理完成！")
    
    def export_code(self):
        """导出PyTorch代码到文件"""
        filename, _ = QFileDialog.getSaveFileName(self, "导出PyTorch代码", "", "Python代码 (*.py);;所有文件 (*)")
        if filename:
            if not filename.endswith('.py'):
                filename += '.py'
            
            # 生成代码，确保包含GPU支持并遵循当前设置的运行模式
            model_dict = self.canvas.get_model_dict()
            model_dict["use_gpu"] = self.use_gpu and self.has_gpu
            model_dict["run_mode"] = self.run_mode
            
            code = self.code_generator.generate_code(model_dict)
            
            # 添加GPU检测和使用代码
            gpu_code = """
# 自动检测并使用GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
"""
            # 在import语句之后插入GPU检测代码
            import_end = code.find("\n\n", code.find("import torch"))
            if import_end != -1:
                code = code[:import_end+2] + gpu_code + code[import_end+2:]
            
            # 保存代码到文件
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(code)
            
            self.statusBar.showMessage(f"代码已导出到: {filename}")
            
            # 显示成功对话框
            QMessageBox.information(self, "导出成功", f"PyTorch代码已成功导出到:\n{filename}")
    
    def show_about(self):
        """显示关于对话框"""
        about_text = """
        <h3>DeepLearn Playground - 深度学习探索乐园</h3>
        <p>版本: 2.0.0</p>
        <p>一个面向儿童和初学者的深度学习可视化教学软件</p>
        <p>让深度学习变得简单有趣!</p>
        <p>特性:</p>
        <ul>
            <li>可视化构建神经网络模型</li>
            <li>支持训练和推理模式</li>
            <li>自动检测并利用GPU加速</li>
            <li>现代化界面设计</li>
        </ul>
        """
        QMessageBox.about(self, "关于", about_text)
    
    def load_template(self, item):
        """加载预设模型模板"""
        if not item:
            return
            
        template_name = item.text()
        
        # 获取模板数据
        template_data = self.template_manager.get_template(template_name)
        if not template_data:
            QMessageBox.warning(self, "模板加载失败", f"无法加载模板: {template_name}")
            return
        
        # 确认是否替换当前模型
        if self.canvas.components:
            reply = QMessageBox.question(
                self, "加载模板", 
                "加载模板将替换当前的模型。是否继续?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
        
        # 清空画布并加载模板
        self.canvas.clear_canvas()
        
        # 根据模板类型执行不同的加载逻辑
        if template_name == "LeNet-5":
            self.load_lenet5_template()
        elif template_name == "简单CNN":
            self.load_simple_cnn_template()
        elif template_name == "MLP分类器":
            self.load_mlp_template()
        else:
            # 通用模板加载逻辑
            try:
                # 尝试从模板文件加载
                template_file = os.path.join("templates", f"{template_name}.json")
                if os.path.exists(template_file):
                    self.canvas.load_model(template_file)
                    self.statusBar.showMessage(f"已加载模板: {template_name}")
                else:
                    QMessageBox.warning(self, "模板加载失败", f"模板文件不存在: {template_file}")
            except Exception as e:
                QMessageBox.warning(self, "模板加载失败", f"加载模板时出错: {str(e)}")
    
    def load_dataset(self, dataset_name):
        """加载数据集"""
        self.canvas.load_dataset(dataset_name)
        
        # 更新状态栏
        if dataset_name == "mnist":
            self.statusBar.showMessage(f"已加载MNIST数据集 | {('GPU加速已启用' if self.use_gpu and self.has_gpu else 'CPU模式')}")
        elif dataset_name == "cifar10":
            self.statusBar.showMessage(f"已加载CIFAR-10数据集 | {('GPU加速已启用' if self.use_gpu and self.has_gpu else 'CPU模式')}")
        else:
            self.statusBar.showMessage(f"已加载数据集: {dataset_name} | {('GPU加速已启用' if self.use_gpu and self.has_gpu else 'CPU模式')}")
    
    def validate_model(self):
        """验证模型结构"""
        is_valid, message = self.canvas.validate_model()
        
        if is_valid:
            QMessageBox.information(self, "验证成功", "模型结构验证通过！")
            self.statusBar.showMessage("模型结构验证通过")
        else:
            QMessageBox.warning(self, "验证失败", f"模型结构有问题:\n{message}")
            self.statusBar.showMessage("模型结构验证失败")
    
    def new_model(self):
        """创建新模型"""
        if self.canvas.components:
            reply = QMessageBox.question(
                self, "新建模型", 
                "创建新模型将清除当前模型。是否继续?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
        
        self.canvas.clear_canvas()
        self.statusBar.showMessage("已创建新模型")
    
    def save_model(self):
        """保存模型"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存模型", "", "DeepLearn模型 (*.dlp);;所有文件 (*)"
        )
        
        if filename:
            if not filename.endswith('.dlp'):
                filename += '.dlp'
                
            self.canvas.save_model(filename)
            self.statusBar.showMessage(f"模型已保存到: {filename}")
    
    def load_model(self):
        """加载模型"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "加载模型", "", "DeepLearn模型 (*.dlp);;所有文件 (*)"
        )
        
        if filename:
            try:
                self.canvas.load_model(filename)
                self.statusBar.showMessage(f"已加载模型: {filename}")
            except Exception as e:
                QMessageBox.warning(self, "加载失败", f"加载模型时出错: {str(e)}")
    
    def show_tutorial(self):
        """显示教程"""
        tutorial_text = """
        <h3>DeepLearn Playground 快速入门</h3>
        
        <h4>1. 构建模型</h4>
        <ul>
            <li>从左侧组件面板拖拽神经网络组件到中央画布</li>
            <li>点击组件并拖动到另一个组件以创建连接</li>
            <li>双击组件可以编辑其属性</li>
            <li>右键点击组件显示更多选项</li>
        </ul>
        
        <h4>2. 运行模型</h4>
        <ul>
            <li>选择数据集（MNIST或CIFAR-10）</li>
            <li>选择运行模式（训练或推理）</li>
            <li>配置训练或推理参数</li>
            <li>点击"运行模型"按钮开始执行</li>
        </ul>
        
        <h4>3. 使用模板</h4>
        <ul>
            <li>在"经典模型"标签页可以找到预设模板</li>
            <li>双击模板名称或选择后点击"加载选中模板"按钮</li>
            <li>加载模板后可以进一步修改模型结构</li>
        </ul>
        
        <h4>4. 导出和共享</h4>
        <ul>
            <li>点击"导出代码"按钮可以将模型导出为PyTorch代码</li>
            <li>使用"保存模型"功能保存当前模型以便日后使用</li>
        </ul>
        
        <p>更多详细教程请访问我们的官方文档。</p>
        """
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("使用教程")
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText(tutorial_text)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()
    
    # 预设模板加载方法
    def load_lenet5_template(self):
        """加载LeNet-5模型模板"""
        # 先清除旧组件和连接
        self.canvas.clear_canvas()
        
        # 创建输入层
        input_comp = self.canvas.add_component("input", 100, 100)
        input_comp.config = {"shape": [1, 28, 28]}  # MNIST图像尺寸
        input_comp.update_text()
        
        # 创建第一个卷积层
        conv1 = self.canvas.add_component("conv2d", 100, 200)
        conv1.config = {"in_channels": 1, "out_channels": 6, "kernel_size": 5, "stride": 1, "padding": 0}
        conv1.update_text()
        
        # 创建第一个池化层
        pool1 = self.canvas.add_component("maxpool", 100, 300)
        pool1.config = {"kernel_size": 2, "stride": 2}
        pool1.update_text()
        
        # 创建第二个卷积层
        conv2 = self.canvas.add_component("conv2d", 100, 400)
        conv2.config = {"in_channels": 6, "out_channels": 16, "kernel_size": 5, "stride": 1, "padding": 0}
        conv2.update_text()
        
        # 创建第二个池化层
        pool2 = self.canvas.add_component("maxpool", 100, 500)
        pool2.config = {"kernel_size": 2, "stride": 2}
        pool2.update_text()
        
        # 创建展平层
        flatten = self.canvas.add_component("flatten", 100, 600)
        
        # 创建第一个全连接层
        fc1 = self.canvas.add_component("linear", 100, 700)
        fc1.config = {"in_features": 256, "out_features": 120}
        fc1.update_text()
        
        # 创建第二个全连接层
        fc2 = self.canvas.add_component("linear", 100, 800)
        fc2.config = {"in_features": 120, "out_features": 84}
        fc2.update_text()
        
        # 创建输出层
        output = self.canvas.add_component("output", 100, 900)
        output.config = {"num_classes": 10}
        output.update_text()
        
        # 手动创建连接而不是使用finish_connection方法
        # 输入 -> 卷积1
        conn1 = ConnectionLine(input_comp, conv1)
        self.canvas.scene.addItem(conn1)
        input_comp.connections_out.append(conn1)
        conv1.connections_in.append(conn1)
        self.canvas.connections.append(conn1)
        
        # 卷积1 -> 池化1
        conn2 = ConnectionLine(conv1, pool1)
        self.canvas.scene.addItem(conn2)
        conv1.connections_out.append(conn2)
        pool1.connections_in.append(conn2)
        self.canvas.connections.append(conn2)
        
        # 池化1 -> 卷积2
        conn3 = ConnectionLine(pool1, conv2)
        self.canvas.scene.addItem(conn3)
        pool1.connections_out.append(conn3)
        conv2.connections_in.append(conn3)
        self.canvas.connections.append(conn3)
        
        # 卷积2 -> 池化2
        conn4 = ConnectionLine(conv2, pool2)
        self.canvas.scene.addItem(conn4)
        conv2.connections_out.append(conn4)
        pool2.connections_in.append(conn4)
        self.canvas.connections.append(conn4)
        
        # 池化2 -> 展平
        conn5 = ConnectionLine(pool2, flatten)
        self.canvas.scene.addItem(conn5)
        pool2.connections_out.append(conn5)
        flatten.connections_in.append(conn5)
        self.canvas.connections.append(conn5)
        
        # 展平 -> 全连接1
        conn6 = ConnectionLine(flatten, fc1)
        self.canvas.scene.addItem(conn6)
        flatten.connections_out.append(conn6)
        fc1.connections_in.append(conn6)
        self.canvas.connections.append(conn6)
        
        # 全连接1 -> 全连接2
        conn7 = ConnectionLine(fc1, fc2)
        self.canvas.scene.addItem(conn7)
        fc1.connections_out.append(conn7)
        fc2.connections_in.append(conn7)
        self.canvas.connections.append(conn7)
        
        # 全连接2 -> 输出
        conn8 = ConnectionLine(fc2, output)
        self.canvas.scene.addItem(conn8)
        fc2.connections_out.append(conn8)
        output.connections_in.append(conn8)
        self.canvas.connections.append(conn8)
        
        # 加载MNIST数据集
        self.load_dataset("mnist")
        self.dataset_combo.setCurrentText("MNIST")
        
        self.statusBar.showMessage("已加载LeNet-5模板")
        
    def load_simple_cnn_template(self):
        """加载简单CNN模型模板"""
        # 先清除旧组件和连接
        self.canvas.clear_canvas()
        
        # 创建输入层
        input_comp = self.canvas.add_component("input", 100, 100)
        input_comp.config = {"shape": [3, 32, 32]}  # CIFAR-10图像尺寸
        input_comp.update_text()
        
        # 创建第一个卷积层
        conv1 = self.canvas.add_component("conv2d", 100, 200)
        conv1.config = {"in_channels": 3, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1}
        conv1.update_text()
        
        # 创建ReLU激活
        relu1 = self.canvas.add_component("relu", 100, 280)
        
        # 创建池化层
        pool1 = self.canvas.add_component("maxpool", 100, 360)
        pool1.config = {"kernel_size": 2, "stride": 2}
        pool1.update_text()
        
        # 创建第二个卷积层
        conv2 = self.canvas.add_component("conv2d", 100, 440)
        conv2.config = {"in_channels": 32, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1}
        conv2.update_text()
        
        # 创建ReLU激活
        relu2 = self.canvas.add_component("relu", 100, 520)
        
        # 创建池化层
        pool2 = self.canvas.add_component("maxpool", 100, 600)
        pool2.config = {"kernel_size": 2, "stride": 2}
        pool2.update_text()
        
        # 创建展平层
        flatten = self.canvas.add_component("flatten", 100, 680)
        
        # 创建全连接层
        fc = self.canvas.add_component("linear", 100, 760)
        fc.config = {"in_features": 4096, "out_features": 10}
        fc.update_text()
        
        # 创建输出层
        output = self.canvas.add_component("output", 100, 840)
        output.config = {"num_classes": 10}
        output.update_text()
        
        # 手动创建连接
        # 输入 -> 卷积1
        conn1 = ConnectionLine(input_comp, conv1)
        self.canvas.scene.addItem(conn1)
        input_comp.connections_out.append(conn1)
        conv1.connections_in.append(conn1)
        self.canvas.connections.append(conn1)
        
        # 卷积1 -> ReLU1
        conn2 = ConnectionLine(conv1, relu1)
        self.canvas.scene.addItem(conn2)
        conv1.connections_out.append(conn2)
        relu1.connections_in.append(conn2)
        self.canvas.connections.append(conn2)
        
        # ReLU1 -> 池化1
        conn3 = ConnectionLine(relu1, pool1)
        self.canvas.scene.addItem(conn3)
        relu1.connections_out.append(conn3)
        pool1.connections_in.append(conn3)
        self.canvas.connections.append(conn3)
        
        # 池化1 -> 卷积2
        conn4 = ConnectionLine(pool1, conv2)
        self.canvas.scene.addItem(conn4)
        pool1.connections_out.append(conn4)
        conv2.connections_in.append(conn4)
        self.canvas.connections.append(conn4)
        
        # 卷积2 -> ReLU2
        conn5 = ConnectionLine(conv2, relu2)
        self.canvas.scene.addItem(conn5)
        conv2.connections_out.append(conn5)
        relu2.connections_in.append(conn5)
        self.canvas.connections.append(conn5)
        
        # ReLU2 -> 池化2
        conn6 = ConnectionLine(relu2, pool2)
        self.canvas.scene.addItem(conn6)
        relu2.connections_out.append(conn6)
        pool2.connections_in.append(conn6)
        self.canvas.connections.append(conn6)
        
        # 池化2 -> 展平
        conn7 = ConnectionLine(pool2, flatten)
        self.canvas.scene.addItem(conn7)
        pool2.connections_out.append(conn7)
        flatten.connections_in.append(conn7)
        self.canvas.connections.append(conn7)
        
        # 展平 -> 全连接
        conn8 = ConnectionLine(flatten, fc)
        self.canvas.scene.addItem(conn8)
        flatten.connections_out.append(conn8)
        fc.connections_in.append(conn8)
        self.canvas.connections.append(conn8)
        
        # 全连接 -> 输出
        conn9 = ConnectionLine(fc, output)
        self.canvas.scene.addItem(conn9)
        fc.connections_out.append(conn9)
        output.connections_in.append(conn9)
        self.canvas.connections.append(conn9)
        
        # 加载CIFAR-10数据集
        self.load_dataset("cifar10")
        self.dataset_combo.setCurrentText("CIFAR-10")
        
        self.statusBar.showMessage("已加载简单CNN模板")
    
    def load_mlp_template(self):
        """加载MLP分类器模板"""
        # 先清除旧组件和连接
        self.canvas.clear_canvas()
        
        # 创建组件
        input_comp = self.canvas.add_component("input", 100, 100)
        input_comp.config = {"shape": [1, 28, 28]}  # MNIST图像尺寸
        input_comp.update_text()
        
        flatten = self.canvas.add_component("flatten", 100, 200)
        
        fc1 = self.canvas.add_component("linear", 100, 300)
        fc1.config = {"in_features": 784, "out_features": 512}
        fc1.update_text()
        
        relu1 = self.canvas.add_component("relu", 100, 400)
        
        dropout1 = self.canvas.add_component("dropout", 100, 500)
        dropout1.config = {"p": 0.2}
        dropout1.update_text()
        
        fc2 = self.canvas.add_component("linear", 100, 600)
        fc2.config = {"in_features": 512, "out_features": 128}
        fc2.update_text()
        
        relu2 = self.canvas.add_component("relu", 100, 700)
        
        fc3 = self.canvas.add_component("linear", 100, 800)
        fc3.config = {"in_features": 128, "out_features": 10}
        fc3.update_text()
        
        output = self.canvas.add_component("output", 100, 900)
        output.config = {"num_classes": 10}
        output.update_text()
        
        # 手动创建连接
        # 输入 -> 展平
        conn1 = ConnectionLine(input_comp, flatten)
        self.canvas.scene.addItem(conn1)
        input_comp.connections_out.append(conn1)
        flatten.connections_in.append(conn1)
        self.canvas.connections.append(conn1)
        
        # 展平 -> 全连接1
        conn2 = ConnectionLine(flatten, fc1)
        self.canvas.scene.addItem(conn2)
        flatten.connections_out.append(conn2)
        fc1.connections_in.append(conn2)
        self.canvas.connections.append(conn2)
        
        # 全连接1 -> ReLU1
        conn3 = ConnectionLine(fc1, relu1)
        self.canvas.scene.addItem(conn3)
        fc1.connections_out.append(conn3)
        relu1.connections_in.append(conn3)
        self.canvas.connections.append(conn3)
        
        # ReLU1 -> Dropout1
        conn4 = ConnectionLine(relu1, dropout1)
        self.canvas.scene.addItem(conn4)
        relu1.connections_out.append(conn4)
        dropout1.connections_in.append(conn4)
        self.canvas.connections.append(conn4)
        
        # Dropout1 -> 全连接2
        conn5 = ConnectionLine(dropout1, fc2)
        self.canvas.scene.addItem(conn5)
        dropout1.connections_out.append(conn5)
        fc2.connections_in.append(conn5)
        self.canvas.connections.append(conn5)
        
        # 全连接2 -> ReLU2
        conn6 = ConnectionLine(fc2, relu2)
        self.canvas.scene.addItem(conn6)
        fc2.connections_out.append(conn6)
        relu2.connections_in.append(conn6)
        self.canvas.connections.append(conn6)
        
        # ReLU2 -> 全连接3
        conn7 = ConnectionLine(relu2, fc3)
        self.canvas.scene.addItem(conn7)
        relu2.connections_out.append(conn7)
        fc3.connections_in.append(conn7)
        self.canvas.connections.append(conn7)
        
        # 全连接3 -> 输出
        conn8 = ConnectionLine(fc3, output)
        self.canvas.scene.addItem(conn8)
        fc3.connections_out.append(conn8)
        output.connections_in.append(conn8)
        self.canvas.connections.append(conn8)
        
        # 加载MNIST数据集
        self.load_dataset("mnist")
        self.dataset_combo.setCurrentText("MNIST")
        
        self.statusBar.showMessage("已加载MLP分类器模板")