#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepLearnPlayground - 一个面向儿童和初学者的深度学习可视化教学软件
"""

import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

# 导入自定义模块
from components.main_window import MainWindow

def check_dependencies():
    """检查必要的依赖库是否安装"""
    try:
        import torch
        import numpy
        import matplotlib
        return True
    except ImportError as e:
        return False

def main():
    """主函数"""
    # 检查依赖
    if not check_dependencies():
        print("警告: 部分依赖库未安装。请运行: pip install torch torchvision numpy matplotlib")
        msg = "部分依赖库未安装。请运行:\npip install torch torchvision numpy matplotlib"
        if QApplication.instance() is None:
            app = QApplication(sys.argv)
            QMessageBox.warning(None, "依赖缺失", msg)
            sys.exit(1)
    
    # 创建应用程序
    app = QApplication(sys.argv)
    app.setApplicationName("DeepLearnPlayground")
    app.setOrganizationName("DeepLearnPlayground")
    
    # 设置样式
    app.setStyle("Fusion")
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()