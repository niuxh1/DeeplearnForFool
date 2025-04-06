#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
网络模板管理器 - 提供预定义的神经网络模型模板
"""

import os
import json
from PyQt5.QtWidgets import QListWidget, QListWidgetItem
from PyQt5.QtCore import Qt

class TemplateManager:
    """网络模板管理器类，提供经典的神经网络模型模板"""
    
    def __init__(self):
        # 模板配置
        self.templates = {
            "LeNet-5": self.create_lenet(),
            "AlexNet简化版": self.create_alexnet(),
            "VGG简化版": self.create_vgg(),
            "简单MLP": self.create_mlp(),
            "简单CNN": self.create_simple_cnn(),
            "LSTM文本分类": self.create_lstm()
        }
    
    def load_templates_to_list(self, list_widget):
        """加载模板到列表控件"""
        list_widget.clear()
        for name in self.templates.keys():
            item = QListWidgetItem(name)
            list_widget.addItem(item)
    
    def get_template(self, name):
        """获取指定名称的模板"""
        return self.templates.get(name)
    
    def get_template_list(self):
        """获取所有可用模板的名称列表"""
        return list(self.templates.keys())
    
    def create_lenet(self):
        """创建LeNet-5模型模板"""
        # LeNet-5架构
        # 输入 -> 卷积层 -> 池化层 -> 卷积层 -> 池化层 -> 展平层 -> 全连接层 -> 全连接层 -> 输出
        
        # 创建组件
        components = [
            {
                "id": "input_1",
                "type": "input",
                "x": 50,
                "y": 150,
                "config": {
                    "shape": [1, 28, 28]  # MNIST数据集形状
                }
            },
            {
                "id": "conv_1",
                "type": "conv2d",
                "x": 200,
                "y": 150,
                "config": {
                    "in_channels": 1,
                    "out_channels": 6,
                    "kernel_size": 5,
                    "stride": 1,
                    "padding": 0
                }
            },
            {
                "id": "pool_1",
                "type": "maxpool",
                "x": 350,
                "y": 150,
                "config": {
                    "kernel_size": 2,
                    "stride": 2
                }
            },
            {
                "id": "conv_2",
                "type": "conv2d",
                "x": 500,
                "y": 150,
                "config": {
                    "in_channels": 6,
                    "out_channels": 16,
                    "kernel_size": 5,
                    "stride": 1,
                    "padding": 0
                }
            },
            {
                "id": "pool_2",
                "type": "maxpool",
                "x": 650,
                "y": 150,
                "config": {
                    "kernel_size": 2,
                    "stride": 2
                }
            },
            {
                "id": "flatten_1",
                "type": "flatten",
                "x": 200,
                "y": 250
            },
            {
                "id": "fc_1",
                "type": "linear",
                "x": 350,
                "y": 250,
                "config": {
                    "in_features": 256,  # 16*4*4 = 256
                    "out_features": 120
                }
            },
            {
                "id": "relu_1",
                "type": "relu",
                "x": 500,
                "y": 250
            },
            {
                "id": "fc_2",
                "type": "linear",
                "x": 650,
                "y": 250,
                "config": {
                    "in_features": 120,
                    "out_features": 84
                }
            },
            {
                "id": "relu_2",
                "type": "relu",
                "x": 200,
                "y": 350
            },
            {
                "id": "fc_3",
                "type": "linear",
                "x": 350,
                "y": 350,
                "config": {
                    "in_features": 84,
                    "out_features": 10
                }
            },
            {
                "id": "output_1",
                "type": "output",
                "x": 500,
                "y": 350,
                "config": {
                    "num_classes": 10
                }
            }
        ]
        
        # 创建连接
        connections = [
            {"from": "input_1", "to": "conv_1"},
            {"from": "conv_1", "to": "pool_1"},
            {"from": "pool_1", "to": "conv_2"},
            {"from": "conv_2", "to": "pool_2"},
            {"from": "pool_2", "to": "flatten_1"},
            {"from": "flatten_1", "to": "fc_1"},
            {"from": "fc_1", "to": "relu_1"},
            {"from": "relu_1", "to": "fc_2"},
            {"from": "fc_2", "to": "relu_2"},
            {"from": "relu_2", "to": "fc_3"},
            {"from": "fc_3", "to": "output_1"}
        ]
        
        return {"components": components, "connections": connections}
    
    def create_alexnet(self):
        """创建AlexNet简化版模型模板"""
        # AlexNet简化版架构
        
        # 创建组件
        components = [
            {
                "id": "input_1",
                "type": "input",
                "x": 50,
                "y": 100,
                "config": {
                    "shape": [3, 224, 224]
                }
            },
            {
                "id": "conv_1",
                "type": "conv2d",
                "x": 200,
                "y": 100,
                "config": {
                    "in_channels": 3,
                    "out_channels": 64,
                    "kernel_size": 11,
                    "stride": 4,
                    "padding": 2
                }
            },
            {
                "id": "relu_1",
                "type": "relu",
                "x": 350,
                "y": 100
            },
            {
                "id": "pool_1",
                "type": "maxpool",
                "x": 500,
                "y": 100,
                "config": {
                    "kernel_size": 3,
                    "stride": 2
                }
            },
            {
                "id": "conv_2",
                "type": "conv2d",
                "x": 650,
                "y": 100,
                "config": {
                    "in_channels": 64,
                    "out_channels": 192,
                    "kernel_size": 5,
                    "stride": 1,
                    "padding": 2
                }
            },
            {
                "id": "relu_2",
                "type": "relu",
                "x": 200,
                "y": 200
            },
            {
                "id": "pool_2",
                "type": "maxpool",
                "x": 350,
                "y": 200,
                "config": {
                    "kernel_size": 3,
                    "stride": 2
                }
            },
            {
                "id": "conv_3",
                "type": "conv2d",
                "x": 500,
                "y": 200,
                "config": {
                    "in_channels": 192,
                    "out_channels": 384,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1
                }
            },
            {
                "id": "relu_3",
                "type": "relu",
                "x": 650,
                "y": 200
            },
            {
                "id": "flatten_1",
                "type": "flatten",
                "x": 200,
                "y": 300
            },
            {
                "id": "fc_1",
                "type": "linear",
                "x": 350,
                "y": 300,
                "config": {
                    "in_features": 9600,  # 简化后的尺寸
                    "out_features": 4096
                }
            },
            {
                "id": "relu_4",
                "type": "relu",
                "x": 500,
                "y": 300
            },
            {
                "id": "dropout_1",
                "type": "dropout",
                "x": 650,
                "y": 300,
                "config": {
                    "p": 0.5
                }
            },
            {
                "id": "fc_2",
                "type": "linear",
                "x": 200,
                "y": 400,
                "config": {
                    "in_features": 4096,
                    "out_features": 1000
                }
            },
            {
                "id": "output_1",
                "type": "output",
                "x": 350,
                "y": 400,
                "config": {
                    "num_classes": 1000
                }
            }
        ]
        
        # 创建连接
        connections = [
            {"from": "input_1", "to": "conv_1"},
            {"from": "conv_1", "to": "relu_1"},
            {"from": "relu_1", "to": "pool_1"},
            {"from": "pool_1", "to": "conv_2"},
            {"from": "conv_2", "to": "relu_2"},
            {"from": "relu_2", "to": "pool_2"},
            {"from": "pool_2", "to": "conv_3"},
            {"from": "conv_3", "to": "relu_3"},
            {"from": "relu_3", "to": "flatten_1"},
            {"from": "flatten_1", "to": "fc_1"},
            {"from": "fc_1", "to": "relu_4"},
            {"from": "relu_4", "to": "dropout_1"},
            {"from": "dropout_1", "to": "fc_2"},
            {"from": "fc_2", "to": "output_1"}
        ]
        
        return {"components": components, "connections": connections}
    
    def create_vgg(self):
        """创建VGG简化版模型模板"""
        # VGG简化版架构
        
        # 创建组件
        components = [
            {
                "id": "input_1",
                "type": "input",
                "x": 50,
                "y": 100,
                "config": {
                    "shape": [3, 224, 224]
                }
            },
            # 第一个卷积块
            {
                "id": "conv_1_1",
                "type": "conv2d",
                "x": 200,
                "y": 100,
                "config": {
                    "in_channels": 3,
                    "out_channels": 64,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1
                }
            },
            {
                "id": "relu_1_1",
                "type": "relu",
                "x": 350,
                "y": 100
            },
            {
                "id": "pool_1",
                "type": "maxpool",
                "x": 500,
                "y": 100,
                "config": {
                    "kernel_size": 2,
                    "stride": 2
                }
            },
            # 第二个卷积块
            {
                "id": "conv_2_1",
                "type": "conv2d",
                "x": 650,
                "y": 100,
                "config": {
                    "in_channels": 64,
                    "out_channels": 128,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1
                }
            },
            {
                "id": "relu_2_1",
                "type": "relu",
                "x": 200,
                "y": 200
            },
            {
                "id": "pool_2",
                "type": "maxpool",
                "x": 350,
                "y": 200,
                "config": {
                    "kernel_size": 2,
                    "stride": 2
                }
            },
            # 第三个卷积块
            {
                "id": "conv_3_1",
                "type": "conv2d",
                "x": 500,
                "y": 200,
                "config": {
                    "in_channels": 128,
                    "out_channels": 256,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1
                }
            },
            {
                "id": "relu_3_1",
                "type": "relu",
                "x": 650,
                "y": 200
            },
            {
                "id": "pool_3",
                "type": "maxpool",
                "x": 200,
                "y": 300,
                "config": {
                    "kernel_size": 2,
                    "stride": 2
                }
            },
            # 全连接层
            {
                "id": "flatten_1",
                "type": "flatten",
                "x": 350,
                "y": 300
            },
            {
                "id": "fc_1",
                "type": "linear",
                "x": 500,
                "y": 300,
                "config": {
                    "in_features": 256 * 28 * 28,  # 简化后的尺寸
                    "out_features": 4096
                }
            },
            {
                "id": "relu_4",
                "type": "relu",
                "x": 650,
                "y": 300
            },
            {
                "id": "dropout_1",
                "type": "dropout",
                "x": 200,
                "y": 400,
                "config": {
                    "p": 0.5
                }
            },
            {
                "id": "fc_2",
                "type": "linear",
                "x": 350,
                "y": 400,
                "config": {
                    "in_features": 4096,
                    "out_features": 1000
                }
            },
            {
                "id": "output_1",
                "type": "output",
                "x": 500,
                "y": 400,
                "config": {
                    "num_classes": 1000
                }
            }
        ]
        
        # 创建连接
        connections = [
            {"from": "input_1", "to": "conv_1_1"},
            {"from": "conv_1_1", "to": "relu_1_1"},
            {"from": "relu_1_1", "to": "pool_1"},
            {"from": "pool_1", "to": "conv_2_1"},
            {"from": "conv_2_1", "to": "relu_2_1"},
            {"from": "relu_2_1", "to": "pool_2"},
            {"from": "pool_2", "to": "conv_3_1"},
            {"from": "conv_3_1", "to": "relu_3_1"},
            {"from": "relu_3_1", "to": "pool_3"},
            {"from": "pool_3", "to": "flatten_1"},
            {"from": "flatten_1", "to": "fc_1"},
            {"from": "fc_1", "to": "relu_4"},
            {"from": "relu_4", "to": "dropout_1"},
            {"from": "dropout_1", "to": "fc_2"},
            {"from": "fc_2", "to": "output_1"}
        ]
        
        return {"components": components, "connections": connections}
    
    def create_mlp(self):
        """创建简单的多层感知机模型模板"""
        # 简单MLP架构
        
        # 创建组件
        components = [
            {
                "id": "input_1",
                "type": "input",
                "x": 100,
                "y": 100,
                "config": {
                    "shape": [1, 28, 28]  # MNIST数据集形状
                }
            },
            {
                "id": "flatten_1",
                "type": "flatten",
                "x": 300,
                "y": 100
            },
            {
                "id": "fc_1",
                "type": "linear",
                "x": 500,
                "y": 100,
                "config": {
                    "in_features": 784,  # 28*28=784
                    "out_features": 256
                }
            },
            {
                "id": "relu_1",
                "type": "relu",
                "x": 100,
                "y": 200
            },
            {
                "id": "fc_2",
                "type": "linear",
                "x": 300,
                "y": 200,
                "config": {
                    "in_features": 256,
                    "out_features": 128
                }
            },
            {
                "id": "relu_2",
                "type": "relu",
                "x": 500,
                "y": 200
            },
            {
                "id": "fc_3",
                "type": "linear",
                "x": 100,
                "y": 300,
                "config": {
                    "in_features": 128,
                    "out_features": 10
                }
            },
            {
                "id": "output_1",
                "type": "output",
                "x": 300,
                "y": 300,
                "config": {
                    "num_classes": 10
                }
            }
        ]
        
        # 创建连接
        connections = [
            {"from": "input_1", "to": "flatten_1"},
            {"from": "flatten_1", "to": "fc_1"},
            {"from": "fc_1", "to": "relu_1"},
            {"from": "relu_1", "to": "fc_2"},
            {"from": "fc_2", "to": "relu_2"},
            {"from": "relu_2", "to": "fc_3"},
            {"from": "fc_3", "to": "output_1"}
        ]
        
        return {"components": components, "connections": connections}
    
    def create_simple_cnn(self):
        """创建简单的CNN模型模板"""
        # 简单CNN架构
        
        # 创建组件
        components = [
            {
                "id": "input_1",
                "type": "input",
                "x": 100,
                "y": 100,
                "config": {
                    "shape": [3, 32, 32]  # CIFAR-10数据集形状
                }
            },
            {
                "id": "conv_1",
                "type": "conv2d",
                "x": 300,
                "y": 100,
                "config": {
                    "in_channels": 3,
                    "out_channels": 32,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1
                }
            },
            {
                "id": "relu_1",
                "type": "relu",
                "x": 500,
                "y": 100
            },
            {
                "id": "pool_1",
                "type": "maxpool",
                "x": 100,
                "y": 200,
                "config": {
                    "kernel_size": 2,
                    "stride": 2
                }
            },
            {
                "id": "conv_2",
                "type": "conv2d",
                "x": 300,
                "y": 200,
                "config": {
                    "in_channels": 32,
                    "out_channels": 64,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1
                }
            },
            {
                "id": "relu_2",
                "type": "relu",
                "x": 500,
                "y": 200
            },
            {
                "id": "pool_2",
                "type": "maxpool",
                "x": 100,
                "y": 300,
                "config": {
                    "kernel_size": 2,
                    "stride": 2
                }
            },
            {
                "id": "flatten_1",
                "type": "flatten",
                "x": 300,
                "y": 300
            },
            {
                "id": "fc_1",
                "type": "linear",
                "x": 500,
                "y": 300,
                "config": {
                    "in_features": 64 * 8 * 8,  # 64 * 8 * 8 = 4096
                    "out_features": 512
                }
            },
            {
                "id": "relu_3",
                "type": "relu",
                "x": 100,
                "y": 400
            },
            {
                "id": "fc_2",
                "type": "linear",
                "x": 300,
                "y": 400,
                "config": {
                    "in_features": 512,
                    "out_features": 10
                }
            },
            {
                "id": "output_1",
                "type": "output",
                "x": 500,
                "y": 400,
                "config": {
                    "num_classes": 10
                }
            }
        ]
        
        # 创建连接
        connections = [
            {"from": "input_1", "to": "conv_1"},
            {"from": "conv_1", "to": "relu_1"},
            {"from": "relu_1", "to": "pool_1"},
            {"from": "pool_1", "to": "conv_2"},
            {"from": "conv_2", "to": "relu_2"},
            {"from": "relu_2", "to": "pool_2"},
            {"from": "pool_2", "to": "flatten_1"},
            {"from": "flatten_1", "to": "fc_1"},
            {"from": "fc_1", "to": "relu_3"},
            {"from": "relu_3", "to": "fc_2"},
            {"from": "fc_2", "to": "output_1"}
        ]
        
        return {"components": components, "connections": connections}
    
    def create_lstm(self):
        """创建LSTM文本分类模型模板"""
        # LSTM文本分类架构
        
        # 创建组件
        components = [
            {
                "id": "input_1",
                "type": "input",
                "x": 100,
                "y": 150,
                "config": {
                    "shape": [1, 100, 1]  # 序列长度100
                }
            },
            {
                "id": "lstm_1",
                "type": "lstm",
                "x": 300,
                "y": 150,
                "config": {
                    "input_size": 1,
                    "hidden_size": 128,
                    "num_layers": 2
                }
            },
            {
                "id": "flatten_1",
                "type": "flatten",
                "x": 500,
                "y": 150
            },
            {
                "id": "fc_1",
                "type": "linear",
                "x": 100,
                "y": 250,
                "config": {
                    "in_features": 12800,  # 简化后的尺寸
                    "out_features": 64
                }
            },
            {
                "id": "relu_1",
                "type": "relu",
                "x": 300,
                "y": 250
            },
            {
                "id": "fc_2",
                "type": "linear",
                "x": 500,
                "y": 250,
                "config": {
                    "in_features": 64,
                    "out_features": 2
                }
            },
            {
                "id": "output_1",
                "type": "output",
                "x": 300,
                "y": 350,
                "config": {
                    "num_classes": 2
                }
            }
        ]
        
        # 创建连接
        connections = [
            {"from": "input_1", "to": "lstm_1"},
            {"from": "lstm_1", "to": "flatten_1"},
            {"from": "flatten_1", "to": "fc_1"},
            {"from": "fc_1", "to": "relu_1"},
            {"from": "relu_1", "to": "fc_2"},
            {"from": "fc_2", "to": "output_1"}
        ]
        
        return {"components": components, "connections": connections}