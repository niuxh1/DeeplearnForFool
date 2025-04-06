#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据集管理器 - 提供示例数据集
"""

import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

class DatasetManager:
    """数据集管理器类，提供示例数据集"""
    
    def __init__(self, data_dir="./datasets"):
        self.data_dir = data_dir
        
        # 确保数据目录存在
        os.makedirs(data_dir, exist_ok=True)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def load_mnist(self, batch_size=64):
        """加载MNIST数据集"""
        try:
            # 训练集
            train_dataset = datasets.MNIST(
                root=self.data_dir,
                train=True,
                download=True,
                transform=self.transform
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            
            # 测试集
            test_dataset = datasets.MNIST(
                root=self.data_dir,
                train=False,
                download=True,
                transform=self.transform
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False
            )
            
            # 示例图像
            sample_data = next(iter(train_loader))
            sample_images, sample_labels = sample_data
            
            return {
                "name": "MNIST",
                "train_loader": train_loader,
                "test_loader": test_loader,
                "classes": list(range(10)),
                "input_shape": [1, 28, 28],
                "sample_images": sample_images,
                "sample_labels": sample_labels
            }
        except Exception as e:
            print(f"加载MNIST数据集时出错: {e}")
            return None
    
    def load_cifar10(self, batch_size=64):
        """加载CIFAR-10数据集"""
        try:
            # 训练集
            train_dataset = datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                download=True,
                transform=self.transform_rgb
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            
            # 测试集
            test_dataset = datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                download=True,
                transform=self.transform_rgb
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False
            )
            
            # 类别名称
            classes = ['飞机', '汽车', '鸟', '猫', '鹿',
                       '狗', '青蛙', '马', '船', '卡车']
            
            # 示例图像
            sample_data = next(iter(train_loader))
            sample_images, sample_labels = sample_data
            
            return {
                "name": "CIFAR-10",
                "train_loader": train_loader,
                "test_loader": test_loader,
                "classes": classes,
                "input_shape": [3, 32, 32],
                "sample_images": sample_images,
                "sample_labels": sample_labels
            }
        except Exception as e:
            print(f"加载CIFAR-10数据集时出错: {e}")
            return None
    
    def load_mini_imagenet(self, batch_size=32):
        """加载简化版ImageNet数据集（仅用于示例）"""
        try:
            # 创建随机示例数据
            # 在实际使用时，应该下载真实的数据集
            class DummyDataset(Dataset):
                def __init__(self, size=1000, image_size=(3, 64, 64), num_classes=10):
                    self.size = size
                    self.image_size = image_size
                    self.num_classes = num_classes
                    
                    # 创建随机图像和标签
                    self.images = torch.randn(size, *image_size)
                    self.labels = torch.randint(0, num_classes, (size,))
                
                def __len__(self):
                    return self.size
                
                def __getitem__(self, idx):
                    return self.images[idx], self.labels[idx]
            
            # 创建假数据集
            train_dataset = DummyDataset(size=1000)
            test_dataset = DummyDataset(size=200)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False
            )
            
            # 示例图像
            sample_data = next(iter(train_loader))
            sample_images, sample_labels = sample_data
            
            return {
                "name": "Mini-ImageNet (示例)",
                "train_loader": train_loader,
                "test_loader": test_loader,
                "classes": [f"类别{i}" for i in range(10)],
                "input_shape": [3, 64, 64],
                "sample_images": sample_images,
                "sample_labels": sample_labels
            }
        except Exception as e:
            print(f"加载Mini-ImageNet数据集时出错: {e}")
            return None
    
    def load_dataset(self, name, batch_size=64):
        """根据名称加载数据集"""
        if name.lower() == "mnist":
            return self.load_mnist(batch_size)
        elif name.lower() == "cifar10":
            return self.load_cifar10(batch_size)
        elif name.lower() == "imagenet":
            return self.load_mini_imagenet(batch_size)
        else:
            print(f"未知数据集: {name}")
            return None