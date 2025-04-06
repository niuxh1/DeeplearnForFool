#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
代码生成器 - 将可视化模型转换为PyTorch代码
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLabel, QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class CodeGenerator(QWidget):
    """代码生成器类，将可视化构建的神经网络转换为PyTorch代码"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        # 保存最后一次生成的代码
        self.last_code = ""
    
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 标题
        title = QLabel("<h3>PyTorch 代码</h3>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 代码编辑器
        self.code_editor = QTextEdit()
        self.code_editor.setReadOnly(True)
        self.code_editor.setFont(QFont("Courier New", 10))
        self.code_editor.setLineWrapMode(QTextEdit.NoWrap)
        self.code_editor.setText("# 拖放组件并连接它们以生成PyTorch代码\n# 然后点击\"生成代码\"按钮")
        
        layout.addWidget(self.code_editor)
    
    def set_code(self, code):
        """设置代码编辑器的内容"""
        if not code:
            # 避免设置空代码
            print("警告: 尝试设置空代码")
            return
            
        # 保存代码的副本
        self.last_code = code
        
        # 先清除当前内容
        self.code_editor.clear()
        
        # 强制刷新UI
        QApplication.processEvents()
        
        # 设置新内容
        self.code_editor.setText(code)
        
        # 确保更新显示
        self.code_editor.update()
        print(f"代码已更新，长度: {len(code)} 字符")
    
    def showEvent(self, event):
        """当窗口变为可见时触发"""
        super().showEvent(event)
        
        # 如果有上次生成的代码，确保它仍然显示
        if self.last_code and not self.code_editor.toPlainText():
            self.code_editor.setText(self.last_code)
            print("在showEvent中恢复了代码")
            
    def get_current_code(self):
        """获取当前显示的代码"""
        return self.code_editor.toPlainText()
    
    def generate_code(self, model_dict):
        """生成PyTorch代码"""
        components = model_dict.get("components", [])
        connections = model_dict.get("connections", [])
        
        if not components:
            return "# 模型为空，请添加组件"
        
        # 创建组件ID到索引的映射
        comp_id_to_idx = {comp["id"]: i for i, comp in enumerate(components)}
        
        # 找到输入和输出组件
        input_comps = [comp for comp in components if comp["type"] == "input"]
        output_comps = [comp for comp in components if comp["type"] == "output"]
        
        if not input_comps or not output_comps:
            return "# 模型必须包含输入层和输出层"
        
        # 构建连接图
        graph = {comp["id"]: [] for comp in components}
        for conn in connections:
            from_id = conn["from"]
            to_id = conn["to"]
            graph[from_id].append(to_id)
        
        # 生成导入语句
        code = "import torch\n"
        code += "import torch.nn as nn\n"
        code += "import torch.nn.functional as F\n\n"
        
        # 生成模型类
        code += "class MyModel(nn.Module):\n"
        code += "    def __init__(self):\n"
        code += "        super(MyModel, self).__init__()\n"
        
        # 生成层定义
        for comp in components:
            comp_id = comp["id"]
            comp_type = comp["type"]
            config = comp.get("config", {})
            
            # 根据组件类型生成不同的层定义
            if comp_type == "conv2d":
                in_channels = config.get("in_channels", 3)
                out_channels = config.get("out_channels", 16)
                kernel_size = config.get("kernel_size", 3)
                stride = config.get("stride", 1)
                padding = config.get("padding", 0)
                
                code += f"        self.conv_{comp_id[:6]} = nn.Conv2d({in_channels}, {out_channels}, "
                code += f"kernel_size={kernel_size}, stride={stride}, padding={padding})\n"
                
            elif comp_type == "maxpool":
                kernel_size = config.get("kernel_size", 2)
                stride = config.get("stride", 2)
                
                code += f"        self.pool_{comp_id[:6]} = nn.MaxPool2d(kernel_size={kernel_size}, stride={stride})\n"
                
            elif comp_type == "linear":
                in_features = config.get("in_features", 64)
                out_features = config.get("out_features", 10)
                
                code += f"        self.fc_{comp_id[:6]} = nn.Linear({in_features}, {out_features})\n"
                
            elif comp_type == "dropout":
                p = config.get("p", 0.5)
                
                code += f"        self.dropout_{comp_id[:6]} = nn.Dropout(p={p})\n"
                
            elif comp_type == "batchnorm":
                num_features = config.get("num_features", 16)
                
                code += f"        self.bn_{comp_id[:6]} = nn.BatchNorm2d({num_features})\n"
                
            elif comp_type == "lstm":
                input_size = config.get("input_size", 28)
                hidden_size = config.get("hidden_size", 64)
                num_layers = config.get("num_layers", 1)
                
                code += f"        self.lstm_{comp_id[:6]} = nn.LSTM("
                code += f"input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers})\n"
                
            elif comp_type == "gru":
                input_size = config.get("input_size", 28)
                hidden_size = config.get("hidden_size", 64)
                num_layers = config.get("num_layers", 1)
                
                code += f"        self.gru_{comp_id[:6]} = nn.GRU("
                code += f"input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers})\n"
                
            elif comp_type == "attention":
                embed_dim = config.get("embed_dim", 64)
                num_heads = config.get("num_heads", 4)
                
                code += f"        self.attention_{comp_id[:6]} = nn.MultiheadAttention("
                code += f"embed_dim={embed_dim}, num_heads={num_heads})\n"
                
            elif comp_type == "flatten":
                # 添加Flatten模块定义，使其在模型结构中可见
                code += f"        self.flatten_{comp_id[:6]} = nn.Flatten()\n"
        
        code += "\n"
        
        # 生成前向传播函数
        code += "    def forward(self, x):\n"
        
        # 检查输入形状
        input_comp = input_comps[0]  # 只处理第一个输入组件
        input_shape = input_comp.get("config", {}).get("shape", [3, 32, 32])
        code += f"        # 输入形状: [{input_shape[0]}, {input_shape[1]}, {input_shape[2]}]\n"
        
        # 使用拓扑排序生成前向传播代码
        # 简化起见，我们使用BFS遍历图
        visited = set()
        queue = [input_comp["id"]]
        visited.add(input_comp["id"])
        layer_outputs = {input_comp["id"]: "x"}
        
        while queue:
            current_id = queue.pop(0)
            current_comp = next((c for c in components if c["id"] == current_id), None)
            
            # 修复错误：确保当前组件ID在layer_outputs中存在
            if current_id not in layer_outputs:
                code += f"        # 警告：组件 {current_id[:6]} 没有输入连接，跳过\n"
                continue
                
            current_output = layer_outputs[current_id]
            
            # 根据当前组件类型生成操作
            if current_comp["type"] == "conv2d":
                output_var = f"x_{current_id[:6]}"
                code += f"        {output_var} = self.conv_{current_id[:6]}({current_output})\n"
                layer_outputs[current_id] = output_var
                
            elif current_comp["type"] == "maxpool":
                output_var = f"x_{current_id[:6]}"
                code += f"        {output_var} = self.pool_{current_id[:6]}({current_output})\n"
                layer_outputs[current_id] = output_var
                
            elif current_comp["type"] == "linear":
                output_var = f"x_{current_id[:6]}"
                code += f"        {output_var} = self.fc_{current_id[:6]}({current_output})\n"
                layer_outputs[current_id] = output_var
                
            elif current_comp["type"] == "relu":
                output_var = f"x_{current_id[:6]}"
                code += f"        {output_var} = F.relu({current_output})\n"
                layer_outputs[current_id] = output_var
                
            elif current_comp["type"] == "dropout":
                output_var = f"x_{current_id[:6]}"
                code += f"        {output_var} = self.dropout_{current_id[:6]}({current_output})\n"
                layer_outputs[current_id] = output_var
                
            elif current_comp["type"] == "batchnorm":
                output_var = f"x_{current_id[:6]}"
                code += f"        {output_var} = self.bn_{current_id[:6]}({current_output})\n"
                layer_outputs[current_id] = output_var
                
            elif current_comp["type"] == "flatten":
                output_var = f"x_{current_id[:6]}"
                code += f"        {output_var} = self.flatten_{current_id[:6]}({current_output})\n"
                layer_outputs[current_id] = output_var
                
            elif current_comp["type"] == "lstm":
                output_var = f"x_{current_id[:6]}"
                code += f"        {output_var}, _ = self.lstm_{current_id[:6]}({current_output})\n"
                layer_outputs[current_id] = output_var
                
            elif current_comp["type"] == "gru":
                output_var = f"x_{current_id[:6]}"
                code += f"        {output_var}, _ = self.gru_{current_id[:6]}({current_output})\n"
                layer_outputs[current_id] = output_var
                
            elif current_comp["type"] == "attention":
                output_var = f"x_{current_id[:6]}"
                code += f"        {output_var}, _ = self.attention_{current_id[:6]}({current_output}, {current_output}, {current_output})\n"
                layer_outputs[current_id] = output_var
                
            elif current_comp["type"] == "output":
                # 输出层通常是最后一个线性层的输出
                # 不需要特殊处理，只需将当前输入传递给返回语句
                pass
            
            # 将连接的后续节点加入队列
            for next_id in graph.get(current_id, []):
                if next_id not in visited:
                    visited.add(next_id)
                    queue.append(next_id)
                    
                    # 为确保下一个组件有连接，将当前组件输出记录为下一个组件的输入
                    layer_outputs.setdefault(next_id, current_output)
        
        # 生成返回语句，使用最后一个输出组件的输入作为最终输出
        output_comp = output_comps[0]  # 只处理第一个输出组件
        output_id = output_comp["id"]
        
        # 找到连接到输出组件的节点
        for conn in connections:
            if conn["to"] == output_id:
                from_id = conn["from"]
                if from_id in layer_outputs:
                    code += f"        return {layer_outputs[from_id]}\n"
                else:
                    # 增加处理：如果输出组件的前置节点未处理
                    code += f"        # 警告：输出层 {output_id[:6]} 的前置组件 {from_id[:6]} 未正确处理\n"
                    code += "        return x  # 使用输入作为输出\n"
                break
        else:
            code += "        return x  # 输出层未正确连接\n"
        
        # 示例使用代码
        code += "\n# 使用示例\n"
        code += "def main():\n"
        code += "    # 创建模型实例\n"
        code += "    model = MyModel()\n"
        code += f"    # 创建随机输入张量 (批量大小=1, 通道数={input_shape[0]}, 高度={input_shape[1]}, 宽度={input_shape[2]})\n"
        code += f"    x = torch.randn(1, {input_shape[0]}, {input_shape[1]}, {input_shape[2]})\n"
        code += "    # 前向传播\n"
        code += "    output = model(x)\n"
        code += "    print('输出形状:', output.shape)\n\n"
        code += "if __name__ == '__main__':\n"
        code += "    main()\n"
        
        return code