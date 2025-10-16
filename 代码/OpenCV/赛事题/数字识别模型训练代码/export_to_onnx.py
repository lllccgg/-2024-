#!/usr/bin/env python3
"""
LeNet-5 PyTorch模型导出为ONNX格式的命令行工具
用法: python export_to_onnx.py --model_path lenet5_armor.pth --output_path lenet5_armor.onnx
"""

import torch
import torch.nn as nn
import argparse
import os
import sys

# 定义LeNet-5网络结构（与训练时保持一致）
class LeNet5(nn.Module):
    def __init__(self, num_classes=9):
        super(LeNet5, self).__init__()
        
        # 第一层卷积层：输入1通道，输出6通道，卷积核5x5，padding=same
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        # 第一层池化层：2x2平均池化
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        # 第二层卷积层：输入6通道，输出16通道，卷积核5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 第二层池化层：2x2平均池化
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        # 第三层卷积层：输入16通道，输出120通道，卷积核6x6（修正）
        self.conv3 = nn.Conv2d(16, 120, kernel_size=6)
        # 展平层
        self.flatten = nn.Flatten()
        # 全连接层：输入120，输出84
        self.fc1 = nn.Linear(120, 84)
        # 全连接层：输入84，输出9（装甲板数字识别：0-8）
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        # 第一层卷积 + 激活 + 池化
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        # 第二层卷积 + 激活 + 池化
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        # 第三层卷积 + 激活
        x = torch.relu(self.conv3(x))
        # 展平
        x = self.flatten(x)
        # 全连接层 + 激活
        x = torch.relu(self.fc1(x))
        # 输出层
        x = self.fc2(x)
        return x

def export_to_onnx(model_path, output_path, num_classes=9, input_size=(1, 1, 32, 32)):
    """
    将PyTorch模型导出为ONNX格式
    
    Args:
        model_path (str): PyTorch模型文件路径(.pth)
        output_path (str): 输出ONNX文件路径(.onnx)
        input_size (tuple): 输入张量尺寸 (batch_size, channels, height, width)
    """
    
    # 检查输入文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件 '{model_path}' 不存在!")
        return False
    
    try:
        print(f"🔄 正在加载模型: {model_path}")
        
        # 创建模型实例
        model = LeNet5(num_classes=num_classes)
        
        # 加载训练好的权重
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 处理不同的保存格式
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        # 设置为评估模式
        model.eval()
        
        print(f"✅ 模型加载成功!")
        print(f"📊 模型结构:")
        print(model)
        
        # 创建虚拟输入张量
        dummy_input = torch.randn(input_size)
        print(f"🎯 输入尺寸: {input_size}")
        
        # 导出为ONNX
        print(f"🔄 正在导出ONNX模型: {output_path}")
        
        torch.onnx.export(
            model,                          # 模型
            dummy_input,                    # 虚拟输入
            output_path,                    # 输出路径
            export_params=True,             # 导出参数
            opset_version=11,               # ONNX操作集版本
            do_constant_folding=True,       # 常量折叠优化
            input_names=['input'],          # 输入名称
            output_names=['output'],        # 输出名称
            dynamic_axes={                  # 动态轴（支持不同batch size）
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"✅ ONNX导出成功!")
        print(f"📁 输出文件: {output_path}")
        
        # 验证导出的ONNX模型
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print(f"✅ ONNX模型验证通过!")
            
            # 显示模型信息
            print(f"📊 ONNX模型信息:")
            print(f"   - 输入: {onnx_model.graph.input[0].name} {[d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]}")
            print(f"   - 输出: {onnx_model.graph.output[0].name} {[d.dim_value for d in onnx_model.graph.output[0].type.tensor_type.shape.dim]}")
            
        except ImportError:
            print("⚠️  警告: 未安装onnx库，跳过模型验证")
            print("   可通过 'pip install onnx' 安装")
        
        return True
        
    except Exception as e:
        print(f"❌ 导出失败: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='将LeNet-5 PyTorch模型导出为ONNX格式')
    parser.add_argument('--model_path', '-m', type=str, default='lenet5_armor.pth',
                       help='PyTorch模型文件路径 (默认: lenet5_armor.pth)')
    parser.add_argument('--output_path', '-o', type=str, default='lenet5_armor.onnx',
                       help='输出ONNX文件路径 (默认: lenet5_armor.onnx)')
    parser.add_argument('--num_classes', '-n', type=int, default=9,
                       help='模型输出类别数 (默认: 9, 装甲板数字0-8)')
    parser.add_argument('--batch_size', '-b', type=int, default=1,
                       help='批处理大小 (默认: 1)')
    parser.add_argument('--height', type=int, default=32,
                       help='输入图像高度 (默认: 32)')
    parser.add_argument('--width', type=int, default=32,
                       help='输入图像宽度 (默认: 32)')
    
    args = parser.parse_args()
    
    print("🚀 LeNet-5 PyTorch → ONNX 转换工具")
    print("=" * 50)
    
    # 构建输入尺寸
    input_size = (args.batch_size, 1, args.height, args.width)
    
    # 执行导出
    success = export_to_onnx(args.model_path, args.output_path, args.num_classes, input_size)
    
    if success:
        print("\n🎉 转换完成!")
        print(f"💡 使用建议:")
        print(f"   - C++部署: 使用ONNX Runtime")
        print(f"   - Python推理: 使用onnxruntime库")
        print(f"   - 模型优化: 可使用onnx-simplifier进一步优化")
        sys.exit(0)
    else:
        print("\n❌ 转换失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()