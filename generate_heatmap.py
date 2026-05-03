#!/usr/bin/env python3
"""生成模型激活热力图（Grad-CAM）- 改进版"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import json
import torch.nn.functional as F

sys.path.append('models')
from models.overlock import overlock_b, overlock_s, overlock_t, overlock_xt


class GradCAMPlusPlus:
    """Grad-CAM++ 实现 - 更好的效果"""

    def __init__(self, model, target_layers):
        """
        Args:
            model: 模型
            target_layers: 目标层列表（可以是多个层）
        """
        self.model = model
        self.target_layers = target_layers if isinstance(target_layers, list) else [target_layers]
        self.gradients = {}
        self.activations = {}
        self.hooks = []

        # 注册钩子
        self._register_hooks()

    def _register_hooks(self):
        """注册前向和反向钩子"""
        for i, layer in enumerate(self.target_layers):
            # 前向钩子
            def forward_hook(module, input, output, idx=i):
                self.activations[idx] = output.detach()

            # 反向钩子
            def backward_hook(module, grad_in, grad_out, idx=i):
                self.gradients[idx] = grad_out[0].detach()

            handle_f = layer.register_forward_hook(forward_hook)
            handle_b = layer.register_full_backward_hook(backward_hook)
            self.hooks.extend([handle_f, handle_b])

    def remove_hooks(self):
        """移除所有钩子"""
        for handle in self.hooks:
            handle.remove()

    def generate(self, input_tensor, class_idx=None):
        """生成 Grad-CAM++ 热力图"""
        self.model.eval()

        # 前向传播
        with torch.enable_grad():
            output = self.model(input_tensor)

            # 处理 OverLoCK 的双输出
            if isinstance(output, dict):
                output = output['main']

            # 获取预测类别
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
                probs = F.softmax(output, dim=1)
                confidence = probs[0, class_idx].item()
                print(f"  预测类别: {class_idx}, 置信度: {confidence:.4f}")

            # 反向传播
            self.model.zero_grad()
            output[0, class_idx].backward(retain_graph=True)

        # 处理所有目标层
        cams = []
        for i in range(len(self.target_layers)):
            if i not in self.gradients or i not in self.activations:
                print(f"  警告: 层 {i} 没有梯度或激活")
                continue

            gradients = self.gradients[i][0]  # [C, H, W]
            activations = self.activations[i][0]  # [C, H, W]

            # Grad-CAM++ 计算
            # 计算全局平均池化的梯度
            weights = F.adaptive_avg_pool2d(gradients, 1).squeeze()  # [C]

            # 加权组合
            cam = (weights.view(-1, 1, 1) * activations).sum(dim=0)  # [H, W]

            # ReLU
            cam = F.relu(cam)

            # 归一化
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())

            cams.append(cam.cpu().numpy())

        # 如果有多个 CAM，取最大的
        if len(cams) > 1:
            final_cam = np.maximum.reduce(cams)
        elif len(cams) == 1:
            final_cam = cams[0]
        else:
            print("  错误: 没有生成任何 CAM")
            final_cam = np.zeros((224, 224))

        return final_cam, class_idx


def find_target_layers(model):
    """智能查找目标层"""
    print("\n查找目标层...")

    target_layers = []
    layer_names = []

    # 获取所有模块
    all_modules = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            all_modules.append((name, module))

    print(f"找到 {len(all_modules)} 个卷积/BN 层")

    # 策略1: 查找 backbone 中的最后几个卷积层
    backbone_layers = []
    for name, module in all_modules:
        if any(key in name for key in ['backbone', 'stem', 'stages', 'layer', 'block']):
            if isinstance(module, nn.Conv2d):
                backbone_layers.append((name, module))

    if backbone_layers:
        # 取最后2个卷积层
        selected = backbone_layers[-2:]
        for name, layer in selected:
            target_layers.append(layer)
            layer_names.append(name)
        print(f"策略1: 选择了 backbone 的最后 {len(selected)} 个卷积层")

    # 策略2: 如果没找到，使用最后几个卷积层
    if not target_layers:
        selected = all_modules[-3:]
        for name, layer in selected:
            target_layers.append(layer)
            layer_names.append(name)
        print(f"策略2: 选择了最后 {len(selected)} 个卷积层")

    print(f"选中的层: {layer_names}")

    return target_layers, layer_names


def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    print(f"\n加载模型: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 从 checkpoint 获取配置
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        model_name = config_dict.get('model_name', 'overlock_b')
        num_classes = config_dict.get('num_classes', 45)
    else:
        model_name = 'overlock_b'
        num_classes = 45

    print(f"模型: {model_name}, 类别数: {num_classes}")

    # 创建模型
    if model_name == 'overlock_b':
        model = overlock_b(pretrained=False, num_classes=num_classes)
    elif model_name == 'overlock_s':
        model = overlock_s(pretrained=False, num_classes=num_classes)
    elif model_name == 'overlock_t':
        model = overlock_t(pretrained=False, num_classes=num_classes)
    elif model_name == 'overlock_xt':
        model = overlock_xt(pretrained=False, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    print(f"模型加载完成，来自 epoch {epoch}")

    # 查找目标层
    target_layers, layer_names = find_target_layers(model)

    return model, target_layers, layer_names, num_classes


def get_image_transform(image_size=224):
    """获取图像预处理"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def denormalize(tensor):
    """反归一化图像"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean


def generate_heatmap_image(cam, original_image):
    """生成热力图叠加图像"""
    # 调整 CAM 大小
    cam_resized = cv2.resize(cam, (original_image.size[0], original_image.size[1]))

    # 应用 colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 转换为 float
    heatmap = heatmap.astype(float) / 255
    original_np = np.array(original_image).astype(float) / 255

    # 叠加
    alpha = 0.4
    overlaid = (1 - alpha) * original_np + alpha * heatmap

    return np.clip(overlaid, 0, 1)


def interactive_mode():
    """交互式模式"""
    print("=" * 70)
    print("激活热力图生成工具 (Grad-CAM++)")
    print("=" * 70)

    # 1. 选择数据集/模型
    print("\n可用的模型:")
    datasets = []
    outputs_dir = "outputs"
    if os.path.exists(outputs_dir):
        for d in sorted(os.listdir(outputs_dir)):
            if os.path.isdir(os.path.join(outputs_dir, d)):
                checkpoint_path = os.path.join(outputs_dir, d, "best_checkpoint.pth")
                if os.path.exists(checkpoint_path):
                    datasets.append(d)
                    print(f"  {len(datasets)}. {d}")

    if not datasets:
        print("错误: 在 outputs 文件夹下没有找到任何训练好的模型!")
        return

    print()
    choice = input(f"请选择数据集 (1-{len(datasets)}): ").strip()

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(datasets):
            print("无效选择!")
            return
        dataset_name = datasets[idx]
    except ValueError:
        print("无效输入!")
        return

    checkpoint_path = os.path.join(outputs_dir, dataset_name, "best_checkpoint.pth")

    # 2. 选择图片
    print()
    print("可用的图片:")
    images_dir = "heatmap/images"
    if not os.path.exists(images_dir):
        print(f"错误: {images_dir} 目录不存在!")
        return

    images = []
    for f in sorted(os.listdir(images_dir)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            images.append(f)
            print(f"  {len(images)}. {f}")

    if not images:
        print(f"错误: {images_dir} 目录下没有图片!")
        return

    print()
    choice = input(f"请选择图片 (1-{len(images)}, 或输入 'all'): ").strip()

    if choice.lower() == 'all':
        selected_images = images
    else:
        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(images):
                print("无效选择!")
                return
            selected_images = [images[idx]]
        except ValueError:
            print("无效输入!")
            return

    # 3. 加载数据集类别映射
    dataset_dir = f"datasets/{dataset_name}"
    class_to_idx_path = os.path.join(dataset_dir, "class_to_idx.json")
    idx_to_class = None

    if os.path.exists(class_to_idx_path):
        with open(class_to_idx_path, 'r') as f:
            class_to_idx = json.load(f)
            idx_to_class = {v: k for k, v in class_to_idx.items()}
        print(f"\n数据集类别: {len(class_to_idx)} 个")
    else:
        print(f"\n警告: 未找到 {class_to_idx_path}")

    # 4. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model, target_layers, layer_names, num_classes = load_model(checkpoint_path, device)
    grad_cam = GradCAMPlusPlus(model, target_layers)

    # 创建结果目录
    results_dir = "heatmap/results"
    os.makedirs(results_dir, exist_ok=True)

    transform = get_image_transform()

    # 5. 处理每张图片
    for image_name in selected_images:
        print()
        print("-" * 70)
        print(f"处理图片: {image_name}")

        image_path = os.path.join(images_dir, image_name)
        original_image = Image.open(image_path).convert('RGB')

        # 预处理
        input_tensor = transform(original_image).unsqueeze(0).to(device)

        # 生成 Grad-CAM
        try:
            cam, pred_class = grad_cam.generate(input_tensor)

            # 获取类别名称
            if idx_to_class and pred_class in idx_to_class:
                class_name = idx_to_class[pred_class]
            else:
                class_name = f"Class_{pred_class}"

            # 检查 CAM 是否有效
            if cam.max() == cam.min():
                print("  警告: CAM 是均匀的，可能梯度没有正确流动")
            else:
                print(f"  CAM 范围: [{cam.min():.4f}, {cam.max():.4f}]")

            # 生成可视化
            overlaid = generate_heatmap_image(cam, original_image)

            # 保存结果
            result_path = os.path.join(results_dir, f"heatmap_{os.path.splitext(image_name)[0]}.png")

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # 原图
            axes[0].imshow(original_image)
            axes[0].set_title('Original Image', fontsize=12)
            axes[0].axis('off')

            # 热力图
            im = axes[1].imshow(cam, cmap='jet', vmin=0, vmax=1)
            axes[1].set_title(f'Grad-CAM++\nPred: {class_name}', fontsize=12)
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046)

            # 叠加图
            axes[2].imshow(overlaid)
            axes[2].set_title('Overlay', fontsize=12)
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(result_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  已保存: {result_path}")

        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()

    # 移除钩子
    grad_cam.remove_hooks()

    print()
    print("=" * 70)
    print(f"完成! 热力图已保存到 {results_dir} 目录")
    print("=" * 70)


if __name__ == '__main__':
    interactive_mode()