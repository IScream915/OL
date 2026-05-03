#!/usr/bin/env python3
"""t-SNE 特征可视化 - 分析 OverLoCK 模型学到的特征表示"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import json
import argparse
from tqdm import tqdm
import seaborn as sns

sys.path.append('models')

from models.overlock import overlock_b, overlock_s, overlock_t, overlock_xt
from dataset import RSITMDDataset, get_transforms


class FeatureExtractor(nn.Module):
    """特征提取器 - 提取模型倒数第二层的特征"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        """前向传播，返回倒数第二层特征（分类头之前）"""
        # OverLoCK 模型有 forward_features 方法
        # 它返回 (x, ctx_cls) 元组，其中 x 是主要特征
        features, _ = self.model.forward_features(x)
        return features


def load_model(checkpoint_path, config):
    """加载训练好的模型"""
    print(f"加载模型: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 从 checkpoint 获取配置
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        model_name = config_dict.get('model_name', 'overlock_b')
        num_classes = config_dict.get('num_classes', config.num_classes)
    else:
        model_name = config.model.model_name if hasattr(config, 'model_name') else 'overlock_b'
        num_classes = config.num_classes

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
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(config.device)
    model.eval()

    return model, num_classes


def extract_features(model, test_loader, config, max_samples=5000):
    """提取特征和标签"""
    print("\n提取特征...")

    # 创建特征提取器
    feature_extractor = FeatureExtractor(model)
    feature_extractor.eval()

    all_features = []
    all_labels = []
    sample_count = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='提取特征'):
            inputs = inputs.to(config.device)

            # 提取特征
            features = feature_extractor(inputs)

            # 全局平均池化
            features = torch.nn.functional.adaptive_avg_pool2d(features, 1)
            features = features.flatten(1)  # [batch_size, feature_dim]

            all_features.append(features.cpu().numpy())
            all_labels.append(targets.numpy())

            sample_count += len(targets)
            if sample_count >= max_samples:
                break

    # 合并所有特征和标签
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f"提取了 {len(features)} 个样本，特征维度: {features.shape[1]}")

    return features, labels


def apply_tsne(features, perplexity=30, n_iter=1000):
    """应用 t-SNE 降维"""
    print(f"\n应用 t-SNE 降维 (perplexity={perplexity}, n_iter={n_iter})...")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=42,
        verbose=1,
        n_jobs=-1
    )

    features_2d = tsne.fit_transform(features)

    print(f"t-SNE 完成，降维后形状: {features_2d.shape}")

    return features_2d


def plot_tsne(features_2d, labels, class_names, output_path, title="t-SNE Visualization"):
    """绘制 t-SNE 可视化图"""
    print(f"\n绘制可视化图...")

    # 设置风格
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 10))

    # 生成颜色
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    # 使用不同的配色方案
    if n_classes <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    elif n_classes <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    else:
        colors = plt.cm.hsv(np.linspace(0, 0.85, n_classes))

    # 绘制每个类别的点
    for i, label in enumerate(unique_labels):
        mask = labels == label
        class_name = class_names.get(label, f"Class_{label}")

        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[i]],
            label=class_name,
            alpha=0.6,
            s=20,
            edgecolors='black',
            linewidths=0.3
        )

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)

    # 不显示图例
    pass

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"已保存: {output_path}")


def plot_tsne_by_dataset(features_2d, labels, class_names, output_path, title="t-SNE Visualization"):
    """按数据集分组绘制（适用于大量类别的情况）"""
    print(f"\n绘制分组可视化图...")

    sns.set_style("whitegrid")

    # 计算需要的子图数量
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    # 每个子图最多显示 16 个类别
    n_cols = 4
    n_rows = int(np.ceil(n_classes / 16 / n_cols))
    n_subplots = n_rows * n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_subplots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # 生成颜色
    colors = plt.cm.hsv(np.linspace(0, 0.85, 17))

    for idx in range(n_subplots):
        ax = axes[idx]
        start_idx = idx * 16
        end_idx = min(start_idx + 16, n_classes)
        current_labels = unique_labels[start_idx:end_idx]

        # 绘制当前子图的类别
        for i, label in enumerate(current_labels):
            mask = labels == label
            class_name = class_names.get(label, f"Class_{label}")

            ax.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=[colors[i]],
                label=class_name,
                alpha=0.6,
                s=15,
                edgecolors='black',
                linewidths=0.2
            )

        # ax.set_title(f'Classes {start_idx+1}-{end_idx}', fontsize=10)
        # ax.legend(fontsize=7, loc='upper right')
        # ax.grid(True, alpha=0.3)

        ax.set_title(f'Classes {start_idx + 1}-{end_idx}', fontsize=10)
        # 将图例锚点设置在图的右上角外部
        ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for idx in range(n_subplots, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"已保存: {output_path}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='t-SNE 特征可视化')
    parser.add_argument('--data_dir', type=str, default='datasets/RSITMD', help='数据集路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径')
    parser.add_argument('--model_name', type=str, default='overlock_b',
                        choices=['overlock_t', 'overlock_s', 'overlock_b', 'overlock_xt'],
                        help='模型名称')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--image_size', type=int, default=224, help='图像尺寸')
    parser.add_argument('--max_samples', type=int, default=5000, help='最大样本数')
    parser.add_argument('--perplexity', type=int, default=30, help='t-SNE perplexity 参数')
    parser.add_argument('--n_iter', type=int, default=1000, help='t-SNE 迭代次数')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    parser.add_argument('--batch', action='store_true', help='批量模式：处理 outputs/ 下所有模型的 best_checkpoint.pth')

    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 配置对象
    config = type('Config', (), {})()

    # 从数据集路径提取数据集名称
    dataset_name = os.path.basename(args.data_dir)

    # 配置参数
    config.data_dir = args.data_dir
    config.model_name = args.model_name
    config.image_size = args.image_size
    config.batch_size = args.batch_size
    config.num_workers = 4
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.num_classes = None  # 将从数据集获取

    # 设置检查点路径
    if args.checkpoint is None:
        checkpoint_path = f'outputs/{dataset_name}/best_checkpoint.pth'
    else:
        checkpoint_path = args.checkpoint

    # 设置输出目录
    if args.output_dir is None:
        output_dir = f'tsne_results/{dataset_name}'
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("t-SNE 特征可视化工具")
    print("=" * 70)
    print(f"数据集: {dataset_name}")
    print(f"检查点: {checkpoint_path}")
    print(f"设备: {config.device}")
    print(f"最大样本数: {args.max_samples}")

    # 加载数据集
    print("\n加载数据集...")
    _, val_transform = get_transforms(config.image_size)
    test_dataset = RSITMDDataset(config.data_dir, split='val', transform=val_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    config.num_classes = len(test_dataset.class_to_idx)
    print(f"验证集大小: {len(test_dataset)}")
    print(f"类别数: {config.num_classes}")

    # 加载模型
    model, _ = load_model(checkpoint_path, config)

    # 提取特征
    features, labels = extract_features(model, test_loader, config, args.max_samples)

    # 保存原始特征
    np.save(f'{output_dir}/features.npy', features)
    np.save(f'{output_dir}/labels.npy', labels)
    print(f"\n原始特征已保存: {output_dir}/features.npy")

    # 应用 t-SNE
    features_2d = apply_tsne(features, perplexity=args.perplexity, n_iter=args.n_iter)

    # 保存 t-SNE 结果
    np.save(f'{output_dir}/features_2d.npy', features_2d)
    print(f"t-SNE 结果已保存: {output_dir}/features_2d.npy")

    # 获取类别名称
    idx_to_class = dict(test_dataset.idx_to_class)

    # 绘制可视化图
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")

    if config.num_classes <= 25:
        # 类别较少时绘制单图
        output_path = f'{output_dir}/tsne_{timestamp}.png'
        plot_tsne(features_2d, labels, idx_to_class, output_path,
                 title=f't-SNE Visualization - {dataset_name}')
    else:
        # 类别较多时分页绘制
        output_path = f'{output_dir}/tsne_grid_{timestamp}.png'
        plot_tsne_by_dataset(features_2d, labels, idx_to_class, output_path,
                            title=f't-SNE Visualization - {dataset_name}')

    # 保存样本信息
    sample_info = {
        'dataset': dataset_name,
        'n_samples': len(features),
        'feature_dim': features.shape[1],
        'n_classes': config.num_classes,
        'perplexity': args.perplexity,
        'n_iter': args.n_iter,
        'checkpoint': checkpoint_path
    }

    with open(f'{output_dir}/info.json', 'w') as f:
        json.dump(sample_info, f, indent=4)

    print("\n" + "=" * 70)
    print(f"完成! 结果已保存到 {output_dir}/ 目录")
    print("=" * 70)


if __name__ == '__main__':
    main()
