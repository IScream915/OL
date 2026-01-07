import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import argparse

sys.path.append('models')

from models.overlock import overlock_b, overlock_s, overlock_t, overlock_xt
from dataset import RSITMDDataset, get_transforms


def load_model(checkpoint_path, config):
    """加载训练好的模型"""
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 从checkpoint中获取num_classes
    checkpoint_num_classes = None
    if 'config' in checkpoint and 'num_classes' in checkpoint['config']:
        checkpoint_num_classes = checkpoint['config']['num_classes']

    # 使用config中的num_classes（已经从数据集获取）
    num_classes = config.num_classes

    # 检查类别数是否匹配
    if checkpoint_num_classes is not None and checkpoint_num_classes != num_classes:
        print(f"WARNING: Checkpoint has {checkpoint_num_classes} classes, but dataset has {num_classes} classes.")
        print(f"Reinitializing the classification head to match dataset classes.")

    # 创建模型（使用数据集的类别数）
    if config.model_name == 'overlock_b':
        model = overlock_b(pretrained=False, num_classes=num_classes)
    elif config.model_name == 'overlock_s':
        model = overlock_s(pretrained=False, num_classes=num_classes)
    elif config.model_name == 'overlock_t':
        model = overlock_t(pretrained=False, num_classes=num_classes)
    elif config.model_name == 'overlock_xt':
        model = overlock_xt(pretrained=False, num_classes=num_classes)

    # 加载权重，处理类别数不匹配的情况
    model_state_dict = checkpoint['model_state_dict']

    if checkpoint_num_classes != num_classes:
        # 过滤掉分类器层的权重
        filtered_state_dict = {}
        for k, v in model_state_dict.items():
            if 'head' not in k and 'fc' not in k and 'classifier' not in k:
                filtered_state_dict[k] = v
        # 只加载骨干网络权重，分类头使用随机初始化
        model_dict = model.state_dict()
        model_dict.update(filtered_state_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded backbone weights only. Classification head reinitialized for {num_classes} classes.")
    else:
        # 类别数匹配，加载全部权重
        model.load_state_dict(model_state_dict)
        print(f"Loaded full model weights from checkpoint.")

    model.to(config.device)
    model.eval()

    return model, checkpoint['epoch']


def evaluate_model(model, test_loader, config):
    """评估模型"""
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_targets), np.array(all_probs)


def plot_confusion_matrix(cm, class_names, output_path):
    """绘制混淆矩阵"""
    # 将class_names转换为列表
    class_names = list(class_names)

    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_class_distribution(targets, class_names, output_path):
    """绘制类别分布"""
    unique, counts = np.unique(targets, return_counts=True)

    # 将class_names转换为列表
    class_names = list(class_names)

    plt.figure(figsize=(15, 8))
    plt.bar(range(len(unique)), counts)
    plt.xticks(range(len(unique)), [class_names[i] for i in unique], rotation=45, ha='right')
    plt.title('Class Distribution in Test Set')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估OverLoCK模型')
    parser.add_argument('--data_dir', type=str, default='datasets/RSITMD', help='数据集路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径')
    parser.add_argument('--model_name', type=str, default='overlock_b',
                        choices=['overlock_t', 'overlock_s', 'overlock_b', 'overlock_xt'],
                        help='模型名称')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--image_size', type=int, default=224, help='图像尺寸')
    parser.add_argument('--output_dir', type=str, default=None, help='评估结果输出目录')

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
        # 默认从对应数据集的输出目录读取
        checkpoint_path = f'outputs/{dataset_name}/best_checkpoint.pth'
    else:
        checkpoint_path = args.checkpoint

    # 设置输出目录
    if args.output_dir is None:
        output_dir = f'evaluation_results/{dataset_name}'
    else:
        output_dir = args.output_dir

    print(f"Evaluating model: {checkpoint_path}")
    print(f"Device: {config.device}")

    # 加载数据集
    _, val_transform = get_transforms(config.image_size)
    test_dataset = RSITMDDataset(config.data_dir, split='val', transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of classes in dataset: {len(test_dataset.class_to_idx)}")

    # 更新 config.num_classes 为数据集实际的类别数
    config.num_classes = len(test_dataset.class_to_idx)

    # 加载模型
    model, epoch = load_model(checkpoint_path, config)
    print(f"Model loaded from epoch {epoch}")

    # 评估
    preds, targets, probs = evaluate_model(model, test_loader, config)

    # 计算指标
    # 整体指标
    accuracy = np.mean(preds == targets)

    # 获取实际存在的标签
    unique_labels = sorted(np.unique(np.concatenate([targets, preds])))
    idx_to_class_dict = dict(test_dataset.idx_to_class)

    # 只对实际存在的标签生成target_names
    actual_target_names = [idx_to_class_dict[i] for i in unique_labels]

    # 详细分类报告
    report = classification_report(
        targets, preds,
        labels=unique_labels,
        target_names=actual_target_names,
        output_dict=True
    )

    # 打印结果
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"\nNote: Found {len(unique_labels)} classes in data vs {len(test_dataset.class_to_idx)} in class_to_idx.json")
    print("\nClassification Report:")
    print(classification_report(
        targets, preds,
        labels=unique_labels,
        target_names=actual_target_names
    ))

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)

    # 保存分类报告
    with open(f'{output_dir}/classification_report.json', 'w') as f:
        json.dump(report, f, indent=4)

    # 绘制混淆矩阵 - 只使用实际存在的类别
    cm = confusion_matrix(targets, preds, labels=unique_labels)
    plot_confusion_matrix(cm, actual_target_names,
                         f'{output_dir}/confusion_matrix.png')

    # 绘制类别分布 - 只使用实际存在的类别
    plot_class_distribution(targets, actual_target_names,
                          f'{output_dir}/class_distribution.png')

    # 保存预测结果
    # 获取文件名列表，根据数据集的实际结构
    try:
        # 如果数据集有samples属性
        filenames = [sample['filename'] for sample in test_dataset.samples]
    except:
        # 如果没有，尝试其他方式获取文件名
        try:
            filenames = [os.path.basename(path) for path in test_dataset.image_paths]
        except:
            # 如果都没有，使用索引作为文件名
            filenames = [f'image_{i}.jpg' for i in range(len(targets))]

    # 将idx_to_class转换为字典以便查找
    idx_to_class = dict(test_dataset.idx_to_class)

    results_df = pd.DataFrame({
        'filename': filenames,
        'true_label': targets,
        'predicted_label': preds,
        'true_class': [idx_to_class[label] for label in targets],
        'predicted_class': [idx_to_class[pred] for pred in preds],
        'confidence': [probs[i][preds[i]] for i in range(len(preds))]
    })
    results_df.to_csv(f'{output_dir}/predictions.csv', index=False)

    # 按类别分析
    class_accuracy = {}
    for class_name, class_idx in test_dataset.class_to_idx.items():
        mask = targets == class_idx
        if np.sum(mask) > 0:  # 使用np.sum而不是mask.sum
            class_acc = np.mean(preds[mask] == targets[mask])
            class_accuracy[class_name] = {
                'accuracy': class_acc,
                'support': int(np.sum(mask))
            }

    # 保存类别准确率
    class_acc_df = pd.DataFrame.from_dict(class_accuracy, orient='index')
    class_acc_df = class_acc_df.sort_values('accuracy', ascending=False)
    class_acc_df.to_csv(f'{output_dir}/per_class_accuracy.csv')

    print("\nPer-class accuracy (top 10 best):")
    print(class_acc_df.head(10))
    print("\nPer-class accuracy (bottom 10 worst):")
    print(class_acc_df.tail(10))

    print(f"\nEvaluation results saved to '{output_dir}/' directory")


if __name__ == '__main__':
    main()