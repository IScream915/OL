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

sys.path.append('models')

from models.overlock import overlock_b, overlock_s, overlock_t, overlock_xt
from dataset import RSITMDDataset, get_transforms


def load_model(checkpoint_path, config):
    """加载训练好的模型"""
    # 创建模型
    if config.model_name == 'overlock_b':
        model = overlock_b(pretrained=False, num_classes=config.num_classes)
    elif config.model_name == 'overlock_s':
        model = overlock_s(pretrained=False, num_classes=config.num_classes)
    elif config.model_name == 'overlock_t':
        model = overlock_t(pretrained=False, num_classes=config.num_classes)
    elif config.model_name == 'overlock_xt':
        model = overlock_xt(pretrained=False, num_classes=config.num_classes)

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
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

    plt.figure(figsize=(15, 8))
    plt.bar(range(len(unique)), counts)
    plt.xticks(range(len(unique)), [class_names[i] for i in unique], rotation=45, ha='right')
    plt.title('Class Distribution in Test Set')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    """主函数"""
    config = type('Config', (), {})()

    # 配置参数（从训练配置继承或修改）
    config.data_dir = 'datasets/RSITMD'
    config.model_name = 'overlock_b'
    config.num_classes = 33
    config.image_size = 224
    config.batch_size = 32
    config.num_workers = 4
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 检查点路径
    checkpoint_path = 'outputs/best_checkpoint.pth'

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

    # 加载模型
    model, epoch = load_model(checkpoint_path, config)
    print(f"Model loaded from epoch {epoch}")

    # 评估
    preds, targets, probs = evaluate_model(model, test_loader, config)

    # 计算指标
    # 整体指标
    accuracy = np.mean(preds == targets)

    # 详细分类报告
    report = classification_report(
        targets, preds,
        target_names=test_dataset.idx_to_class.values(),
        output_dict=True
    )

    # 打印结果
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        targets, preds,
        target_names=test_dataset.idx_to_class.values()
    ))

    # 保存结果
    os.makedirs('evaluation_results', exist_ok=True)

    # 保存分类报告
    with open('evaluation_results/classification_report.json', 'w') as f:
        json.dump(report, f, indent=4)

    # 绘制混淆矩阵
    cm = confusion_matrix(targets, preds)
    plot_confusion_matrix(cm, test_dataset.idx_to_class.values(),
                         'evaluation_results/confusion_matrix.png')

    # 绘制类别分布
    plot_class_distribution(targets, test_dataset.idx_to_class.values(),
                          'evaluation_results/class_distribution.png')

    # 保存预测结果
    results_df = pd.DataFrame({
        'filename': [sample['filename'] for sample in test_dataset.samples],
        'true_label': targets,
        'predicted_label': preds,
        'true_class': [test_dataset.idx_to_class[label] for label in targets],
        'predicted_class': [test_dataset.idx_to_class[pred] for pred in preds],
        'confidence': [probs[i][preds[i]] for i in range(len(preds))]
    })
    results_df.to_csv('evaluation_results/predictions.csv', index=False)

    # 按类别分析
    class_accuracy = {}
    for class_name, class_idx in test_dataset.class_to_idx.items():
        mask = targets == class_idx
        if mask.sum() > 0:
            class_acc = np.mean(preds[mask] == targets[mask])
            class_accuracy[class_name] = {
                'accuracy': class_acc,
                'support': mask.sum()
            }

    # 保存类别准确率
    class_acc_df = pd.DataFrame.from_dict(class_accuracy, orient='index')
    class_acc_df = class_acc_df.sort_values('accuracy', ascending=False)
    class_acc_df.to_csv('evaluation_results/per_class_accuracy.csv')

    print("\nPer-class accuracy (top 10 best):")
    print(class_acc_df.head(10))
    print("\nPer-class accuracy (bottom 10 worst):")
    print(class_acc_df.tail(10))

    print("\nEvaluation results saved to 'evaluation_results/' directory")


if __name__ == '__main__':
    main()