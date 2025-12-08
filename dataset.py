import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from collections import Counter


class RSITMDDataset(Dataset):
    """RSITMD数据集加载器"""

    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir: 数据集根目录
            split: 'train' 或 'val'
            transform: 数据变换
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        # 加载类别映射
        with open(os.path.join(data_dir, 'class_to_idx.json'), 'r') as f:
            self.class_to_idx = json.load(f)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # 加载split文件
        split_file = os.path.join(data_dir, f'{split}_split.jsonl')
        self.samples = []
        with open(split_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                self.samples.append(data)

        # 统计类别分布
        labels = [sample['label'] for sample in self.samples]
        self.class_counts = Counter(labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载图像
        img_path = os.path.join(self.data_dir, 'images', sample['filename'])
        image = Image.open(img_path).convert('RGB')

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 获取标签
        label = sample['label']

        return image, label

    def get_class_weights(self):
        """计算类别权重用于处理不平衡问题"""
        total_samples = len(self.samples)
        num_classes = len(self.class_to_idx)

        # 使用逆频率权重
        class_weights = []
        for i in range(num_classes):
            if i in self.class_counts:
                weight = total_samples / (num_classes * self.class_counts[i])
            else:
                weight = 1.0
            class_weights.append(weight)

        return torch.FloatTensor(class_weights)


def get_transforms(image_size=224, normalize_type='imagenet'):
    """获取数据变换"""

    # ImageNet标准化参数
    if normalize_type == 'imagenet':
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    else:
        # 可以计算数据集自己的均值和标准差
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

    # 训练时的数据增强
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    # 验证时的变换
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, val_transform


def create_dataset(data_dir, image_size=224, batch_size=32, num_workers=4):
    """创建数据加载器"""

    # 获取变换
    train_transform, val_transform = get_transforms(image_size)

    # 创建数据集
    train_dataset = RSITMDDataset(data_dir, split='train', transform=train_transform)
    val_dataset = RSITMDDataset(data_dir, split='val', transform=val_transform)

    # 获取类别权重
    class_weights = train_dataset.get_class_weights()

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, class_weights, len(train_dataset.class_to_idx)


if __name__ == '__main__':
    # 测试数据加载
    data_dir = 'datasets/RSITMD'

    # 创建数据集
    train_dataset = RSITMDDataset(data_dir, split='train')
    val_dataset = RSITMDDataset(data_dir, split='val')

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"类别数: {len(train_dataset.class_to_idx)}")
    print("\n类别分布（训练集）:")
    for class_name, idx in train_dataset.class_to_idx.items():
        count = train_dataset.class_counts.get(idx, 0)
        print(f"  {class_name}: {count} 样本")