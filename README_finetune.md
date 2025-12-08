# OverLoCK 模型微调指南

本指南帮助你在 RSITMD 数据集上微调 OverLoCK 模型。

## 环境要求

确保已安装以下依赖：

```bash
# PyTorch (根据你的CUDA版本选择)
pip install torch torchvision torchaudio

# 其他依赖
pip install natten timm mmengine scikit-learn seaborn matplotlib tqdm pandas pillow
```

## 项目结构

```
OL/
├── models/                  # OverLoCK模型实现
│   ├── overlock.py         # 主模型文件
│   └── contmix.py          # Context-mixing实现
├── datasets/RSITMD/        # RSITMD数据集
│   ├── class_to_idx.json   # 类别映射
│   ├── train_split.jsonl   # 训练集划分
│   ├── val_split.jsonl     # 验证集划分
│   └── images/             # 图像文件
├── weights/                # 预训练权重
│   └── overlock_b_in1k_224.pth
├── dataset.py              # 数据加载器
├── config.py               # 训练配置
├── train.py                # 训练脚本
├── evaluate.py             # 评估脚本
└── outputs/                # 训练输出（自动创建）
```

## 快速开始

### 1. 准备环境

激活你的conda环境或虚拟环境，确保所有依赖已安装。

### 2. 检查数据集

确认 `datasets/RSITMD` 目录包含：
- `class_to_idx.json`: 类别到索引的映射
- `train_split.jsonl`: 训练集文件名和标签
- `val_split.jsonl`: 验证集文件名和标签
- `images/`: 包含所有TIFF图像

### 3. 配置训练参数

编辑 `config.py` 文件，根据需要调整参数：

```python
# 关键参数
model_name = 'overlock_b'  # 模型大小：xt, t, s, b
batch_size = 32           # 批大小（根据GPU内存调整）
epochs = 100              # 训练轮数
learning_rate = 1e-4      # 学习率
image_size = 224          # 输入图像大小
```

### 4. 开始训练

```bash
# 单GPU训练
python train.py

# 多GPU训练（如果有多张GPU）
python -m torch.distributed.launch --nproc_per_node=2 train.py
```

### 5. 监控训练

训练过程中会：
- 在控制台显示进度和指标
- 保存日志到 `outputs/` 目录
- 定期保存模型检查点

### 6. 评估模型

```bash
python evaluate.py
```

评估结果保存在 `evaluation_results/` 目录，包括：
- 分类报告
- 混淆矩阵
- 每个类别的准确率
- 预测结果CSV文件

## 配置说明

### 模型选择

- `overlock_xt`: 16M参数，适合快速实验
- `overlock_t`: 33M参数，平衡性能和速度
- `overlock_s`: 56M参数，更好的精度
- `overlock_b`: 95M参数，最高精度（推荐）

### 微调策略

1. **差异学习率**：骨干网络使用较小的学习率，分类头使用较大学习率
2. **类别权重**：自动计算权重处理类别不平衡
3. **数据增强**：包括翻转、旋转、颜色抖动等

### 训练技巧

1. **使用小批大小**：如果GPU内存不足，减小 `batch_size`
2. **调整学习率**：如果损失震荡，减小学习率
3. **早停**：默认开启，避免过拟合
4. **混合精度**：默认开启，加速训练并节省内存

## 常见问题

### Q: GPU内存不足怎么办？
A:
- 减小 `batch_size` 到16或8
- 使用更小的模型（如 `overlock_t`）
- 减小 `image_size` 到 192

### Q: 如何恢复训练？
A: 设置 `config.py` 中的 `resume` 参数：
```python
resume = 'outputs/latest_checkpoint.pth'
```

### Q: 如何使用其他数据集？
A:
- 参考 `dataset.py` 创建新的数据集类
- 更新 `config.py` 中的 `num_classes`
- 调整数据预处理参数

### Q: 训练太慢怎么办？
A:
- 启用混合精度（`use_amp = True`）
- 使用多GPU训练
- 使用更小的模型
- 增加数据加载的 `num_workers`

## 进阶使用

### 1. 自定义损失函数

在 `config.py` 中选择不同的损失函数：
- `crossentropy`: 标准交叉熵
- `labelsmooth`: 标签平滑
- `focal`: Focal Loss（用于类别不平衡）

### 2. 学习率调度

可选的调度器：
- `cosine`: 余弦退火
- `step`: 固定步长衰减
- `multistep`: 多步长衰减
- `plateau`: 基于验证精度调整

### 3. 数据增强

在 `dataset.py` 的 `get_transforms` 函数中添加更多增强：
```python
transforms.RandomResizedCrop(image_size),
transforms.RandomGrayscale(p=0.2),
transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
```

## 性能优化建议

1. **数据预处理**：提前计算数据集的均值和标准差
2. **模型优化**：考虑使用结构重参数化加速推理
3. **分布式训练**：使用多GPU加速大规模训练
4. **检查GPU使用**：确保GPU利用率高，数据加载不是瓶颈

## 引用

如果在研究中使用了此代码，请引用原论文：
```bibtex
@inproceedings{overlock2025,
  title={OverLoCK: Overview-first-Look-Closely-next ConvNet},
  author={...},
  booktitle={CVPR},
  year={2025}
}
```