import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from datetime import datetime
import logging
from tqdm import tqdm
import argparse

# 添加模型路径
sys.path.append('models')

from models.overlock import overlock_b, overlock_s, overlock_t, overlock_xt
from dataset import create_dataset, get_transforms
from config import Config


def setup_logger(output_dir):
    """设置日志"""
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def get_model(config):
    """获取模型"""
    # 根据配置选择模型
    if config.model_name == 'overlock_b':
        model = overlock_b(pretrained=False, num_classes=config.num_classes)
    elif config.model_name == 'overlock_s':
        model = overlock_s(pretrained=False, num_classes=config.num_classes)
    elif config.model_name == 'overlock_t':
        model = overlock_t(pretrained=False, num_classes=config.num_classes)
    elif config.model_name == 'overlock_xt':
        model = overlock_xt(pretrained=False, num_classes=config.num_classes)
    else:
        raise ValueError(f"Unknown model: {config.model_name}")

    # 加载预训练权重
    if os.path.exists(config.pretrained_weights):
        print(f"Loading pretrained weights from {config.pretrained_weights}")
        checkpoint = torch.load(config.pretrained_weights, map_location='cpu')

        # 处理不同的权重格式
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # 如果权重有module前缀（来自DataParallel），去掉它
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # 过滤掉分类器层的权重
        pretrained_dict = {k: v for k, v in state_dict.items()
                          if 'head' not in k and 'fc' not in k and 'classifier' not in k}

        # 加载权重
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        print(f"Loaded {len(pretrained_dict)}/{len(state_dict)} parameters")
    else:
        print(f"Warning: Pretrained weights not found at {config.pretrained_weights}")

    return model


def get_loss_function(config, class_weights=None):
    """获取损失函数"""
    if config.loss_function == 'crossentropy':
        if config.use_class_weights and class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(config.device))
        else:
            criterion = nn.CrossEntropyLoss()
    elif config.loss_function == 'labelsmooth':
        from timm.loss import LabelSmoothingCrossEntropy
        criterion = LabelSmoothingCrossEntropy(smoothing=config.label_smoothing)
    elif config.loss_function == 'focal':
        from timm.loss import FocalLoss
        criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
    else:
        raise ValueError(f"Unknown loss function: {config.loss_function}")

    return criterion


def get_optimizer(model, config):
    """获取优化器"""
    # 参数分组：骨干网络和学习率较小的分类头
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if 'head' in name or 'fc' in name or 'classifier' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    if config.optimizer == 'adam':
        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': config.learning_rate * 0.1},
            {'params': head_params, 'lr': config.learning_rate}
        ], weight_decay=config.weight_decay)
    elif config.optimizer == 'adamw':
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': config.learning_rate * 0.1},
            {'params': head_params, 'lr': config.learning_rate}
        ], weight_decay=config.weight_decay)
    elif config.optimizer == 'sgd':
        optimizer = optim.SGD([
            {'params': backbone_params, 'lr': config.learning_rate * 0.1},
            {'params': head_params, 'lr': config.learning_rate}
        ], momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    return optimizer


def get_scheduler(optimizer, config):
    """获取学习率调度器"""
    if config.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.min_lr
        )
    elif config.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.step_size, gamma=config.gamma
        )
    elif config.scheduler == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config.milestones, gamma=config.gamma
        )
    elif config.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=config.patience, verbose=True
        )
    else:
        scheduler = None

    return scheduler


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, config, logger, epoch):
    """训练一个epoch"""
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Train]')

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(config.device), targets.to(config.device)

        optimizer.zero_grad()

        # # 前向传播
        # if config.use_amp:
        #     with autocast():
        #         outputs = model(inputs)
        #         loss = criterion(outputs, targets)
        # else:
        #     outputs = model(inputs)
        #     loss = criterion(outputs, targets)

        # 前向传播
        if config.use_amp:
            with autocast():
                outputs = model(inputs)

                # --- [Fix Start] 处理 OverLoCK 返回字典的情况 ---
                if isinstance(outputs, dict):
                    # 1. 计算主损失
                    main_loss = criterion(outputs['main'], targets)
                    # 2. 计算辅助损失 (Auxiliary Loss)，通常权重设为 0.4
                    aux_loss = criterion(outputs['aux'], targets)
                    loss = main_loss + 0.4 * aux_loss

                    # 3. 修正 outputs 变量，让它只指向主输出，以便后面计算准确率
                    outputs = outputs['main']
                else:
                    loss = criterion(outputs, targets)
                # --- [Fix End] ---
        else:
            outputs = model(inputs)
            # 同样的逻辑应用在非 AMP 模式
            if isinstance(outputs, dict):
                main_loss = criterion(outputs['main'], targets)
                aux_loss = criterion(outputs['aux'], targets)
                loss = main_loss + 0.4 * aux_loss
                outputs = outputs['main']
            else:
                loss = criterion(outputs, targets)

        # 反向传播
        if config.use_amp:
            scaler.scale(loss).backward()

            # 梯度裁剪
            if config.use_grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            # 梯度裁剪
            if config.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

            optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.3f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })

        # 记录日志
        if batch_idx % config.log_freq == 0:
            logger.info(
                f'Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] '
                f'Loss: {running_loss/(batch_idx+1):.3f} '
                f'Acc: {100.*correct/total:.2f}%'
            )

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total

    return train_loss, train_acc


def validate(model, val_loader, criterion, config):
    """验证模型"""
    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    val_loss = val_loss / len(val_loader)
    val_acc = 100. * correct / total

    # 计算其他指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='weighted'
    )

    return val_loss, val_acc, precision, recall, f1, all_preds, all_targets


def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, config, is_best=False):
    """保存检查点"""
    os.makedirs(config.output_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_acc': val_acc,
        'config': config.__dict__
    }

    # 保存最新检查点
    torch.save(checkpoint, os.path.join(config.output_dir, 'latest_checkpoint.pth'))

    # 保存最好的检查点
    if is_best:
        torch.save(checkpoint, os.path.join(config.output_dir, 'best_checkpoint.pth'))

    # 定期保存
    if (epoch + 1) % config.save_freq == 0:
        torch.save(checkpoint, os.path.join(config.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练OverLoCK模型')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None, help='批大小')
    parser.add_argument('--image_size', type=int, default=None, help='图像尺寸')
    parser.add_argument('--learning_rate', type=float, default=None, help='学习率')
    parser.add_argument('--data_dir', type=str, default=None, help='数据集路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--model_name', type=str, default=None, choices=['overlock_t', 'overlock_s', 'overlock_b', 'overlock_xt'], help='模型名称')

    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 加载配置
    config = Config()

    # 用命令行参数覆盖配置
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.image_size is not None:
        config.image_size = args.image_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.resume is not None:
        config.resume = args.resume
    if args.model_name is not None:
        config.model_name = args.model_name

    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # 设置日志
    logger = setup_logger(config.output_dir)
    logger.info(f"Starting training with config: {config.__dict__}")

    # 创建数据集
    logger.info("Creating datasets...")
    train_loader, val_loader, class_weights, num_classes = create_dataset(
        data_dir=config.data_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    config.num_classes = num_classes
    logger.info(f"Dataset created: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val samples")

    # 创建模型
    logger.info("Creating model...")
    model = get_model(config)
    model = model.to(config.device)

    logger.info(f"Model created: {config.model_name} with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # 创建损失函数
    criterion = get_loss_function(config, class_weights)

    # 创建优化器
    optimizer = get_optimizer(model, config)

    # 创建学习率调度器
    scheduler = get_scheduler(optimizer, config)

    # 自动混合精度
    scaler = GradScaler() if config.use_amp else None

    # 训练循环
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(config.epochs):
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, config, logger, epoch
        )

        # 验证
        if (epoch + 1) % config.eval_freq == 0:
            val_loss, val_acc, precision, recall, f1, _, _ = validate(
                model, val_loader, criterion, config
            )

            logger.info(
                f'Epoch {epoch+1}: Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, '
                f'Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.2f}%, '
                f'Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}'
            )

            # 更新学习率
            if scheduler is not None:
                if config.scheduler == 'plateau':
                    scheduler.step(val_acc)
                else:
                    scheduler.step()

            # 保存最佳模型
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                epochs_no_improve = 0
                logger.info(f"New best validation accuracy: {best_val_acc:.2f}%")
            else:
                epochs_no_improve += 1

            # 保存检查点
            save_checkpoint(model, optimizer, scheduler, epoch, val_acc, config, is_best)

            # 早停
            if config.use_early_stopping and epochs_no_improve >= config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break


if __name__ == '__main__':
    main()