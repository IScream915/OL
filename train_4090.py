import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from datetime import datetime
import logging
from tqdm import tqdm
import wandb  # 可选：用于更详细的日志记录

# 添加模型路径
sys.path.append('models')

from models.overlock import overlock_b, overlock_s, overlock_t, overlock_xt
from dataset import RSITMDDataset, get_transforms
from config import Config
import torchvision.transforms as transforms


class RTX4090Trainer:
    """针对RTX 4090优化的训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 设置TensorBoard
        self.writer = SummaryWriter(log_dir=f'runs/overlock_finetune_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

        # 初始化Wandb（可选）
        if hasattr(config, 'use_wandb') and config.use_wandb:
            wandb.init(
                project="overlock-finetune",
                name=f"{config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=config.__dict__
            )

        self.setup()

    def setup(self):
        """设置训练环境"""
        # 设置随机种子
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        # 启用cudnn优化
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True  # 加速训练
        torch.backends.cudnn.deterministic = False  # 不需要确定性结果

        # 设置日志
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.logger = self.setup_logger()

        # GPU信息
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    def setup_logger(self):
        """设置日志"""
        log_file = os.path.join(self.config.output_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        return logging.getLogger(__name__)

    def create_datasets(self):
        """创建数据集，利用大内存进行预加载"""
        self.logger.info("Creating datasets with optimized settings...")

        # 使用更大的transform
        train_transform, val_transform = self.get_optimized_transforms()

        # 创建数据集
        from dataset import RSITMDDataset

        train_dataset = RSITMDDataset(
            self.config.data_dir,
            split='train',
            transform=train_transform
        )
        val_dataset = RSITMDDataset(
            self.config.data_dir,
            split='val',
            transform=val_transform
        )

        # 获取类别权重
        class_weights = train_dataset.get_class_weights()
        self.config.num_classes = len(train_dataset.class_to_idx)

        # 创建数据加载器，优化4090性能
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=2,  # 预取更多数据
            drop_last=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size * 2,  # 验证时可以使用更大的batch size
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers
        )

        self.logger.info(f"Datasets created: {len(train_dataset)} train, {len(val_dataset)} val")

        return class_weights

    def get_optimized_transforms(self):
        """获取优化的数据增强"""
        # 更强的数据增强
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(
                self.config.image_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),  # 遥感图像可能需要
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),  # 随机擦除
        ])

        val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        return train_transform, val_transform

    def get_model(self):
        """获取模型"""
        self.logger.info(f"Creating model: {self.config.model_name}")

        model_map = {
            'overlock_b': overlock_b,
            'overlock_s': overlock_s,
            'overlock_t': overlock_t,
            'overlock_xt': overlock_xt
        }

        model = model_map[self.config.model_name](
            pretrained=False,
            num_classes=self.config.num_classes
        )

        # 加载预训练权重
        if os.path.exists(self.config.pretrained_weights):
            self.load_pretrained_weights(model)

        model = model.to(self.device)

        # 编译模型（PyTorch 2.0+）
        if hasattr(torch, 'compile'):
            self.logger.info("Compiling model with torch.compile...")
            model = torch.compile(model)

        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info(f"Total parameters: {total_params/1e6:.2f}M")
        self.logger.info(f"Trainable parameters: {trainable_params/1e6:.2f}M")

        return model

    def load_pretrained_weights(self, model):
        """加载预训练权重"""
        checkpoint = torch.load(self.config.pretrained_weights, map_location='cpu')

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # 处理权重
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        pretrained_dict = {k: v for k, v in state_dict.items()
                          if 'head' not in k and 'fc' not in k and 'classifier' not in k}

        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        self.logger.info(f"Loaded {len(pretrained_dict)} pretrained parameters")

    def get_optimizer(self, model):
        """获取优化器"""
        # 分层学习率
        backbone_params = []
        head_params = []

        for name, param in model.named_parameters():
            if 'head' in name or 'fc' in name or 'classifier' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        if self.config.optimizer == 'adamw':
            optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': self.config.learning_rate * 0.1},
                {'params': head_params, 'lr': self.config.learning_rate}
            ], weight_decay=self.config.weight_decay, betas=(0.9, 0.999))
        elif self.config.optimizer == 'sgd':
            optimizer = optim.SGD([
                {'params': backbone_params, 'lr': self.config.learning_rate * 0.1},
                {'params': head_params, 'lr': self.config.learning_rate}
            ], momentum=self.config.momentum, weight_decay=self.config.weight_decay, nesterov=True)

        return optimizer

    def get_scheduler(self, optimizer):
        """获取学习率调度器"""
        if self.config.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=self.config.epochs // 3, T_mult=2, eta_min=self.config.min_lr
            )
        elif self.config.scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.7, patience=7, verbose=True
            )

        return scheduler

    def train_one_epoch(self, model, criterion, optimizer, scaler, epoch):
        """训练一个epoch"""
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()

        # 使用tqdm显示进度
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.epochs}')

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

            # # 混合精度训练
            # with autocast():
            #     outputs = model(inputs)
            #     loss = criterion(outputs, targets)

            # 混合精度训练
            with autocast():
                outputs = model(inputs)

                # --- [修改开始] 适配 OverLoCK 返回字典的情况 ---
                if isinstance(outputs, dict):
                    # OverLoCK 返回 {'main': ..., 'aux': ...}
                    # 这里的 0.4 是辅助损失的权重，帮助模型更稳定收敛
                    main_loss = criterion(outputs['main'], targets)
                    aux_loss = criterion(outputs['aux'], targets)
                    loss = main_loss + 0.4 * aux_loss

                    # 重要：将 outputs 指回主输出 Tensor，否则后面计算准确率(outputs.max)会报错
                    outputs = outputs['main']
                else:
                    # 正常情况（验证模式或非 OverLoCK 模型）
                    loss = criterion(outputs, targets)
                # --- [修改结束] ---

            # 反向传播
            scaler.scale(loss).backward()

            # 梯度裁剪
            if self.config.use_grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()

            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 更新进度条
            current_acc = 100. * correct / total
            current_loss = running_loss / (batch_idx + 1)

            pbar.set_postfix({
                'Loss': f'{current_loss:.3f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

            # 记录到TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
            self.writer.add_scalar('Train/BatchAcc', current_acc, global_step)

        epoch_time = time.time() - epoch_start
        train_loss = running_loss / len(self.train_loader)
        train_acc = 100. * correct / total

        return train_loss, train_acc, epoch_time

    def validate(self, model, criterion, epoch):
        """验证模型"""
        model.eval()

        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc='Validation'):
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_loss = val_loss / len(self.val_loader)
        val_acc = 100. * correct / total

        # 计算F1分数
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted'
        )

        # 记录到TensorBoard
        self.writer.add_scalar('Val/Loss', val_loss, epoch)
        self.writer.add_scalar('Val/Accuracy', val_acc, epoch)
        self.writer.add_scalar('Val/F1', f1, epoch)

        return val_loss, val_acc, precision, recall, f1

    def train(self):
        """主训练循环"""
        # 创建数据集和模型
        class_weights = self.create_datasets()
        model = self.get_model()
        optimizer = self.get_optimizer(model)
        scheduler = self.get_scheduler(optimizer)

        # 损失函数
        if self.config.use_class_weights and class_weights is not None:
            criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(self.device),
                label_smoothing=self.config.label_smoothing
            )
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)

        # 混合精度
        scaler = GradScaler()

        # 训练循环
        best_val_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            # 训练
            train_loss, train_acc, train_time = self.train_one_epoch(
                model, criterion, optimizer, scaler, epoch
            )

            # 验证
            val_loss, val_acc, precision, recall, f1 = self.validate(
                model, criterion, epoch
            )

            # 更新学习率
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()

            epoch_time = time.time() - epoch_start

            # 记录日志
            self.logger.info(
                f'Epoch {epoch+1}/{self.config.epochs} [{epoch_time:.1f}s] - '
                f'Train: {train_loss:.3f}, {train_acc:.2f}% | '
                f'Val: {val_loss:.3f}, {val_acc:.2f}% | '
                f'F1: {f1:.3f}'
            )

            # 记录学习率
            self.writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], epoch)
            self.writer.add_scalar('Time/EpochTime', epoch_time, epoch)

            # Wandb记录（如果启用）
            if hasattr(self.config, 'use_wandb') and self.config.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_f1': f1,
                    'lr': optimizer.param_groups[0]['lr'],
                    'epoch_time': epoch_time
                })

            # 保存最佳模型
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                epochs_no_improve = 0
                self.save_checkpoint(model, optimizer, scheduler, epoch, val_acc, is_best)
                self.logger.info(f'New best validation accuracy: {best_val_acc:.2f}%')
            else:
                epochs_no_improve += 1

            # 定期保存
            if (epoch + 1) % self.config.save_freq == 0:
                self.save_checkpoint(model, optimizer, scheduler, epoch, val_acc, False)

            # 早停
            if self.config.use_early_stopping and epochs_no_improve >= self.config.early_stopping_patience:
                self.logger.info(f'Early stopping triggered after {epoch+1} epochs')
                break

        self.writer.close()
        if hasattr(self.config, 'use_wandb') and self.config.use_wandb:
            wandb.finish()

        self.logger.info(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')

    def save_checkpoint(self, model, optimizer, scheduler, epoch, val_acc, is_best):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_acc': val_acc,
            'config': self.config.__dict__
        }

        # 保存最新
        torch.save(checkpoint, os.path.join(self.config.output_dir, 'latest_checkpoint.pth'))

        # 保存最佳
        if is_best:
            torch.save(checkpoint, os.path.join(self.config.output_dir, 'best_checkpoint.pth'))

        # 定期保存
        if (epoch + 1) % self.config.save_freq == 0:
            torch.save(checkpoint, os.path.join(self.config.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))


def main():
    """主函数"""
    config = Config()

    # 4090特定优化
    config.use_wandb = False  # 可选：设置为True使用wandb

    trainer = RTX4090Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()