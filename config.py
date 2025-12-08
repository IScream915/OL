import os


class Config:
    """微调配置文件 - RTX 4090 优化版"""

    def __init__(self):
        # 数据集配置
        self.data_dir = 'datasets/RSITMD'
        self.dataset_name = os.path.basename(self.data_dir)  # 从data_dir提取数据集名称
        self.image_size = 224  # RTX 4090可以支持更大分辨率，可以尝试256或384
        self.batch_size = 32  # 调小batch_size以解决OOM问题
        self.gradient_accumulation_steps = 2  # 梯度累积步数，有效batch_size = batch_size * gradient_accumulation_steps
        self.num_workers = 8  # 120G内存可以支持更多worker
        self.pin_memory = True  # 加速数据传输到GPU
        self.persistent_workers = True  # 保持worker进程存活

        # 根据数据集名称动态设置输出目录
        self.output_dir = f'outputs/{self.dataset_name}'

        # 模型配置
        self.model_name = 'overlock_b'  # 可选: overlock_xt, overlock_t, overlock_s, overlock_b
        self.num_classes = 33  # RSITMD有33个类别
        self.pretrained_weights = 'weights/overlock_b_in1k_224.pth'

        # 训练配置
        self.epochs = 100
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.min_lr = 1e-6
        self.warmup_epochs = 5
        self.use_warmup = True

        # 优化器配置
        self.optimizer = 'adamw'  # 可选: adam, adamw, sgd
        self.momentum = 0.9  # 仅SGD使用

        # 学习率调度器
        self.scheduler = 'cosine'  # 可选: cosine, step, multistep, plateau
        self.step_size = 30  # 仅step scheduler使用
        self.gamma = 0.1  # 仅step/multistep scheduler使用
        self.milestones = [30, 60, 80]  # 仅multistep scheduler使用
        self.patience = 10  # 仅plateau scheduler使用

        # 损失函数
        self.loss_function = 'crossentropy'  # 可选: crossentropy, focal, labelsmooth
        self.label_smoothing = 0.1  # 仅labelsmooth loss使用
        self.focal_alpha = 1.0  # 仅focal loss使用
        self.focal_gamma = 2.0  # 仅focal loss使用
        self.use_class_weights = True  # 是否使用类别权重处理不平衡

        # 正则化
        self.dropout = 0.0
        self.mixup_prob = 0.0
        self.cutmix_prob = 0.0

        # 梯度裁剪
        self.grad_clip_norm = 5.0
        self.use_grad_clip = True

        # 保存和日志
        self.save_freq = 10  # 每隔多少epoch保存一次
        self.log_freq = 50  # 每隔多少batch记录一次日志
        self.eval_freq = 5  # 每隔多少epoch验证一次

        # 早停
        self.use_early_stopping = True
        self.early_stopping_patience = 20

        # 设备配置
        self.device = 'cuda' if os.system('nvidia-smi') == 0 else 'cpu'
        self.use_amp = True  # 自动混合精度
        self.distributed = False  # 是否使用分布式训练

        # 微调策略
        self.freeze_backbone = False  # 是否冻结骨干网络
        self.freeze_epochs = 0  # 冻结多少个epoch后开始训练所有层

        # 恢复训练
        self.resume = None  # 恢复训练的检查点路径

        # 其他
        self.seed = 42
        self.print_freq = 100  # 打印频率