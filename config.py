import os


class Config:
    """微调配置文件 - RTX 4090 优化版"""

    # 数据集配置
    data_dir = 'datasets/RSITMD'
    image_size = 224  # RTX 4090可以支持更大分辨率，可以尝试256或384
    batch_size = 32  # 调小batch_size以解决OOM问题
    num_workers = 8  # 120G内存可以支持更多worker
    pin_memory = True  # 加速数据传输到GPU
    persistent_workers = True  # 保持worker进程存活

    # 模型配置
    model_name = 'overlock_b'  # 可选: overlock_xt, overlock_t, overlock_s, overlock_b
    num_classes = 33  # RSITMD有33个类别
    pretrained_weights = 'weights/overlock_b_in1k_224.pth'

    # 训练配置
    epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-4
    min_lr = 1e-6
    warmup_epochs = 5
    use_warmup = True

    # 优化器配置
    optimizer = 'adamw'  # 可选: adam, adamw, sgd
    momentum = 0.9  # 仅SGD使用

    # 学习率调度器
    scheduler = 'cosine'  # 可选: cosine, step, multistep, plateau
    step_size = 30  # 仅step scheduler使用
    gamma = 0.1  # 仅step/multistep scheduler使用
    milestones = [30, 60, 80]  # 仅multistep scheduler使用
    patience = 10  # 仅plateau scheduler使用

    # 损失函数
    loss_function = 'crossentropy'  # 可选: crossentropy, focal, labelsmooth
    label_smoothing = 0.1  # 仅labelsmooth loss使用
    focal_alpha = 1.0  # 仅focal loss使用
    focal_gamma = 2.0  # 仅focal loss使用
    use_class_weights = True  # 是否使用类别权重处理不平衡

    # 正则化
    dropout = 0.0
    mixup_prob = 0.0
    cutmix_prob = 0.0

    # 梯度裁剪
    grad_clip_norm = 5.0
    use_grad_clip = True

    # 保存和日志
    output_dir = 'outputs'
    save_freq = 10  # 每隔多少epoch保存一次
    log_freq = 50  # 每隔多少batch记录一次日志
    eval_freq = 5  # 每隔多少epoch验证一次

    # 早停
    use_early_stopping = True
    early_stopping_patience = 20

    # 设备配置
    device = 'cuda' if os.system('nvidia-smi') == 0 else 'cpu'
    use_amp = True  # 自动混合精度
    distributed = False  # 是否使用分布式训练

    # 微调策略
    freeze_backbone = False  # 是否冻结骨干网络
    freeze_epochs = 0  # 冻结多少个epoch后开始训练所有层

    # 恢复训练
    resume = None  # 恢复训练的检查点路径

    # 其他
    seed = 42
    print_freq = 100  # 打印频率