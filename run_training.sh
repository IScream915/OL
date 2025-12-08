#!/bin/bash

# OverLoCK RTX 4090 微调启动脚本

echo "=========================================="
echo "OverLoCK Model Fine-tuning on RTX 4090"
echo "=========================================="

# 检查GPU
nvidia-smi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 创建输出目录
mkdir -p outputs
mkdir -p logs
mkdir -p checkpoints

# 检查数据集
if [ ! -d "datasets/RSITMD" ]; then
    echo "错误：数据集 datasets/RSITMD 不存在！"
    exit 1
fi

# 检查预训练权重
if [ ! -f "weights/overlock_b_in1k_224.pth" ]; then
    echo "错误：预训练权重 weights/overlock_b_in1k_224.pth 不存在！"
    exit 1
fi

echo ""
echo "配置信息："
echo "- 模型：OverLoCK-B (95M参数)"
echo "- 批大小：32 (调整为避免OOM)"
echo "- 数据加载器：8个worker"
echo "- 图像尺寸：224x224"
echo "- 混合精度：启用"
echo "- 数据增强：强增强"
echo ""

# 选择运行模式
echo "选择运行模式："
echo "1. 快速测试 (5 epochs, batch_size=16)"
echo "2. 标准训练 (100 epochs, batch_size=32)"
echo "3. 高性能训练 (更大图像，更强增强)"
echo "4. 恢复训练"
read -p "请输入选择 (1-4): " choice

case $choice in
    1)
        echo "运行快速测试模式..."
        python train.py --epochs 5 --batch_size 16 --image_size 224
        ;;
    2)
        echo "运行标准训练模式..."
        python train_4090.py
        ;;
    3)
        echo "运行高性能训练模式..."
        # 修改配置为高性能模式
        sed -i 's/image_size = 224/image_size = 256/' config.py
        sed -i 's/batch_size = 32/batch_size = 16/' config.py
        python train_4090.py
        ;;
    4)
        read -p "请输入检查点路径: " checkpoint_path
        if [ -f "$checkpoint_path" ]; then
            echo "恢复训练从: $checkpoint_path"
            sed -i "s|resume = None|resume = '$checkpoint_path'|" config.py
            python train_4090.py
        else
            echo "错误：检查点文件不存在！"
            exit 1
        fi
        ;;
    *)
        echo "无效选择！"
        exit 1
        ;;
esac

echo ""
echo "训练完成！"
echo "最佳模型保存在: outputs/best_checkpoint.pth"
echo "训练日志保存在: outputs/"
echo ""
echo "运行以下命令评估模型："
echo "python evaluate.py"