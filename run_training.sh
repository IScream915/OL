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

# 检查可用的数据集
echo "检测可用的数据集..."
available_datasets=()
for dataset_dir in datasets/*/; do
    if [ -d "$dataset_dir" ]; then
        dataset_name=$(basename "$dataset_dir")
        available_datasets+=("$dataset_name")
        echo "- $dataset_name"
    fi
done

if [ ${#available_datasets[@]} -eq 0 ]; then
    echo "错误：datasets 目录下没有找到任何数据集！"
    exit 1
fi

# 选择数据集
echo ""
echo "请选择要训练的数据集："
for i in "${!available_datasets[@]}"; do
    echo "$((i+1)). ${available_datasets[i]}"
done
read -p "请输入选择 (1-${#available_datasets[@]}): " dataset_choice

if [[ $dataset_choice -ge 1 && $dataset_choice -le ${#available_datasets[@]} ]]; then
    selected_dataset="${available_datasets[$((dataset_choice-1))]}"
else
    echo "无效选择！"
    exit 1
fi

echo ""
echo "已选择数据集：$selected_dataset"

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
        python train.py --epochs 5 --batch_size 16 --image_size 224 --data_dir datasets/$selected_dataset
        ;;
    2)
        echo "运行标准训练模式..."
        python train_4090.py --data_dir datasets/$selected_dataset
        ;;
    3)
        echo "运行高性能训练模式..."
        python train_4090.py --data_dir datasets/$selected_dataset --image_size 256 --batch_size 16
        ;;
    4)
        read -p "请输入检查点路径: " checkpoint_path
        if [ -f "$checkpoint_path" ]; then
            echo "恢复训练从: $checkpoint_path"
            python train_4090.py --data_dir datasets/$selected_dataset --resume $checkpoint_path
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
echo "最佳模型保存在: outputs/$selected_dataset/best_checkpoint.pth"
echo "训练日志保存在: outputs/$selected_dataset/"
echo ""
echo "运行以下命令评估模型："
echo "python evaluate.py --data_dir datasets/$selected_dataset"