import json
import os

# 检查所有数据集的标签
datasets = ['RSITMD', 'RSICD', 'NWPU', 'AID']

for dataset in datasets:
    print(f"\n=== 检查 {dataset} 数据集 ===")

    # 检查类别映射文件
    class_mapping_path = f'datasets/{dataset}/class_to_idx.json'
    if os.path.exists(class_mapping_path):
        with open(class_mapping_path, 'r') as f:
            class_to_idx = json.load(f)
        print(f"类别数: {len(class_to_idx)}")
        print(f"类别映射示例: {list(class_to_idx.items())[:5]}")
    else:
        print(f"警告：找不到 {class_mapping_path}")
        continue

    # 检查训练集标签
    train_split_path = f'datasets/{dataset}/train_split.jsonl'
    if os.path.exists(train_split_path):
        labels = []
        with open(train_split_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                labels.append(data['label'])

        min_label = min(labels)
        max_label = max(labels)
        print(f"训练集标签范围: {min_label} - {max_label}")

        if max_label >= len(class_to_idx):
            print(f"⚠️  警告：标签 {max_label} 超出类别范围 [0, {len(class_to_idx)-1}]")
        else:
            print("✅ 标签范围正常")
    else:
        print(f"警告：找不到 {train_split_path}")

    # 检查验证集标签
    val_split_path = f'datasets/{dataset}/val_split.jsonl'
    if os.path.exists(val_split_path):
        labels = []
        with open(val_split_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                labels.append(data['label'])

        min_label = min(labels)
        max_label = max(labels)
        print(f"验证集标签范围: {min_label} - {max_label}")

        if max_label >= len(class_to_idx):
            print(f"⚠️  警告：标签 {max_label} 超出类别范围 [0, {len(class_to_idx)-1}]")
        else:
            print("✅ 标签范围正常")
    else:
        print(f"警告：找不到 {val_split_path}")
