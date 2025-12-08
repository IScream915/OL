import torch
import json

# 加载预训练权重文件
checkpoint = torch.load('weights/overlock_b_in1k_224.pth', map_location='cpu')

print("预训练权重文件信息：")
print(f"类型: {type(checkpoint)}")
print(f"键的列表: {list(checkpoint.keys())}")

if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    print("\n包含 state_dict")
elif 'model' in checkpoint:
    state_dict = checkpoint['model']
    print("\n包含 model 键")
else:
    state_dict = checkpoint
    print("\n直接包含权重")

print(f"\n权重参数数量: {len(state_dict)}")
print(f"\n前10个键名:")
for i, key in enumerate(state_dict.keys()):
    if i < 10:
        print(f"  {key}")

# 检查分类器层的权重
clf_keys = [k for k in state_dict.keys() if 'head' in k or 'fc' in k or 'classifier' in k]
print(f"\n分类器相关层: {clf_keys}")

# 检查是否有其他元数据
meta_keys = [k for k in checkpoint.keys() if k not in ['state_dict', 'model']]
if meta_keys:
    print(f"\n其他元数据: {meta_keys}")
    for key in meta_keys:
        print(f"  {key}: {checkpoint[key]}")