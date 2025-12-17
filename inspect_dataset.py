"""查看数据集结构"""
from datasets import load_dataset
import json

dataset_name = "snips_built_in_intents"
print(f"加载数据集: {dataset_name}")
dataset = load_dataset(dataset_name, cache_dir="./datasets")

print("\n" + "=" * 60)
print("数据集基本信息")
print("=" * 60)
print(f"数据集分割: {list(dataset.keys())}")

for split_name in dataset.keys():
    split_data = dataset[split_name]
    print(f"\n{split_name} 分割:")
    print(f"  大小: {len(split_data)} 个样本")
    print(f"  字段: {list(split_data.features.keys())}")
    print(f"  字段类型:")
    for key, feature in split_data.features.items():
        print(f"    {key}: {feature}")

print("\n" + "=" * 60)
print("样本示例")
print("=" * 60)

# 显示前5个样本
train_data = dataset.get("train", list(dataset.values())[0])
for i in range(min(5, len(train_data))):
    print(f"\n样本 {i+1}:")
    sample = train_data[i]
    for key, value in sample.items():
        if isinstance(value, (list, dict)):
            print(f"  {key}: {json.dumps(value, ensure_ascii=False, indent=4)}")
        else:
            print(f"  {key}: {value}")

# 统计标签分布
if "label" in train_data.features:
    print("\n" + "=" * 60)
    print("标签分布统计")
    print("=" * 60)
    labels = [sample["label"] for sample in train_data]
    from collections import Counter
    label_counts = Counter(labels)
    print(f"总标签数: {len(label_counts)}")
    print("\n标签分布（按标签ID）:")
    for label, count in sorted(label_counts.items()):
        label_name = train_data.features["label"].names[label]
        print(f"  标签 {label} ({label_name}): {count} 个样本")
    
    print("\n标签名称列表:")
    for i, name in enumerate(train_data.features["label"].names):
        print(f"  {i}: {name}")

print("\n" + "=" * 60)
print("数据集结构总结")
print("=" * 60)
print("""
数据集结构:
├── 分割 (Splits)
│   └── train: 328 个样本
│
├── 字段 (Fields)
│   ├── text: string - 用户输入的文本（自然语言）
│   └── label: int - 意图标签ID (0-9)
│
└── 标签映射 (Label Mapping)
    0: ComparePlaces
    1: RequestRide
    2: GetWeather
    3: SearchPlace
    4: GetPlaceDetails
    5: ShareCurrentLocation
    6: GetTrafficInformation
    7: BookRestaurant
    8: GetDirections
    9: ShareETA

数据格式示例:
{
    "text": "Share my location with Hillary's sister",
    "label": 5  # 对应 ShareCurrentLocation
}

用途:
- text: 作为 Router 模型的输入（用户意图表达）
- label: 作为训练目标（需要调用的工具/意图类别）
""")

