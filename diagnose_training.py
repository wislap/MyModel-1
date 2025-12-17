#!/usr/bin/env python3
"""
训练诊断脚本 - 检查数据质量、损失计算、模型初始化等问题
"""
import json
from pathlib import Path
from datasets import load_dataset
from collections import Counter
import numpy as np

def diagnose_dataset(dataset_path: str):
    """诊断数据集质量"""
    print("=" * 60)
    print("数据集诊断")
    print("=" * 60)
    
    # 加载数据集
    if Path(dataset_path).exists():
        dataset = load_dataset("json", data_files=dataset_path, split="train")
    else:
        dataset = load_dataset(dataset_path, split="train")
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 检查标签分布
    labels = [item.get("label", 0) for item in dataset]
    label_counts = Counter(labels)
    print(f"\n标签分布:")
    for label, count in sorted(label_counts.items()):
        pct = count / len(labels) * 100
        print(f"  Label {label}: {count} ({pct:.1f}%)")
    
    # 检查标签平衡性
    if len(label_counts) == 2:
        pos_count = label_counts.get(1, 0)
        neg_count = label_counts.get(0, 0)
        if pos_count > 0 and neg_count > 0:
            ratio = max(pos_count, neg_count) / min(pos_count, neg_count)
            print(f"\n标签平衡性: {ratio:.2f}:1")
            if ratio > 10:
                print("  ⚠️  标签严重不平衡！这可能导致训练困难")
            elif ratio > 5:
                print("  ⚠️  标签不平衡，建议使用类别权重")
            else:
                print("  ✅ 标签相对平衡")
    
    # 检查数据质量
    print(f"\n数据质量检查:")
    empty_queries = sum(1 for item in dataset if not item.get("query", "").strip())
    empty_tools = sum(1 for item in dataset if not item.get("tool_description", "").strip() and not item.get("tool_name", "").strip())
    
    print(f"  空查询: {empty_queries} ({empty_queries/len(dataset)*100:.1f}%)")
    print(f"  空工具描述: {empty_tools} ({empty_tools/len(dataset)*100:.1f}%)")
    
    # 检查查询和工具描述长度
    query_lengths = [len(item.get("query", "")) for item in dataset]
    tool_lengths = [len(item.get("tool_description", "") or item.get("tool_name", "")) for item in dataset]
    
    print(f"\n查询长度统计:")
    print(f"  平均: {np.mean(query_lengths):.1f}")
    print(f"  中位数: {np.median(query_lengths):.1f}")
    print(f"  最小: {np.min(query_lengths)}")
    print(f"  最大: {np.max(query_lengths)}")
    
    print(f"\n工具描述长度统计:")
    print(f"  平均: {np.mean(tool_lengths):.1f}")
    print(f"  中位数: {np.median(tool_lengths):.1f}")
    print(f"  最小: {np.min(tool_lengths)}")
    print(f"  最大: {np.max(tool_lengths)}")
    
    # 检查样本示例
    print(f"\n样本示例（前3个）:")
    for i in range(min(3, len(dataset))):
        item = dataset[i]
        print(f"\n  样本 {i+1}:")
        print(f"    Query: {item.get('query', '')[:100]}...")
        print(f"    Tool: {item.get('tool_description', '')[:100] or item.get('tool_name', '')[:100]}...")
        print(f"    Label: {item.get('label', 0)}")

def diagnose_loss_calculation():
    """诊断损失计算"""
    print("\n" + "=" * 60)
    print("损失计算诊断")
    print("=" * 60)
    
    print("\n当前损失函数: Binary Cross Entropy with Logits")
    print("\n问题分析:")
    print("  1. BCE with logits 期望 logits 范围: (-∞, +∞)")
    print("  2. 当前相似度分数范围: [-1, 1] (归一化后的点积)")
    print("  3. 这可能导致梯度信号弱，训练困难")
    
    print("\n建议:")
    print("  1. 使用对比学习损失（InfoNCE/Contrastive Loss）")
    print("  2. 或者将相似度分数放大（乘以温度系数）")
    print("  3. 或者使用 margin-based loss")

def diagnose_model_init():
    """诊断模型初始化"""
    print("\n" + "=" * 60)
    print("模型初始化诊断")
    print("=" * 60)
    
    print("\n当前配置:")
    print("  init_from_pretrained: true")
    print("  问题: 从预训练模型加载可能导致初始化不当")
    print("  建议: 使用随机初始化 (init_from_pretrained=false)")

if __name__ == "__main__":
    import sys
    
    # 从config.json读取数据集路径
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        dataset_path = config.get("dataset", {}).get("name", "./converted_dataset.json")
    else:
        dataset_path = "./converted_dataset.json"
    
    diagnose_dataset(dataset_path)
    diagnose_loss_calculation()
    diagnose_model_init()
    
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)

