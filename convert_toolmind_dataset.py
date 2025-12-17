"""
将 ToolMind 数据集转换为工具匹配格式

ToolMind 数据集包含：
- 多轮对话轨迹
- 工具调用信息
- 函数图结构

转换格式：
- query: 用户查询（从对话中提取）
- tool_description: 工具描述
- tool_name: 工具名称
- tool_schema: 完整的工具 schema
- label: 1=工具被调用，0=工具未被调用

使用方法:
    python convert_toolmind_dataset.py --output converted_dataset_toolmind.json
"""

import json
import argparse
from datasets import load_dataset
from typing import List, Dict, Any, Optional
import re
from tqdm import tqdm


def extract_user_query(messages: List[Dict[str, Any]]) -> str:
    """
    从消息列表中提取用户查询（第一条用户消息）
    
    Args:
        messages: 消息列表
        
    Returns:
        用户查询文本
    """
    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        
        if role in ["user", "human"] and content:
            return content.strip()
    
    return ""


def extract_tool_calls(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    从消息列表中提取工具调用
    
    Args:
        messages: 消息列表
        
    Returns:
        工具调用列表
    """
    tool_calls = []
    
    for msg in messages:
        role = msg.get("role", "").lower()
        
        if role in ["assistant", "gpt"]:
            # 检查是否有 tool_calls 字段
            if "tool_calls" in msg:
                tool_calls.extend(msg["tool_calls"])
            
            # 检查 content 中是否有工具调用标记
            content = msg.get("content", "")
            if content:
                # 匹配 <tool_call> 标签
                tool_call_pattern = r'<tool_call>\s*({.*?})\s*</tool_call>'
                matches = re.findall(tool_call_pattern, content, re.DOTALL)
                
                for match in matches:
                    try:
                        tool_call = json.loads(match)
                        tool_calls.append(tool_call)
                    except json.JSONDecodeError:
                        continue
    
    return tool_calls


def normalize_tool_schema(tool: Dict[str, Any]) -> Dict[str, Any]:
    """
    标准化工具 schema 格式
    
    Args:
        tool: 工具字典
        
    Returns:
        标准化的工具 schema
    """
    normalized = {
        "name": tool.get("name", ""),
        "description": tool.get("description", ""),
    }
    
    # 处理 parameters
    if "parameters" in tool:
        params = tool["parameters"]
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {}
        normalized["parameters"] = params
    else:
        normalized["parameters"] = {}
    
    # 处理 required 字段
    if "required" in tool:
        normalized["required"] = tool["required"]
    elif "parameters" in normalized and isinstance(normalized["parameters"], dict):
        required = normalized["parameters"].get("required", [])
        if required:
            normalized["required"] = required
    
    return normalized


def convert_toolmind_to_tool_matching(
    dataset_name: str = "Nanbeige/ToolMind",
    split: str = "graph_syn_datasets",
    max_samples: Optional[int] = None,
    include_negative_samples: bool = True,
) -> List[Dict[str, Any]]:
    """
    将 ToolMind 数据集转换为工具匹配格式
    
    Args:
        dataset_name: 数据集名称
        split: 数据集分割
        max_samples: 最大处理样本数（用于测试）
        include_negative_samples: 是否包含负样本（未使用的工具）
        
    Returns:
        转换后的数据列表
    """
    print(f"加载数据集: {dataset_name} (split: {split})")
    
    try:
        # 尝试加载数据集
        if max_samples:
            # 使用 streaming 模式加载，然后取前 max_samples 个
            ds = load_dataset(dataset_name, split=split, streaming=True)
            ds_list = []
            for i, item in enumerate(ds):
                if i >= max_samples:
                    break
                ds_list.append(item)
            ds = ds_list
        else:
            ds = load_dataset(dataset_name, split=split, cache_dir="./datasets")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        print("尝试使用 open_datasets split...")
        try:
            if max_samples:
                ds = load_dataset(dataset_name, split="open_datasets", streaming=True)
                ds_list = []
                for i, item in enumerate(ds):
                    if i >= max_samples:
                        break
                    ds_list.append(item)
                ds = ds_list
            else:
                ds = load_dataset(dataset_name, split="open_datasets", cache_dir="./datasets")
        except Exception as e2:
            print(f"加载 open_datasets 也失败: {e2}")
            return []
    
    converted_data = []
    skipped = 0
    
    # 使用 tqdm 显示进度
    iterator = tqdm(ds, desc="处理数据") if hasattr(ds, '__iter__') and not isinstance(ds, list) else ds
    
    for idx, item in enumerate(iterator):
        if isinstance(iterator, tqdm):
            iterator.set_postfix({"已转换": len(converted_data), "跳过": skipped})
        
        # 提取消息列表
        messages = item.get("messages", [])
        if not isinstance(messages, list) or not messages:
            skipped += 1
            continue
        
        # 提取用户查询
        user_query = extract_user_query(messages)
        if not user_query:
            skipped += 1
            continue
        
        # 提取工具列表
        tools = item.get("tools", [])
        if not isinstance(tools, list):
            # 尝试从其他字段获取工具
            if "functions" in item:
                tools = item["functions"]
            elif "functions_list" in item:
                tools = item["functions_list"]
            else:
                skipped += 1
                continue
        
        if not tools:
            skipped += 1
            continue
        
        # 提取工具调用
        tool_calls = extract_tool_calls(messages)
        called_tool_names = set()
        
        for call in tool_calls:
            # 处理不同的工具调用格式
            if isinstance(call, dict):
                tool_name = call.get("name") or call.get("function", {}).get("name", "")
                if tool_name:
                    called_tool_names.add(tool_name)
            elif isinstance(call, str):
                try:
                    call_dict = json.loads(call)
                    tool_name = call_dict.get("name", "")
                    if tool_name:
                        called_tool_names.add(tool_name)
                except:
                    pass
        
        # 标准化工具列表
        normalized_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                normalized_tool = normalize_tool_schema(tool)
                if normalized_tool["name"]:
                    normalized_tools.append(normalized_tool)
            elif isinstance(tool, str):
                try:
                    tool_dict = json.loads(tool)
                    normalized_tool = normalize_tool_schema(tool_dict)
                    if normalized_tool["name"]:
                        normalized_tools.append(normalized_tool)
                except:
                    continue
        
        if not normalized_tools:
            skipped += 1
            continue
        
        # 获取元数据
        metadata = {
            "source": "ToolMind",
            "split": split,
        }
        
        # 添加其他可能的元数据字段
        for key in ["domain", "category", "task_type", "trajectory_id"]:
            if key in item:
                metadata[key] = item[key]
        
        # 为每个工具创建样本
        for tool in normalized_tools:
            tool_name = tool.get("name", "")
            tool_desc = tool.get("description", "") or tool_name
            
            if not tool_name:
                continue
            
            # 检查工具是否被调用
            is_used = tool_name in called_tool_names
            
            # 创建样本
            sample = {
                "query": user_query,
                "tool_name": tool_name,
                "tool_description": tool_desc,
                "tool_schema": tool,
                "label": 1 if is_used else 0,
                "metadata": metadata,
            }
            
            # 添加正样本（工具被调用）
            if is_used:
                converted_data.append(sample)
            # 添加负样本（工具未被调用）
            elif include_negative_samples:
                # 如果没有任何工具被调用，所有工具都是负样本
                if not called_tool_names or len(called_tool_names) == 0:
                    converted_data.append(sample)
                # 或者随机选择一些负样本（避免负样本过多）
                elif len(converted_data) % 3 == 0:  # 每3个正样本添加1个负样本
                    converted_data.append(sample)
    
    print(f"\n转换完成:")
    print(f"  - 处理样本数: {len(ds) if hasattr(ds, '__len__') else 'N/A'}")
    print(f"  - 跳过样本: {skipped}")
    print(f"  - 转换后样本: {len(converted_data)}")
    print(f"  - 正样本: {sum(1 for d in converted_data if d['label'] == 1)}")
    print(f"  - 负样本: {sum(1 for d in converted_data if d['label'] == 0)}")
    
    return converted_data


def main():
    parser = argparse.ArgumentParser(description="转换 ToolMind 数据集为工具匹配格式")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Nanbeige/ToolMind",
        help="数据集名称"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="graph_syn_datasets",
        help="数据集分割 (graph_syn_datasets 或 open_datasets)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="converted_dataset_toolmind.json",
        help="输出文件路径"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大处理样本数（用于测试）"
    )
    parser.add_argument(
        "--no-negative-samples",
        action="store_true",
        help="不包含负样本"
    )
    
    args = parser.parse_args()
    
    # 转换数据
    converted_data = convert_toolmind_to_tool_matching(
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        include_negative_samples=not args.no_negative_samples,
    )
    
    if not converted_data:
        print("错误: 没有转换任何数据")
        return
    
    # 保存为 JSON
    print(f"\n保存到: {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print("完成！")
    
    # 显示示例
    if converted_data:
        print("\n示例数据:")
        for i, sample in enumerate(converted_data[:3]):
            print(f"\n样本 {i+1}:")
            print(f"  Query: {sample['query'][:150]}...")
            print(f"  Tool: {sample['tool_name']}")
            print(f"  Tool Description: {sample['tool_description'][:100]}...")
            print(f"  Label: {sample['label']}")
            if 'metadata' in sample:
                print(f"  Metadata: {sample['metadata']}")


if __name__ == "__main__":
    main()

