"""
将 Hermes Reasoning Tool Use 数据集转换为工具匹配格式

改进：
1. 简化 conversation：只提取用户查询（第一条用户消息）
2. 合并 task 和 conversation：将 task 和用户查询合并作为 query
3. tools 转换为标准 JSON 格式（而不是字符串）
4. 保留有用的元数据字段

使用方法:
    python convert_hermes_dataset.py --output converted_dataset.json
"""

import json
import argparse
from datasets import load_dataset
from typing import List, Dict, Any, Optional
import re


def extract_tool_calls_from_content(content: str) -> List[Dict[str, Any]]:
    """
    从助手回复中提取工具调用信息
    
    Args:
        content: 助手回复内容，可能包含 <tool_call> 标签
        
    Returns:
        工具调用列表
    """
    tool_calls = []
    
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


def parse_tools_to_json(tools_input: Any) -> List[Dict[str, Any]]:
    """
    将工具 schema 转换为标准 JSON 格式
    
    Args:
        tools_input: 工具 schema（可能是字符串、列表或字典）
        
    Returns:
        标准化的工具列表（JSON 格式）
    """
    if tools_input is None:
        return []
    
    # 如果是字符串，尝试解析 JSON
    if isinstance(tools_input, str):
        try:
            tools = json.loads(tools_input)
        except json.JSONDecodeError:
            # 如果解析失败，返回空列表
            return []
    else:
        tools = tools_input
    
    # 确保是列表格式
    if isinstance(tools, dict):
        # 如果是单个工具字典，转换为列表
        tools = [tools]
    elif not isinstance(tools, list):
        return []
    
    # 标准化每个工具的结构
    normalized_tools = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        
        # 标准化工具结构
        normalized_tool = {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {}),
        }
        
        # 确保 parameters 是标准格式
        if isinstance(normalized_tool["parameters"], str):
            try:
                normalized_tool["parameters"] = json.loads(normalized_tool["parameters"])
            except json.JSONDecodeError:
                normalized_tool["parameters"] = {}
        
        # 添加 required 字段（如果存在）
        if "required" in tool:
            normalized_tool["required"] = tool["required"]
        elif "parameters" in normalized_tool and isinstance(normalized_tool["parameters"], dict):
            # 从 parameters 中提取 required
            required = normalized_tool["parameters"].get("required", [])
            if required:
                normalized_tool["required"] = required
        
        normalized_tools.append(normalized_tool)
    
    return normalized_tools


def extract_user_query_simplified(conversations: List[Dict[str, Any]]) -> str:
    """
    从对话中提取用户查询（简化版：只取第一条用户消息）
    
    Args:
        conversations: ShareGPT 格式的对话列表
        
    Returns:
        用户查询文本（第一条用户消息）
    """
    for msg in conversations:
        # 检查不同的消息格式
        role = msg.get("from") or msg.get("role", "")
        content = msg.get("content", "") or msg.get("text", "") or msg.get("value", "")
        
        if role in ["human", "user", "Human", "User"] and content:
            # 只返回第一条用户消息（简化）
            return content.strip()
    
    return ""


def combine_query_and_task(user_query: str, task: Optional[str]) -> str:
    """
    合并用户查询和任务描述
    
    Args:
        user_query: 用户查询
        task: 任务描述（可选）
        
    Returns:
        合并后的查询文本
    """
    if not task or not task.strip():
        return user_query
    
    # 如果 task 和 user_query 相似或相同，只返回 user_query
    if task.strip() == user_query.strip():
        return user_query
    
    # 合并：task + 用户查询
    combined = f"{task.strip()}\n\n{user_query.strip()}"
    return combined.strip()


def convert_hermes_to_tool_matching(
    dataset_name: str = "interstellarninja/hermes_reasoning_tool_use",
    split: str = "train",
    max_samples: int = None,
    include_negative_samples: bool = True,
    include_metadata: bool = True,
) -> List[Dict[str, Any]]:
    """
    将 Hermes 数据集转换为工具匹配格式
    
    改进：
    1. 简化 conversation：只提取第一条用户消息
    2. 合并 task 和 conversation 作为 query
    3. tools 转换为标准 JSON 格式
    4. 保留有用的元数据（scenario_category, source 等）
    
    Args:
        dataset_name: 数据集名称
        split: 数据集分割
        max_samples: 最大样本数（用于测试）
        include_negative_samples: 是否包含负样本（未使用的工具）
        include_metadata: 是否包含元数据字段
        
    Returns:
        转换后的数据列表
    """
    print(f"加载数据集: {dataset_name}")
    ds = load_dataset(dataset_name, split=split, cache_dir="./datasets")
    
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    
    converted_data = []
    skipped = 0
    
    for idx, item in enumerate(ds):
        if (idx + 1) % 1000 == 0:
            print(f"处理进度: {idx + 1}/{len(ds)}")
        
        # 提取用户查询（简化：只取第一条用户消息）
        conversations = item.get("conversations", [])
        if not isinstance(conversations, list):
            skipped += 1
            continue
        
        user_query = extract_user_query_simplified(conversations)
        if not user_query:
            skipped += 1
            continue
        
        # 获取 task 并合并
        task = item.get("task", "")
        query = combine_query_and_task(user_query, task)
        
        # 解析工具 schema 为标准 JSON 格式
        tools_input = item.get("tools", "")
        tools = parse_tools_to_json(tools_input)
        
        if not tools:
            skipped += 1
            continue
        
        # 提取工具调用
        tool_calls = []
        for msg in conversations:
            role = msg.get("from") or msg.get("role", "")
            content = msg.get("content", "") or msg.get("text", "") or msg.get("value", "")
            if role in ["gpt", "assistant", "Assistant"] and content:
                calls = extract_tool_calls_from_content(content)
                tool_calls.extend(calls)
        
        # 获取被调用的工具名称
        called_tool_names = set()
        for call in tool_calls:
            tool_name = call.get("name", "")
            if tool_name:
                called_tool_names.add(tool_name)
        
        # 获取元数据
        metadata = {}
        if include_metadata:
            if "scenario_category" in item:
                metadata["scenario_category"] = item["scenario_category"]
            if "source" in item:
                metadata["source"] = item["source"]
            if "category" in item:
                metadata["category"] = item["category"]
        
        # 为每个工具创建样本
        for tool in tools:
            tool_name = tool.get("name", "")
            tool_desc = tool.get("description", "")
            
            if not tool_name:
                continue
            
            # 检查工具是否被调用
            is_used = tool_name in called_tool_names
            
            # 创建样本
            sample = {
                "query": query,
                "tool_name": tool_name,
                "tool_description": tool_desc or tool_name,
                "tool_schema": tool,  # 完整的工具 schema（标准 JSON 格式）
                "label": 1 if is_used else 0
            }
            
            # 添加元数据
            if metadata:
                sample["metadata"] = metadata
            
            # 创建正样本（工具被调用）
            if is_used:
                converted_data.append(sample)
            # 创建负样本（工具未被调用）
            elif include_negative_samples:
                # 对于 relevance 场景，所有工具都是负样本
                scenario = item.get("scenario_category", "")
                if scenario == "relevance" or not called_tool_names:
                    converted_data.append(sample)
    
    print(f"\n转换完成:")
    print(f"  - 总样本数: {len(ds)}")
    print(f"  - 跳过样本: {skipped}")
    print(f"  - 转换后样本: {len(converted_data)}")
    print(f"  - 正样本: {sum(1 for d in converted_data if d['label'] == 1)}")
    print(f"  - 负样本: {sum(1 for d in converted_data if d['label'] == 0)}")
    
    return converted_data


def main():
    parser = argparse.ArgumentParser(description="转换 Hermes 数据集为工具匹配格式")
    parser.add_argument(
        "--dataset",
        type=str,
        default="interstellarninja/hermes_reasoning_tool_use",
        help="数据集名称"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="数据集分割"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="converted_hermes_dataset.json",
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
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="不包含元数据字段"
    )
    
    args = parser.parse_args()
    
    # 转换数据
    converted_data = convert_hermes_to_tool_matching(
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        include_negative_samples=not args.no_negative_samples,
        include_metadata=not args.no_metadata,
    )
    
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
            if 'tool_schema' in sample:
                print(f"  Tool Schema (keys): {list(sample['tool_schema'].keys())}")


if __name__ == "__main__":
    main()

