"""
检查数据集的 token 长度分布，评估 max_length 是否足够
"""

import json
from pathlib import Path
from transformers import AutoTokenizer

def check_token_lengths(dataset_path: str = "./converted_dataset.json", max_length: int = 1024):
    """检查数据集的 token 长度分布"""
    
    # 加载 tokenizer
    tokenizer_path = Path("./tokenizers/Qwen_Qwen3-8B")
    if not tokenizer_path.exists():
        print(f"Tokenizer not found at {tokenizer_path}")
        return
    
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    
    # 加载数据
    print(f"\nLoading dataset from {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # 采样检查（如果数据太多，只检查前 5000 个）
    sample_size = min(5000, len(data))
    queries = [d['query'] for d in data[:sample_size]]
    tool_descs = [d['tool_description'] for d in data[:sample_size]]
    
    # 转换为 tokens
    print("\nConverting to tokens...")
    query_tokens = []
    tool_tokens = []
    
    for q, t in zip(queries, tool_descs):
        query_tokens.append(len(tokenizer.encode(q, add_special_tokens=False)))
        tool_tokens.append(len(tokenizer.encode(t, add_special_tokens=False)))
    
    # 统计信息
    def print_stats(name, tokens, max_len):
        sorted_tokens = sorted(tokens)
        n = len(tokens)
        
        print(f"\n{name} token lengths:")
        print(f"  min={min(tokens)}, max={max(tokens)}, avg={sum(tokens)//n}")
        print(f"  50th percentile (median): {sorted_tokens[n//2]}")
        print(f"  90th percentile: {sorted_tokens[int(n*0.9)]}")
        print(f"  95th percentile: {sorted_tokens[int(n*0.95)]}")
        print(f"  99th percentile: {sorted_tokens[int(n*0.99)]}")
        
        # 覆盖率
        within_limit = sum(1 for t in tokens if t <= max_len)
        coverage = within_limit / n * 100
        truncated = n - within_limit
        
        print(f"  Coverage with max_length={max_len}: {coverage:.1f}% ({within_limit}/{n})")
        print(f"  Truncated samples: {truncated} ({100-coverage:.1f}%)")
        
        if truncated > 0:
            print(f"  ⚠️  Warning: {truncated} samples will be truncated!")
            if truncated / n > 0.05:
                print(f"  ⚠️  More than 5% truncated! Consider increasing max_length")
        
        return coverage, truncated
    
    query_coverage, query_truncated = print_stats("Query", query_tokens, max_length)
    tool_coverage, tool_truncated = print_stats("Tool Description", tool_tokens, max_length)
    
    # 建议
    print("\n" + "="*60)
    print("Recommendations:")
    print("="*60)
    
    if query_coverage >= 99 and tool_coverage >= 99:
        print(f"✅ max_length={max_length} is sufficient!")
        print(f"   Query coverage: {query_coverage:.1f}%")
        print(f"   Tool coverage: {tool_coverage:.1f}%")
    elif query_coverage >= 95 and tool_coverage >= 95:
        print(f"⚠️  max_length={max_length} is mostly sufficient, but some truncation occurs")
        print(f"   Query truncated: {query_truncated} samples ({100-query_coverage:.1f}%)")
        print(f"   Tool truncated: {tool_truncated} samples ({100-tool_coverage:.1f}%)")
        print(f"   Consider: max_length={max_length * 2} for better coverage")
    else:
        print(f"❌ max_length={max_length} may be too small!")
        print(f"   Query truncated: {query_truncated} samples ({100-query_coverage:.1f}%)")
        print(f"   Tool truncated: {tool_truncated} samples ({100-tool_coverage:.1f}%)")
        print(f"   Recommended: max_length={max(max(query_tokens), max(tool_tokens)) + 64}")
    
    # 内存估算
    print("\n" + "="*60)
    print("Memory Estimation (batch_size=8):")
    print("="*60)
    
    # 简化估算：每个 token 约 4 bytes (fp32) 或 2 bytes (fp16)
    memory_fp32 = max_length * 768 * 8 * 4 / (1024**3)  # batch_size=8, d_model=768, fp32
    memory_fp16 = max_length * 768 * 8 * 2 / (1024**3)  # batch_size=8, d_model=768, fp16
    
    print(f"max_length={max_length}:")
    print(f"  fp32: ~{memory_fp32:.2f} GB per batch")
    print(f"  fp16: ~{memory_fp16:.2f} GB per batch")
    
    if max_length * 2 <= 2048:
        memory_fp32_2x = max_length * 2 * 768 * 8 * 4 / (1024**3)
        memory_fp16_2x = max_length * 2 * 768 * 8 * 2 / (1024**3)
        print(f"\nmax_length={max_length*2}:")
        print(f"  fp32: ~{memory_fp32_2x:.2f} GB per batch")
        print(f"  fp16: ~{memory_fp16_2x:.2f} GB per batch")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="./converted_dataset.json")
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()
    
    check_token_lengths(args.dataset, args.max_length)

