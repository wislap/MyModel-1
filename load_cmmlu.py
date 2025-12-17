"""
使用 ModelScope 加载意图标注数据集

数据集：交互场景中的句子意图标注数据
数据集链接: https://www.modelscope.cn/datasets/DatatangBeijing/47811Sentence-IntentionAnnotationDataInInteractiveScenes/quickstart

这个数据集更适合 Router 模型训练，包含：
- 交互场景中的句子
- 意图标注
- 符合工具路由/意图识别任务需求
"""

def load_intention_dataset():
    """加载意图标注数据集"""
    # 尝试多个可能的数据集 ID
    dataset_ids = [
        "DatatangBeijing/47811Sentence-IntentionAnnotationDataInInteractiveScenes",
        "modelscope/DatatangBeijing_47811Sentence-IntentionAnnotationDataInInteractiveScenes",
        "47811Sentence-IntentionAnnotationDataInInteractiveScenes",
    ]
    
    dataset_url = "https://www.modelscope.cn/datasets/DatatangBeijing/47811Sentence-IntentionAnnotationDataInInteractiveScenes/quickstart"
    print(f"数据集页面: {dataset_url}")
    
    # 方法1: 使用 ModelScope Dataset 类（ModelScope 原生方式）
    try:
        from modelscope.hub.sdk import HubApi
        from modelscope import MsDataset
        print("\n尝试使用 ModelScope MsDataset 加载...")
        for dataset_id in dataset_ids:
            try:
                # 使用 ModelScope 的 MsDataset
                dataset = MsDataset.load(
                    dataset_id,
                    namespace='DatatangBeijing',
                    split='train',  # 或 'test', 'validation'
                )
                print(f"数据集加载成功!")
                return dataset
            except Exception as e:
                print(f"  {dataset_id} 失败: {e}")
                continue
    except ImportError:
        print("ModelScope MsDataset 不可用，尝试其他方法...")
    except Exception as e:
        print(f"ModelScope MsDataset 加载失败: {e}")
    
    # 方法2: 使用 datasets 库（HuggingFace 格式）
    try:
        from datasets import load_dataset
        print("\n尝试使用 datasets 库加载...")
        for dataset_id in dataset_ids:
            try:
                print(f"  尝试数据集 ID: {dataset_id}")
                dataset = load_dataset(
                    dataset_id,
                    cache_dir="./datasets"
                )
                print(f"数据集加载成功!")
                return dataset
            except Exception as e:
                print(f"  失败: {str(e)[:100]}...")
                continue
    except ImportError:
        print("datasets 库未安装")
    
    # 方法3: 使用 ModelScope snapshot_download（下载原始文件）
    try:
        from modelscope import snapshot_download
        print("\n尝试使用 ModelScope snapshot_download 下载原始文件...")
        for dataset_id in dataset_ids:
            try:
                dataset_path = snapshot_download(
                    dataset_id,
                    cache_dir="./datasets"
                )
                print(f"数据集文件已下载到: {dataset_path}")
                print("注意: 这是原始文件，需要手动解析")
                return dataset_path
            except Exception as e:
                print(f"  {dataset_id} 失败: {str(e)[:100]}...")
                continue
    except ImportError as e:
        print(f"ModelScope 导入错误: {e}")
    
    # 如果都失败了，提供详细说明和替代方案
    print("\n" + "=" * 60)
    print("⚠️  自动加载失败")
    print("=" * 60)
    print("可能的原因:")
    print("1. 数据集 ID 不正确或需要特殊权限")
    print("2. 数据集可能只在 ModelScope 网站提供，不在 Hub 上")
    print("3. 需要登录 ModelScope 账号或申请访问权限")
    print("\n建议操作:")
    print(f"1. 访问数据集页面: {dataset_url}")
    print("2. 查看页面上的 '快速开始' 或 'Quick Start' 部分")
    print("3. 复制页面上的正确数据集 ID 和加载代码")
    print("4. 如果需要，先登录 ModelScope 账号")
    print("5. 或者直接下载数据集文件到本地，然后手动加载")
    print("\n替代方案:")
    print("- 使用其他中文意图识别数据集（如 CLUE 子任务）")
    print("- 使用英文数据集（SNIPS、ATIS）进行预训练")
    print("=" * 60)
    return None


def show_dataset_info():
    """显示数据集信息"""
    print("=" * 60)
    print("意图标注数据集信息")
    print("=" * 60)
    print("数据集名称: 交互场景中的句子意图标注数据")
    print("数据集 ID: DatatangBeijing/47811Sentence-IntentionAnnotationDataInInteractiveScenes")
    print("数据集页面: https://www.modelscope.cn/datasets/DatatangBeijing/47811Sentence-IntentionAnnotationDataInInteractiveScenes/quickstart")
    print("\n数据集特点:")
    print("- 交互场景中的句子意图标注")
    print("- 适合 Router 模型训练（意图识别/工具路由）")
    print("- 包含句子和对应的意图标签")
    print("- 符合 1.md 中描述的 Router 任务需求")
    print("\n与 Router 任务的匹配度:")
    print("✓ 意图识别任务")
    print("✓ 交互场景数据")
    print("✓ 可用于训练工具路由模型")
    print("\n访问方式:")
    print("1. 直接在浏览器中打开数据集页面查看详情")
    print("2. 使用代码加载数据集进行训练")
    print("=" * 60)


def test_alternative_datasets():
    """测试替代数据集是否可用"""
    print("\n" + "=" * 60)
    print("🧪 测试替代数据集（适合 Router 训练）")
    print("=" * 60)
    
    alternative_datasets = [
        ("snips_built_in_intents", "SNIPS 意图识别数据集"),
        ("atis", "ATIS 航班信息查询意图"),
        ("multi_woz_v22", "MultiWOZ 多轮对话数据集"),
    ]
    
    from datasets import load_dataset
    
    for dataset_id, description in alternative_datasets:
        try:
            print(f"\n测试: {description} ({dataset_id})")
            dataset = load_dataset(dataset_id, cache_dir="./datasets")
            print(f"✅ 加载成功!")
            if hasattr(dataset, 'keys'):
                print(f"   可用分割: {list(dataset.keys())}")
                for split in dataset.keys():
                    if len(dataset[split]) > 0:
                        print(f"   {split} 大小: {len(dataset[split])}")
                        # 显示一个示例
                        sample = dataset[split][0]
                        print(f"   示例字段: {list(sample.keys())}")
                        break
            return dataset_id, dataset
        except Exception as e:
            print(f"❌ 加载失败: {str(e)[:100]}")
            continue
    
    print("\n" + "=" * 60)
    print("💡 替代数据集建议")
    print("=" * 60)
    print("\n1. 英文意图识别数据集（可直接使用）:")
    print("   - SNIPS: 意图识别数据集")
    print("   - ATIS: 航班信息查询意图")
    print("   - MultiWOZ: 多轮对话数据集")
    print("\n2. 中文数据集（需要查找）:")
    print("   - CLUE 子任务（中文语言理解评估）")
    print("   - 中文对话数据集")
    print("\n3. 使用示例:")
    print("   from datasets import load_dataset")
    print("   dataset = load_dataset('snips_built_in_intents')")
    print("=" * 60)
    return None, None


def suggest_alternative_datasets():
    """建议替代数据集"""
    print("\n" + "=" * 60)
    print("💡 替代数据集建议（适合 Router 训练）")
    print("=" * 60)
    print("\n1. 英文意图识别数据集（可直接使用）:")
    print("   - SNIPS: 意图识别数据集")
    print("   - ATIS: 航班信息查询意图")
    print("   - MultiWOZ: 多轮对话数据集")
    print("\n2. 中文数据集（需要查找）:")
    print("   - CLUE 子任务（中文语言理解评估）")
    print("   - 中文对话数据集")
    print("\n3. 使用示例:")
    print("   from datasets import load_dataset")
    print("   dataset = load_dataset('snips_built_in_intents')")
    print("=" * 60)


if __name__ == "__main__":
    show_dataset_info()
    print("\n")
    dataset = load_intention_dataset()
    
    if dataset:
        print("\n✅ 数据集加载成功!")
        if hasattr(dataset, 'keys'):
            print(f"可用数据集分割: {list(dataset.keys())}")
            # 显示每个分割的样本
            for split in dataset.keys():
                if len(dataset[split]) > 0:
                    print(f"\n{split} 数据集示例 (前3个样本):")
                    for i, sample in enumerate(dataset[split][:3]):
                        print(f"\n样本 {i+1}:")
                        print(sample)
                    print(f"\n{split} 数据集大小: {len(dataset[split])}")
                    break
        elif isinstance(dataset, dict):
            print(f"数据集结构: {list(dataset.keys())}")
            for key, value in dataset.items():
                if hasattr(value, '__len__'):
                    print(f"{key}: {len(value)} 个样本")
                    if len(value) > 0:
                        print(f"示例: {value[0]}")
                        break
        elif isinstance(dataset, (str, Path)):
            print(f"数据集文件路径: {dataset}")
            print("请手动解析数据集文件")
    else:
        # 测试替代数据集
        alt_dataset_id, alt_dataset = test_alternative_datasets()
        if alt_dataset:
            print(f"\n" + "=" * 60)
            print(f"✅ 找到可用的替代数据集: {alt_dataset_id}")
            print("=" * 60)
            print("可以使用此数据集进行 Router 模型训练")
            print("\n数据集信息:")
            if hasattr(alt_dataset, 'keys'):
                for split in alt_dataset.keys():
                    if len(alt_dataset[split]) > 0:
                        print(f"\n{split} 数据集:")
                        print(f"  大小: {len(alt_dataset[split])} 个样本")
                        sample = alt_dataset[split][0]
                        print(f"  字段: {list(sample.keys())}")
                        print(f"  示例:")
                        for key, value in sample.items():
                            print(f"    {key}: {value}")
                        break
            print("\n使用方式:")
            print(f"  from datasets import load_dataset")
            print(f"  dataset = load_dataset('{alt_dataset_id}')")
            print("=" * 60)

