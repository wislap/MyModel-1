from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import re
import json
import hashlib
import argparse
from safetensors.torch import save_file, load_file
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from mamba_encoder import MambaEncoder


class TokenEmbedding(nn.Module):
    """
    基于 tokenizer 的 Token Embedding 层
    
    Args:
        tokenizer: transformers tokenizer 对象
        embedding_dim: embedding 维度
        vocab_size: 词汇表大小，如果为 None 则从 tokenizer 获取
        padding_idx: padding token 的索引，如果为 None 则从 tokenizer 获取
        init_from_pretrained: 是否从预训练模型初始化权重，如果为 None 则尝试加载
        cache_dir: safetensors 缓存目录，默认为 "./embeddings_cache"
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        embedding_dim: int = 768,
        vocab_size: Optional[int] = None,
        padding_idx: Optional[int] = None,
        init_from_pretrained: Optional[bool] = None,
        cache_dir: str = "./embeddings_cache",
    ):
        super().__init__()
        
        # 确保 tokenizer 有有效的 pad_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # 确定词汇表大小：使用 len(tokenizer) 而不是 vocab_size
        # 因为 len(tokenizer) 包含所有 special tokens，是 embedding 应该匹配的真实大小
        if vocab_size is None:
            vocab_size = len(tokenizer)
        
        # 确定 padding_idx
        if padding_idx is None:
            padding_idx = tokenizer.pad_token_id
        
        # 验证 padding_idx 是否在有效范围内
        if padding_idx is not None and padding_idx >= vocab_size:
            padding_idx = None
        
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.cache_dir = Path(cache_dir)
        self.model_name = self._get_model_name(tokenizer)
        
        # 创建 embedding 层
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx if padding_idx is not None and padding_idx < vocab_size else None
        )
        
        # 初始化权重
        self._initialize_weights(init_from_pretrained)
    
    def _get_model_name(self, tokenizer: AutoTokenizer) -> str:
        """从 tokenizer 获取模型名称，用于文件命名"""
        # 尝试从 tokenizer 的 name_or_path 获取
        if hasattr(tokenizer, 'name_or_path'):
            model_name = tokenizer.name_or_path
        else:
            model_name = "unknown_model"
        
        # 清理模型名称，移除特殊字符
        model_name = re.sub(r'[^\w\-_]', '_', model_name)
        model_name = model_name.replace('/', '_')
        return model_name
    
    def _get_tokenizer_hash(self, tokenizer: AutoTokenizer) -> str:
        """获取 tokenizer 的哈希值，用于缓存文件名"""
        # 使用 vocab 的哈希来标识 tokenizer 版本
        try:
            vocab = tokenizer.get_vocab()
            vocab_str = json.dumps(vocab, sort_keys=True)
            vocab_hash = hashlib.md5(vocab_str.encode()).hexdigest()[:8]
        except Exception:
            # 如果无法获取 vocab，使用 vocab_size 作为备选
            vocab_hash = f"vocab{len(tokenizer)}"
        
        return vocab_hash
    
    def _get_cache_path(self) -> Path:
        """获取当前模型的 safetensors 缓存路径
        
        包含模型名、embedding 维度、vocab 大小和 tokenizer 哈希，确保缓存唯一性
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        vocab_hash = self._get_tokenizer_hash(self.tokenizer)
        vocab_size = len(self.tokenizer)
        cache_file = self.cache_dir / (
            f"{self.model_name}_embedding_dim{self.embedding_dim}_"
            f"vocab{vocab_size}_{vocab_hash}.safetensors"
        )
        return cache_file
    
    def _initialize_weights(self, init_from_pretrained: Optional[bool] = None):
        """初始化 embedding 权重"""
        cache_path = self._get_cache_path()
        
        # 尝试从缓存加载
        if cache_path.exists():
            try:
                self.load_from_safetensors(cache_path)
                print(f"Loaded embedding weights from {cache_path}")
                return
            except Exception as e:
                print(f"Failed to load from cache: {e}, initializing new weights")
        
        # 如果指定从预训练模型初始化
        if init_from_pretrained is True:
            # 只有明确指定时才从预训练模型加载（会下载整个模型）
            self._init_from_pretrained_model()
        else:
            # 默认使用随机初始化，避免下载整个模型
            self._init_random()
        
        # 保存初始化后的权重
        self.save_to_safetensors(cache_path)
        print(f"Saved embedding weights to {cache_path}")
    
    def _init_from_pretrained_model(self):
        """从预训练模型初始化 embedding 权重（只加载 embedding 层，不加载整个模型）"""
        model_name = self.tokenizer.name_or_path if hasattr(self.tokenizer, 'name_or_path') else None
        if model_name is None:
            raise ValueError("Cannot determine model name from tokenizer")
        
        print(f"Loading embedding weights from pretrained model: {model_name}")
        print("Note: This will only load the embedding layer, not the full model.")
        
        try:
            # 尝试只加载 embedding 层，避免下载整个模型
            # 使用 get_input_embeddings() 方法，如果模型支持的话
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            
            # 尝试直接加载模型但只取 embedding 层
            # 注意：这仍然会下载模型，但我们可以立即释放
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            
            # 获取 embedding 层
            embedding_layer = model.get_input_embeddings()
            if embedding_layer is None:
                # 如果模型没有 get_input_embeddings，尝试从 state_dict 查找
                state_dict = model.state_dict()
                embedding_weight = None
                
                # 常见的 embedding 层名称
                possible_names = [
                    'embeddings.word_embeddings.weight',
                    'embed_tokens.weight',
                    'wte.weight',  # GPT style
                    'token_embeddings.weight',
                ]
                
                for name in possible_names:
                    if name in state_dict:
                        embedding_weight = state_dict[name]
                        print(f"Found embedding layer: {name}")
                        break
                
                if embedding_weight is None:
                    # 尝试查找包含 'embed' 的层
                    for name, weight in state_dict.items():
                        if 'embed' in name.lower() and len(weight.shape) == 2:
                            embedding_weight = weight
                            print(f"Found embedding layer: {name}")
                            break
                
                if embedding_weight is None:
                    raise ValueError("Could not find embedding layer in pretrained model")
            else:
                embedding_weight = embedding_layer.weight.data
            
            # 立即释放模型内存
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"Failed to load embedding from model: {e}")
            print("Falling back to loading full model (this will download the entire model)...")
            # 如果上述方法失败，回退到加载完整模型
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            embedding_layer = model.get_input_embeddings()
            embedding_weight = embedding_layer.weight.data if embedding_layer else None
            
            if embedding_weight is None:
                raise ValueError("Could not find embedding layer in pretrained model")
            
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 检查维度是否匹配
        pretrained_vocab_size, pretrained_dim = embedding_weight.shape
        
        if pretrained_dim != self.embedding_dim:
            print(f"Warning: Embedding dim mismatch. Pretrained: {pretrained_dim}, Required: {self.embedding_dim}")
            # 如果维度不匹配，使用线性投影或截断
            if pretrained_dim > self.embedding_dim:
                embedding_weight = embedding_weight[:, :self.embedding_dim]
            else:
                # 扩展维度，用零填充
                padding = torch.zeros(pretrained_vocab_size, self.embedding_dim - pretrained_dim)
                embedding_weight = torch.cat([embedding_weight, padding], dim=1)
        
        # 复制权重
        if pretrained_vocab_size >= self.vocab_size:
            self.embedding.weight.data[:] = embedding_weight[:self.vocab_size]
        else:
            self.embedding.weight.data[:pretrained_vocab_size] = embedding_weight
            # 剩余部分随机初始化
            nn.init.normal_(
                self.embedding.weight.data[pretrained_vocab_size:],
                mean=0.0,
                std=0.02
            )
    
    def _init_random(self):
        """随机初始化 embedding 权重"""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        # 如果有 padding_idx，将其设置为零
        if self.padding_idx is not None:
            self.embedding.weight.data[self.padding_idx].fill_(0.0)
    
    def save_to_safetensors(self, path: Optional[Path] = None):
        """
        保存 embedding 权重到 safetensors 文件
        
        Args:
            path: 保存路径，如果为 None 则使用默认缓存路径
        """
        if path is None:
            path = self._get_cache_path()
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存权重和元数据
        state_dict = {
            "embedding.weight": self.embedding.weight.data,
        }
        
        # 添加元数据（包含 tokenizer 信息，确保缓存唯一性）
        vocab_hash = self._get_tokenizer_hash(self.tokenizer)
        actual_vocab_size = len(self.tokenizer)
        metadata = {
            "model_name": self.model_name,
            "vocab_size": str(self.vocab_size),
            "actual_vocab_size": str(actual_vocab_size),
            "embedding_dim": str(self.embedding_dim),
            "padding_idx": str(self.padding_idx) if self.padding_idx is not None else "None",
            "vocab_hash": vocab_hash,
        }
        
        save_file(state_dict, str(path), metadata=metadata)
        print(f"Saved embedding to {path}")
    
    def load_from_safetensors(self, path: Path):
        """
        从 safetensors 文件加载 embedding 权重
        
        Args:
            path: safetensors 文件路径
        """
        if not path.exists():
            raise FileNotFoundError(f"Embedding file not found: {path}")
        
        state_dict = load_file(str(path))
        
        # 检查维度是否匹配
        loaded_weight = state_dict["embedding.weight"]
        loaded_vocab_size, loaded_dim = loaded_weight.shape
        
        if loaded_dim != self.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch. Loaded: {loaded_dim}, Required: {self.embedding_dim}"
            )
        
        # 复制权重
        if loaded_vocab_size >= self.vocab_size:
            self.embedding.weight.data[:] = loaded_weight[:self.vocab_size]
        else:
            self.embedding.weight.data[:loaded_vocab_size] = loaded_weight
            # 剩余部分随机初始化
            nn.init.normal_(
                self.embedding.weight.data[loaded_vocab_size:],
                mean=0.0,
                std=0.02
            )
        
        print(f"Loaded embedding from {path}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: token IDs，shape 为 (batch_size, seq_len) 或 (seq_len,)
            attention_mask: attention mask，shape 为 (batch_size, seq_len)，用于 mask padding tokens
        
        Returns:
            embeddings: embedding 向量，shape 为 (batch_size, seq_len, embedding_dim) 或 (seq_len, embedding_dim)
        """
        # 检查 input_ids 是否超出词汇表范围（不允许自动扩容，确保可复现性）
        max_token_id = input_ids.max().item()
        if max_token_id >= self.vocab_size:
            raise ValueError(
                f"Token id {max_token_id} out of range (vocab_size={self.vocab_size}). "
                "Did you change tokenizer or add tokens? "
                "Please resize embedding explicitly before training/inference."
            )
        
        x = self.embedding(input_ids)
        
        # 应用 attention_mask 避免 padding 污染
        if attention_mask is not None:
            # attention_mask: (batch_size, seq_len) -> (batch_size, seq_len, 1)
            x = x * attention_mask.unsqueeze(-1)
        
        return x
    
    def get_embedding_dim(self) -> int:
        """返回 embedding 维度"""
        return self.embedding_dim
    
    def get_vocab_size(self) -> int:
        """返回词汇表大小"""
        return self.vocab_size
    
    @classmethod
    def load_from_safetensors_path(
        cls,
        tokenizer: AutoTokenizer,
        safetensors_path: str,
        embedding_dim: Optional[int] = None,
        cache_dir: str = "./embeddings_cache",
    ) -> "TokenEmbedding":
        """
        从 safetensors 文件路径加载模型
        
        Args:
            tokenizer: tokenizer 对象
            safetensors_path: safetensors 文件路径
            embedding_dim: embedding 维度，如果为 None 则从文件元数据读取
            cache_dir: 缓存目录
        
        Returns:
            TokenEmbedding 实例
        """
        safetensors_path = Path(safetensors_path)
        if not safetensors_path.exists():
            raise FileNotFoundError(f"Model file not found: {safetensors_path}")
        
        # 读取元数据获取配置
        try:
            from safetensors import safe_open
            with safe_open(str(safetensors_path), framework="pt") as f:
                metadata = f.metadata()
                if metadata:
                    loaded_embedding_dim = int(metadata.get("embedding_dim", "768"))
                    if embedding_dim is None:
                        embedding_dim = loaded_embedding_dim
                    elif embedding_dim != loaded_embedding_dim:
                        print(f"Warning: Config embedding_dim ({embedding_dim}) != loaded dim ({loaded_embedding_dim})")
                        embedding_dim = loaded_embedding_dim
        except ImportError:
            # 如果无法读取元数据，使用默认值或配置值
            print("Warning: Cannot read metadata from safetensors file, using config or default embedding_dim")
            if embedding_dim is None:
                embedding_dim = 768
        
        # 创建模型实例
        instance = cls(
            tokenizer=tokenizer,
            embedding_dim=embedding_dim or 768,
            cache_dir=cache_dir,
            init_from_pretrained=False,  # 不从预训练加载，而是从文件加载
        )
        
        # 加载权重
        instance.load_from_safetensors(safetensors_path)
        
        return instance


def download_tokenizer_to_local(
    model_name: str,
    local_dir: str = "./tokenizers",
    trust_remote_code: bool = True
) -> Path:
    """
    下载 tokenizer 到本地项目目录
    
    Args:
        model_name: 模型名称，如 "Qwen/Qwen3-8B"
        local_dir: 本地保存目录
        trust_remote_code: 是否信任远程代码
    
    Returns:
        本地 tokenizer 目录路径
    """
    local_path = Path(local_dir) / model_name.replace("/", "_")
    
    # 如果已经存在，直接返回
    if local_path.exists() and (local_path / "tokenizer_config.json").exists():
        print(f"Tokenizer already exists at {local_path}")
        return local_path
    
    print(f"Downloading tokenizer {model_name} to {local_path}...")
    local_path.mkdir(parents=True, exist_ok=True)
    
    # 下载 tokenizer
    # 使用 use_fast=False 避免与某些 tokenizer 文件格式不兼容的问题
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        use_fast=False
    )
    
    # 保存到本地
    tokenizer.save_pretrained(str(local_path))
    print(f"Tokenizer saved to {local_path}")
    
    return local_path


def load_tokenizer_from_local(
    local_path: str,
    trust_remote_code: bool = True
) -> AutoTokenizer:
    """
    从本地目录加载 tokenizer
    
    Args:
        local_path: 本地 tokenizer 目录路径
        trust_remote_code: 是否信任远程代码
    
    Returns:
        tokenizer 对象
    """
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Tokenizer directory not found: {local_path}")
    
    print(f"Loading tokenizer from {local_path}")
    # 使用 use_fast=False 避免与某些 tokenizer 文件格式不兼容的问题
    tokenizer = AutoTokenizer.from_pretrained(
        str(local_path),
        trust_remote_code=trust_remote_code,
        use_fast=False
    )
    return tokenizer


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    return config


def load_tokenizer_from_config(config: Dict[str, Any]) -> AutoTokenizer:
    """
    根据配置加载 tokenizer
    
    Args:
        config: 配置字典
    
    Returns:
        tokenizer 对象
    """
    tokenizer_config = config.get("tokenizer", {})
    model_name = tokenizer_config.get("model_name", "Qwen/Qwen3-8B")
    tokenizer_dir = tokenizer_config.get("local_dir", "./tokenizers")
    trust_remote_code = tokenizer_config.get("trust_remote_code", True)
    
    # 下载 tokenizer 到本地（如果不存在）
    local_tokenizer_path = download_tokenizer_to_local(
        model_name=model_name,
        local_dir=tokenizer_dir,
        trust_remote_code=trust_remote_code
    )
    
    # 从本地加载 tokenizer
    tokenizer = load_tokenizer_from_local(
        local_path=local_tokenizer_path,
        trust_remote_code=trust_remote_code
    )
    
    return tokenizer


def create_model_from_config(config: Dict[str, Any], tokenizer: Optional[AutoTokenizer] = None) -> TokenEmbedding:
    """
    根据配置创建或加载模型
    
    Args:
        config: 配置字典
        tokenizer: tokenizer 对象，如果为 None 则从配置加载
    
    Returns:
        TokenEmbedding 实例
    """
    # 如果没有提供 tokenizer，从配置加载
    if tokenizer is None:
        tokenizer = load_tokenizer_from_config(config)
    
    # 统一从 config["model"]["embedding"] 读取配置
    embedding_config = config.get("model", {}).get("embedding", {})
    embedding_path = embedding_config.get("path", "").strip()
    
    # 其他 embedding 参数也从同一位置读取
    embedding_dim = embedding_config.get("embedding_dim", 768)
    vocab_size = embedding_config.get("vocab_size")
    padding_idx = embedding_config.get("padding_idx")
    init_from_pretrained = embedding_config.get("init_from_pretrained", False)
    cache_dir = embedding_config.get("cache_dir", "./embeddings_cache")
    
    if embedding_path:
        # 从指定路径加载模型
        print(f"Loading embedding model from: {embedding_path}")
        token_embedding = TokenEmbedding.load_from_safetensors_path(
            tokenizer=tokenizer,
            safetensors_path=embedding_path,
            embedding_dim=embedding_dim,
            cache_dir=cache_dir,
        )
    else:
        # 初始化新模型
        print("Initializing new embedding model...")
        token_embedding = TokenEmbedding(
            tokenizer=tokenizer,
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            padding_idx=padding_idx,
            init_from_pretrained=init_from_pretrained,
            cache_dir=cache_dir,
        )
    
    return token_embedding


class RouterModel(nn.Module):
    """
    Router 模型：使用两个 Mamba Encoder 进行工具路由
    
    架构：
    - TokenEmbedding: 将 token IDs 转换为 embedding
    - QueryEncoder (Mamba): 编码用户查询
    - ToolEncoder (Mamba): 编码工具描述（共享或独立）
    - Router: 计算相似度并选择工具（相似度匹配模式）
    """
    
    def __init__(
        self,
        token_embedding: TokenEmbedding,
        d_model: int = 768,
        n_layers: int = 4,
        d_state: int = 16,
        expand: int = 2,
        share_encoder: bool = False,
    ):
        """
        Args:
            token_embedding: TokenEmbedding 实例
            d_model: Mamba Encoder 的隐藏维度
            n_layers: Mamba Encoder 的层数
            d_state: Mamba 状态空间维度
            expand: Mamba 扩展因子
            share_encoder: 是否共享 Query 和 Tool Encoder
        """
        super().__init__()
        
        self.token_embedding = token_embedding
        self.d_model = d_model
        self.share_encoder = share_encoder
        
        # Query Encoder（用户输入）
        self.query_encoder = MambaEncoder(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            expand=expand,
        )
        
        # Tool Encoder（工具描述）
        if share_encoder:
            self.tool_encoder = self.query_encoder  # 共享权重
        else:
            self.tool_encoder = MambaEncoder(
                d_model=d_model,
                n_layers=n_layers,
                d_state=d_state,
                expand=expand,
            )
    
    def encode_query(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        编码用户查询
        
        Args:
            input_ids: [B, T] - token IDs
            attention_mask: [B, T] - attention mask
        
        Returns:
            h_query: [B, D] - 查询语义向量
        """
        # Token embedding
        x = self.token_embedding(input_ids, attention_mask=attention_mask)  # [B, T, D]
        
        # Mamba Encoder
        _, h_query = self.query_encoder(x, attention_mask=attention_mask, return_pooled=True)  # [B, D]
        
        return h_query
    
    def encode_tool(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        编码工具描述
        
        Args:
            input_ids: [B, T] - token IDs
            attention_mask: [B, T] - attention mask
        
        Returns:
            h_tool: [B, D] - 工具语义向量
        """
        # Token embedding
        x = self.token_embedding(input_ids, attention_mask=attention_mask)  # [B, T, D]
        
        # Mamba Encoder
        _, h_tool = self.tool_encoder(x, attention_mask=attention_mask, return_pooled=True)  # [B, D]
        
        return h_tool
    
    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: Optional[torch.Tensor] = None,
        tool_input_ids: Optional[torch.Tensor] = None,
        tool_attention_mask: Optional[torch.Tensor] = None,
        tool_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播（相似度匹配模式）
        
        Args:
            query_input_ids: [B, T] - 查询 token IDs
            query_attention_mask: [B, T] - 查询 attention mask
            tool_input_ids: [B, T_tool] - 工具描述 token IDs（可选）
            tool_attention_mask: [B, T_tool] - 工具 attention mask（可选）
            tool_embeddings: [N, D] - 预计算的工具向量（可选，N 为工具数量）
        
        Returns:
            scores: [B, N] 或 [B, 1] - 与工具的相似度分数
                - 如果提供 tool_embeddings: [B, N] - 与每个工具的相似度
                - 如果提供 tool_input_ids: [B, 1] - 与单个工具的相似度
        """
        # 编码查询
        h_query = self.encode_query(query_input_ids, query_attention_mask)  # [B, D]
        
        # L2 归一化查询向量（添加 epsilon 防止零向量）
        eps = 1e-8
        h_query_norm = h_query.norm(p=2, dim=-1, keepdim=True)
        h_query_norm = torch.clamp(h_query_norm, min=eps)
        h_query = h_query / h_query_norm
        
        # 相似度匹配模式
        if tool_embeddings is not None:
            # 使用预计算的工具向量
            # tool_embeddings: [N, D]
            # 确保工具向量也已归一化
            tool_norm = tool_embeddings.norm(p=2, dim=-1, keepdim=True)
            tool_norm = torch.clamp(tool_norm, min=eps)
            tool_embeddings = tool_embeddings / tool_norm
            scores = torch.matmul(h_query, tool_embeddings.t())  # [B, N]
            # 归一化后的点积已经在 [-1, 1] 范围内，不需要额外 clamp
        elif tool_input_ids is not None:
            # 动态编码工具描述
            h_tool = self.encode_tool(tool_input_ids, tool_attention_mask)  # [B, D]
            # L2 归一化工具向量
            h_tool_norm = h_tool.norm(p=2, dim=-1, keepdim=True)
            h_tool_norm = torch.clamp(h_tool_norm, min=eps)
            h_tool = h_tool / h_tool_norm
            # 计算点积相似度（假设 batch 中每个样本对应一个工具）
            scores = (h_query * h_tool).sum(dim=-1, keepdim=True)  # [B, 1]
            # 归一化后的点积已经在 [-1, 1] 范围内，不需要额外 clamp
        else:
            raise ValueError("Either tool_embeddings or tool_input_ids must be provided")
        
        return scores
    
    def get_top_k_tools(
        self,
        query_input_ids: torch.Tensor,
        tool_embeddings: torch.Tensor,
        query_attention_mask: Optional[torch.Tensor] = None,
        k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取 Top-k 工具
        
        Args:
            query_input_ids: [B, T] - 查询 token IDs
            query_attention_mask: [B, T] - 查询 attention mask
            tool_embeddings: [N, D] - 所有工具向量
            k: Top-k 数量
        
        Returns:
            top_k_indices: [B, k] - Top-k 工具索引
            top_k_scores: [B, k] - Top-k 相似度分数
        """
        # 编码查询
        h_query = self.encode_query(query_input_ids, query_attention_mask)  # [B, D]
        
        # L2 归一化（与训练时保持一致）
        eps = 1e-8
        h_query_norm = h_query.norm(p=2, dim=-1, keepdim=True)
        h_query = h_query / (h_query_norm + eps)
        
        # 确保工具向量也已归一化
        tool_norm = tool_embeddings.norm(p=2, dim=-1, keepdim=True)
        tool_norm = torch.clamp(tool_norm, min=eps)
        tool_embeddings = tool_embeddings / tool_norm
        
        # 计算相似度
        scores = torch.matmul(h_query, tool_embeddings.t())  # [B, N]
        # 归一化后的点积已经在 [-1, 1] 范围内，不需要额外 clamp
        
        # 获取 Top-k
        top_k_scores, top_k_indices = torch.topk(scores, k=k, dim=-1)  # [B, k]
        
        return top_k_indices, top_k_scores


class RouterDataset(Dataset):
    """
    Router 训练数据集（工具匹配模式）
    
    数据格式：{"query": str, "tool_description": str, "label": int}
    """
    
    def __init__(
        self,
        dataset,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        truncation: bool = True,
        padding: str = "max_length",
        precompute_tokens: bool = False,
    ):
        """
        Args:
            dataset: HuggingFace 数据集对象
            tokenizer: tokenizer
            max_length: 最大序列长度
            truncation: 是否截断
            padding: padding 策略
            precompute_tokens: 是否预计算 tokenization（加速数据加载）
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding
        self.precompute_tokens = precompute_tokens
        
        # 工具匹配：使用 "query" 和 "tool_description" 或 "tool_name" 字段
        self.text_field = "query"
        # 优先使用 tool_description，如果没有则使用 tool_name
        if "tool_description" in dataset.column_names:
            self.tool_field = "tool_description"
        elif "tool_name" in dataset.column_names:
            self.tool_field = "tool_name"
        else:
            self.tool_field = "tool_description"  # 默认字段名
        self.label_field = "label"
        
        # 预计算 tokenization（如果启用）
        self._cached_tokens = None
        if precompute_tokens:
            print("预计算 tokenization（加速数据加载）...")
            self._precompute_tokens()
    
    def _precompute_tokens(self):
        """预计算所有样本的 tokenization"""
        cached = []
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            query = item.get(self.text_field, "")
            tool_desc = item.get(self.tool_field, "")
            label = item.get(self.label_field, 0)
            
            # Tokenize 查询
            query_enc = self.tokenizer(
                query,
                max_length=self.max_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors="pt"
            )
            
            # Tokenize 工具描述
            tool_enc = self.tokenizer(
                tool_desc,
                max_length=self.max_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors="pt"
            )
            
            cached.append({
                "query_input_ids": query_enc["input_ids"].squeeze(0),
                "query_attention_mask": query_enc.get("attention_mask", None).squeeze(0) if query_enc.get("attention_mask") is not None else None,
                "tool_input_ids": tool_enc["input_ids"].squeeze(0),
                "tool_attention_mask": tool_enc.get("attention_mask", None).squeeze(0) if tool_enc.get("attention_mask") is not None else None,
                "label": torch.tensor(label, dtype=torch.long),
            })
        self._cached_tokens = cached
        print(f"✓ 预计算完成: {len(cached)} 个样本")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # 如果预计算了，直接返回缓存
        if self._cached_tokens is not None:
            return self._cached_tokens[idx]
        
        # 否则实时 tokenize
        item = self.dataset[idx]
        
        # 工具匹配模式：编码查询和工具描述
        query = item.get(self.text_field, "")
        tool_desc = item.get(self.tool_field, "")
        label = item.get(self.label_field, 0)
        
        # Tokenize 查询
        query_enc = self.tokenizer(
            query,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors="pt"
        )
        
        # Tokenize 工具描述
        tool_enc = self.tokenizer(
            tool_desc,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors="pt"
        )
        
        return {
            "query_input_ids": query_enc["input_ids"].squeeze(0),
            "query_attention_mask": query_enc.get("attention_mask", None).squeeze(0) if query_enc.get("attention_mask") is not None else None,
            "tool_input_ids": tool_enc["input_ids"].squeeze(0),
            "tool_attention_mask": tool_enc.get("attention_mask", None).squeeze(0) if tool_enc.get("attention_mask") is not None else None,
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_training_dataset(dataset_name: str = "snips_built_in_intents", train_size: Optional[int] = None):
    """
    加载训练数据集
    
    支持两种方式：
    1. 从本地 JSON 文件加载（如果 dataset_name 是文件路径）
    2. 从 HuggingFace 加载（如果 dataset_name 是数据集名称）
    
    Args:
        dataset_name: 数据集名称或本地 JSON 文件路径
        train_size: 训练样本数量，如果为 None 则使用全部数据
    
    Returns:
        数据集对象
    """
    dataset_path = Path(dataset_name)
    
    # 检查是否是本地 JSON 文件（通过后缀和路径判断）
    is_local_json = dataset_path.suffix == ".json" or dataset_name.endswith(".json")
    
    if is_local_json:
        # 尝试解析为绝对路径
        if not dataset_path.is_absolute():
            dataset_path = Path.cwd() / dataset_path
        
        # 检查文件是否存在
        if dataset_path.exists():
            print(f"Loading dataset from local file: {dataset_path}")
            from datasets import Dataset
            
            # 从 JSON 文件加载
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 确保是列表格式
            if not isinstance(data, list):
                raise ValueError(f"JSON file must contain a list of samples, got {type(data)}")
            
            # 清理数据：将复杂对象转换为字符串，避免 PyArrow 类型冲突
            cleaned_data = []
            for item in data:
                cleaned_item = {}
                for key, value in item.items():
                    # 保留基本字段（字符串、数字、布尔值）
                    if key in ["query", "tool_name", "tool_description", "label"]:
                        cleaned_item[key] = value
                    # 将复杂对象（tool_schema, metadata）转换为 JSON 字符串
                    elif key in ["tool_schema", "metadata"] and isinstance(value, (dict, list)):
                        cleaned_item[key] = json.dumps(value, ensure_ascii=False)
                    # 其他字段也转换为字符串（如果可能）
                    elif isinstance(value, (dict, list)):
                        cleaned_item[key] = json.dumps(value, ensure_ascii=False)
                    else:
                        cleaned_item[key] = value
                cleaned_data.append(cleaned_item)
            
            # 创建 Dataset
            dataset = Dataset.from_list(cleaned_data)
            
            if train_size is not None and train_size > 0:
                dataset = dataset.select(range(min(train_size, len(dataset))))
                print(f"Using {len(dataset)} samples (limited from {len(data)} total)")
            else:
                print(f"Using all {len(dataset)} training samples")
            
            return dataset
        else:
            # 文件不存在，给出清晰的错误信息
            raise FileNotFoundError(
                f"Dataset file not found: {dataset_path}\n"
                f"Please check the file path in config.json or run the conversion script:\n"
                f"  python convert_hermes_dataset.py --output {dataset_path.name}"
            )
    else:
        # 从 HuggingFace 加载
        print(f"Loading dataset from HuggingFace: {dataset_name}")
        dataset = load_dataset(dataset_name, cache_dir="./datasets")
        
        if "train" in dataset:
            train_data = dataset["train"]
            if train_size is not None and train_size > 0:
                # 限制训练样本数量
                train_data = train_data.select(range(min(train_size, len(train_data))))
                print(f"Using {len(train_data)} samples (limited from {len(dataset['train'])} total)")
            else:
                print(f"Using all {len(train_data)} training samples")
            
            return train_data
        else:
            # 如果没有 train 分割，使用第一个分割
            first_split = list(dataset.keys())[0]
            train_data = dataset[first_split]
            if train_size is not None and train_size > 0:
                train_data = train_data.select(range(min(train_size, len(train_data))))
            print(f"Using {len(train_data)} samples from '{first_split}' split")
            return train_data


def save_checkpoint(
    checkpoint_dir: Path,
    checkpoint_name: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    step: int,
    config: Dict[str, Any],
    best_metric: Optional[float] = None,
    scheduler: Optional[Any] = None,
):
    """
    保存 checkpoint
    
    Args:
        checkpoint_dir: checkpoint 目录
        checkpoint_name: checkpoint 名称
        model: 模型
        optimizer: 优化器
        epoch: 当前 epoch
        step: 当前 step
        config: 配置字典
        best_metric: 最佳指标值
        scheduler: 学习率调度器（可选）
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"{checkpoint_name}_epoch{epoch}_step{step}.pt"
    latest_path = checkpoint_dir / f"{checkpoint_name}_latest.pt"
    
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": config,
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    if best_metric is not None:
        checkpoint["best_metric"] = best_metric
    
    # 保存 checkpoint
    torch.save(checkpoint, checkpoint_path)
    # 同时保存为 latest
    torch.save(checkpoint, latest_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
    scheduler: Optional[Any] = None,
):
    """
    加载 checkpoint
    
    Args:
        checkpoint_path: checkpoint 路径
        model: 模型
        optimizer: 优化器（可选）
        device: 设备
        scheduler: 学习率调度器（可选）
    
    Returns:
        (epoch, step, config, best_metric)
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # 加载 scheduler 状态（如果存在）
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"  Scheduler state loaded")
        except Exception as e:
            print(f"  ⚠️  Failed to load scheduler state: {e}")
    
    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("step", 0)
    config = checkpoint.get("config", {})
    best_metric = checkpoint.get("best_metric", None)
    
    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"  Epoch: {epoch}, Step: {step}")
    if best_metric is not None:
        print(f"  Best metric: {best_metric}")
    
    return epoch, step, config, best_metric


def find_latest_checkpoint(checkpoint_dir: Path, checkpoint_name: str) -> Optional[Path]:
    """查找最新的 checkpoint"""
    latest_path = checkpoint_dir / f"{checkpoint_name}_latest.pt"
    if latest_path.exists():
        return latest_path
    
    # 查找所有 checkpoint 文件
    pattern = f"{checkpoint_name}_epoch*_step*.pt"
    checkpoints = list(checkpoint_dir.glob(pattern))
    if checkpoints:
        # 按修改时间排序，返回最新的
        return max(checkpoints, key=lambda p: p.stat().st_mtime)
    
    return None


class StepLRWrapper:
    """
    包装器：将 step-wise 学习率调度包装为可用的调度器
    """
    def __init__(self, optimizer, lr_lambda_fn, num_steps):
        self.optimizer = optimizer
        self.lr_lambda_fn = lr_lambda_fn
        self.num_steps = num_steps
        self.current_step = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        """更新学习率（每个 step 调用一次）"""
        self.current_step += 1
        lr_multiplier = self.lr_lambda_fn(self.current_step)
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_multiplier
    
    def state_dict(self):
        """保存状态"""
        return {
            'current_step': self.current_step,
            'num_steps': self.num_steps,
        }
    
    def load_state_dict(self, state_dict):
        """加载状态"""
        self.current_step = state_dict.get('current_step', 0)
        self.num_steps = state_dict.get('num_steps', self.num_steps)
        # 恢复学习率
        lr_multiplier = self.lr_lambda_fn(self.current_step)
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_multiplier


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    num_training_steps: int,
) -> Optional[Any]:
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        config: 配置字典（包含 lr_scheduler 配置）
        num_training_steps: 总训练步数
    
    Returns:
        学习率调度器（如果启用）或 None
    """
    lr_config = config.get("training", {}).get("lr_scheduler", {})
    
    if not lr_config.get("enabled", False):
        return None
    
    scheduler_type = lr_config.get("type", "cosine")
    base_lr = optimizer.param_groups[0]["lr"]
    min_lr = lr_config.get("min_lr", 1e-6)
    
    # 计算 warmup steps
    warmup_steps = lr_config.get("warmup_steps", 0)
    if warmup_steps == 0:
        warmup_ratio = lr_config.get("warmup_ratio", 0.1)
        warmup_steps = int(num_training_steps * warmup_ratio)
    
    # 实际训练步数（减去 warmup）
    training_steps = num_training_steps - warmup_steps
    
    if scheduler_type == "cosine":
        # 余弦退火调度器（带 warmup）
        def lr_lambda(step):
            if step < warmup_steps:
                # Warmup 阶段：线性增长
                return step / warmup_steps if warmup_steps > 0 else 1.0
            else:
                # Cosine annealing 阶段
                progress = (step - warmup_steps) / training_steps if training_steps > 0 else 0.0
                progress = min(1.0, max(0.0, progress))  # 限制在 [0, 1]
                cosine_factor = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359)))
                return min_lr / base_lr + (1.0 - min_lr / base_lr) * cosine_factor.item()
        
        scheduler = StepLRWrapper(optimizer, lr_lambda, num_training_steps)
    elif scheduler_type == "linear":
        # 线性衰减调度器（带 warmup）
        def lr_lambda(step):
            if step < warmup_steps:
                # Warmup 阶段：线性增长
                return step / warmup_steps if warmup_steps > 0 else 1.0
            else:
                # Linear decay 阶段
                progress = (step - warmup_steps) / training_steps if training_steps > 0 else 0.0
                progress = min(1.0, max(0.0, progress))  # 限制在 [0, 1]
                return max(min_lr / base_lr, 1.0 - progress)
        
        scheduler = StepLRWrapper(optimizer, lr_lambda, num_training_steps)
    elif scheduler_type == "constant":
        # 固定学习率（只在 warmup 阶段变化）
        if warmup_steps > 0:
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    return 1.0
            scheduler = StepLRWrapper(optimizer, lr_lambda, num_training_steps)
        else:
            return None
    else:
        print(f"⚠️  未知的调度器类型: {scheduler_type}，使用固定学习率")
        return None
    
    print(f"\n学习率调度器配置:")
    print(f"  类型: {scheduler_type}")
    print(f"  初始学习率: {base_lr:.2e}")
    print(f"  最小学习率: {min_lr:.2e}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  总训练步数: {num_training_steps}")
    
    return scheduler


def train_one_epoch(
    model: RouterModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    precision: str = "fp32",
    epoch: int = 0,
    scheduler: Optional[Any] = None,
    gradient_accumulation_steps: int = 1,
    gradient_clip_norm: Optional[float] = None,
    temperature: float = 1.0,
) -> float:
    """
    训练一个 epoch（相似度匹配模式）
    
    Args:
        model: RouterModel
        dataloader: 数据加载器
        optimizer: 优化器
        device: 设备
        precision: 精度（fp32, amp-fp16, amp-bf16）
        epoch: 当前 epoch
        scheduler: 学习率调度器
        gradient_accumulation_steps: 梯度累积步数
    
    Returns:
        平均损失
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # 设置混合精度
    # 注意：ROCm 对混合精度的支持可能不稳定，如果遇到内存错误，建议使用 fp32
    is_rocm = hasattr(torch.version, 'hip') if hasattr(torch, 'version') else False
    
    if precision in ["amp-fp16", "amp-bf16"] and device.type == 'cuda':
        try:
            # 尝试初始化混合精度（ROCm 和 CUDA 都尝试）
            if is_rocm:
                # ROCm 环境：使用旧 API
                scaler = torch.cuda.amp.GradScaler()
            else:
                # CUDA 环境：优先使用新 API
                try:
                    scaler = torch.amp.GradScaler('cuda')
                except (AttributeError, TypeError):
                    # 回退到旧 API
                    scaler = torch.cuda.amp.GradScaler()
            autocast_dtype = torch.float16 if precision == "amp-fp16" else torch.bfloat16 if precision == "amp-bf16" else None
        except Exception as e:
            print(f"⚠️  混合精度初始化失败: {e}")
            print("   回退到 fp32 精度")
            scaler = None
            autocast_dtype = None
    else:
        scaler = None
        autocast_dtype = None
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    # 梯度累积相关
    accumulated_steps = 0
    
    # ROCm + fp16 错误计数器（如果连续失败，回退到 fp32）
    rocm_fp16_failures = 0
    max_rocm_fp16_failures = 3  # 最多允许 3 次失败
    
    for batch_idx, batch in enumerate(pbar):
        try:
            # 移动到设备（使用 non_blocking=False 确保稳定性，特别是 ROCm 环境）
            query_input_ids = batch["query_input_ids"].to(device, non_blocking=False)
            query_attention_mask = batch.get("query_attention_mask")
            if query_attention_mask is not None:
                query_attention_mask = query_attention_mask.to(device, non_blocking=False)
            
            labels = batch["label"].to(device, non_blocking=False)
            
            # 相似度匹配模式：需要工具描述
            if "tool_input_ids" not in batch:
                raise ValueError("工具匹配模式需要 tool_input_ids，请确保数据集包含工具描述")
            
            tool_input_ids = batch["tool_input_ids"].to(device, non_blocking=False)
            tool_attention_mask = batch.get("tool_attention_mask")
            if tool_attention_mask is not None:
                tool_attention_mask = tool_attention_mask.to(device, non_blocking=False)
            
            # 前向传播
            if autocast_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    # 编码查询和工具
                    h_query = model.encode_query(query_input_ids, query_attention_mask)  # [B, D]
                    h_tool = model.encode_tool(tool_input_ids, tool_attention_mask)  # [B, D]
                    
                    # L2 归一化（稳定训练，将点积限制在 [-1, 1] 范围）
                    # 添加 epsilon 防止零向量归一化问题
                    eps = 1e-8
                    h_query_norm = h_query.norm(p=2, dim=-1, keepdim=True)
                    h_tool_norm = h_tool.norm(p=2, dim=-1, keepdim=True)
                    
                    # 检查是否有 NaN 或 Inf
                    if torch.isnan(h_query).any() or torch.isnan(h_tool).any():
                        print(f"\n⚠️  检测到 NaN 在 embeddings (batch {batch_idx})")
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        continue
                    
                    # 防止零向量（如果范数太小，使用单位向量）
                    h_query_norm = torch.clamp(h_query_norm, min=eps)
                    h_tool_norm = torch.clamp(h_tool_norm, min=eps)
                    
                    h_query = h_query / h_query_norm
                    h_tool = h_tool / h_tool_norm
                    
                    # 计算相似度（归一化后的点积，范围 [-1, 1]）
                    # 注意：归一化后的点积已经在 [-1, 1] 范围内
                    scores = (h_query * h_tool).sum(dim=-1)  # [B]
                    
                    # 【关键修复】将相似度分数放大为 logits
                    # BCE with logits 期望 logits 范围较大（通常 [-10, 10]），
                    # 但归一化后的点积范围只有 [-1, 1]，导致梯度信号弱，训练困难
                    # 解决方案：将相似度分数放大（乘以 10-20），使其成为有效的 logits
                    logit_scale = 10.0  # 将 [-1, 1] 映射到 [-10, 10]
                    scores = scores * logit_scale
                    
                    # 应用温度缩放（可选，用于调整相似度差异）
                    # 如果 temperature=1.0，相当于不缩放
                    # 如果 temperature<1.0，会放大相似度差异
                    # 如果 temperature>1.0，会缩小相似度差异
                    if temperature != 1.0:
                        safe_temperature = max(temperature, 1e-6)
                        scores = scores / safe_temperature
                    
                    # 数值稳定性：限制 scores 范围，防止 fp16 溢出
                    # fp16 的范围大约是 [-65504, 65504]，但为了安全，限制在 [-50, 50]
                    # 注意：只有在极端情况下才会触发（如 temperature 非常小）
                    max_score = 50.0 if autocast_dtype == torch.float16 else 100.0
                    scores = torch.clamp(scores, min=-max_score, max=max_score)
                    
                    # 检查 NaN 和 Inf
                    if torch.isnan(scores).any() or torch.isinf(scores).any():
                        print(f"\n⚠️  检测到 NaN/Inf 在 scores (batch {batch_idx})")
                        print(f"   scores 范围: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
                        print(f"   temperature: {temperature}")
                        print(f"   h_query norm: {h_query_norm.min().item():.4f} - {h_query_norm.max().item():.4f}")
                        print(f"   h_tool norm: {h_tool_norm.min().item():.4f} - {h_tool_norm.max().item():.4f}")
                        # 跳过这个 batch
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        continue
                    
                    # 使用二元交叉熵（正样本=1，负样本=0）
                    # 梯度累积时，loss 需要除以累积步数
                    loss = F.binary_cross_entropy_with_logits(scores, labels.float()) / gradient_accumulation_steps
                    
                    # 检查 loss 是否为 NaN
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"\n⚠️  检测到 NaN/Inf loss (batch {batch_idx})")
                        print(f"   scores 范围: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
                        print(f"   labels: {labels.tolist()}")
                        print(f"   temperature: {temperature}")
                        # 跳过这个 batch
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        continue
            else:
                # 编码查询和工具
                h_query = model.encode_query(query_input_ids, query_attention_mask)
                h_tool = model.encode_tool(tool_input_ids, tool_attention_mask)
                
                # L2 归一化（稳定训练，将点积限制在 [-1, 1] 范围）
                # 添加 epsilon 防止零向量归一化问题
                eps = 1e-8
                h_query_norm = h_query.norm(p=2, dim=-1, keepdim=True)
                h_tool_norm = h_tool.norm(p=2, dim=-1, keepdim=True)
                
                # 检查是否有 NaN 或 Inf
                if torch.isnan(h_query).any() or torch.isnan(h_tool).any():
                    print(f"\n⚠️  检测到 NaN 在 embeddings (batch {batch_idx})")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                
                # 防止零向量（如果范数太小，使用单位向量）
                h_query_norm = torch.clamp(h_query_norm, min=eps)
                h_tool_norm = torch.clamp(h_tool_norm, min=eps)
                
                h_query = h_query / h_query_norm
                h_tool = h_tool / h_tool_norm
                
                # 计算相似度（归一化后的点积，范围 [-1, 1]）
                # 注意：归一化后的点积已经在 [-1, 1] 范围内
                scores = (h_query * h_tool).sum(dim=-1)
                
                # 【关键修复】将相似度分数放大为 logits（与训练时保持一致）
                logit_scale = 10.0  # 将 [-1, 1] 映射到 [-10, 10]
                scores = scores * logit_scale
                
                # 应用温度缩放（可选，用于调整相似度差异）
                if temperature != 1.0:
                    safe_temperature = max(temperature, 1e-6)
                    scores = scores / safe_temperature
                
                # 数值稳定性：限制 scores 范围，防止数值溢出
                # 注意：只有在极端情况下才会触发
                max_score = 100.0
                scores = torch.clamp(scores, min=-max_score, max=max_score)
                
                # 检查 NaN 和 Inf
                if torch.isnan(scores).any() or torch.isinf(scores).any():
                    print(f"\n⚠️  检测到 NaN/Inf 在 scores (batch {batch_idx})")
                    print(f"   scores 范围: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
                    print(f"   temperature: {temperature}")
                    # 跳过这个 batch
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                
                # 梯度累积时，loss 需要除以累积步数
                loss = F.binary_cross_entropy_with_logits(scores, labels.float()) / gradient_accumulation_steps
                
                # 检查 loss 是否为 NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n⚠️  检测到 NaN/Inf loss (batch {batch_idx})")
                    print(f"   scores 范围: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
                    print(f"   labels: {labels.tolist()}")
                    print(f"   temperature: {temperature}")
                    # 跳过这个 batch
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
            
            # 反向传播（累积梯度）
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accumulated_steps += 1
            total_loss += loss.item() * gradient_accumulation_steps  # 恢复原始 loss 用于显示
            
            # 达到累积步数时，更新参数
            if accumulated_steps >= gradient_accumulation_steps:
                # 梯度裁剪（防止梯度爆炸）
                if gradient_clip_norm is not None:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    # 计算梯度范数（用于诊断）
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    
                    # 每 100 个批次打印一次梯度信息（用于诊断损失上升问题）
                    if batch_idx % 100 == 0:
                        print(f"\n📊 梯度诊断 (batch {batch_idx}):")
                        print(f"   梯度范数: {grad_norm.item():.4f}")
                        print(f"   当前损失: {display_loss:.4f}")
                        print(f"   学习率: {current_lr:.2e}")
                        if grad_norm.item() > 10.0:
                            print(f"   ⚠️  梯度范数较大，可能需要降低学习率")
                else:
                    # 即使不裁剪，也计算梯度范数用于诊断
                    if batch_idx % 100 == 0:
                        total_norm = 0.0
                        for p in model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** (1. / 2)
                        print(f"\n📊 梯度诊断 (batch {batch_idx}):")
                        print(f"   梯度范数: {total_norm:.4f}")
                        print(f"   当前损失: {display_loss:.4f}")
                        print(f"   学习率: {current_lr:.2e}")
                
                # 更新参数
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                # 清零梯度
                optimizer.zero_grad()
                
                # 更新学习率（每个有效 step 更新一次）
                if scheduler is not None:
                    scheduler.step()
                
                accumulated_steps = 0
            
            num_batches += 1
            
            # 显示当前学习率和损失
            current_lr = optimizer.param_groups[0]['lr']
            display_loss = total_loss / num_batches if num_batches > 0 else 0.0
            
            # 计算移动平均损失（用于判断趋势）
            if not hasattr(train_one_epoch, '_loss_history'):
                train_one_epoch._loss_history = []
            train_one_epoch._loss_history.append(display_loss)
            if len(train_one_epoch._loss_history) > 20:
                train_one_epoch._loss_history.pop(0)
            
            # 计算最近 20 个批次的平均损失（用于判断趋势）
            recent_avg = sum(train_one_epoch._loss_history) / len(train_one_epoch._loss_history) if train_one_epoch._loss_history else display_loss
            
            pbar.set_postfix({
                "loss": f"{display_loss:.4f}", 
                "avg20": f"{recent_avg:.4f}",
                "lr": f"{current_lr:.2e}",
                "acc": f"{accumulated_steps}/{gradient_accumulation_steps}"
            })
            
            # 每 50 个批次检查损失趋势
            if batch_idx > 0 and batch_idx % 50 == 0 and len(train_one_epoch._loss_history) >= 10:
                early_avg = sum(train_one_epoch._loss_history[:10]) / 10
                late_avg = sum(train_one_epoch._loss_history[-10:]) / 10
                if late_avg > early_avg * 1.1:  # 损失上升超过 10%
                    print(f"\n⚠️  警告: 损失上升趋势 (batch {batch_idx})")
                    print(f"   早期平均: {early_avg:.4f}")
                    print(f"   近期平均: {late_avg:.4f}")
                    print(f"   上升幅度: {(late_avg/early_avg - 1)*100:.1f}%")
                    print(f"   建议: 检查学习率、梯度范数或数据质量")
            
            # 清理GPU内存（进一步减少频率以提高速度）
            # 只在必要时清理（每 50 个 batch 或每 100 个 batch）
            if device.type == 'cuda':
                if is_rocm:
                    # ROCm 环境下每 50 个 batch 清理一次（进一步减少频率）
                    if batch_idx % 50 == 0:
                        torch.cuda.empty_cache()
                else:
                    # CUDA 环境下每 100 个 batch 清理一次
                    if batch_idx % 100 == 0:
                        torch.cuda.empty_cache()
        
        except RuntimeError as e:
            # 处理 GPU 内存错误（包括 OutOfMemoryError，它是 RuntimeError 的子类）
            error_msg = str(e).lower()
            # 检查是否是 OutOfMemoryError
            is_oom = "out of memory" in error_msg or isinstance(e, torch.cuda.OutOfMemoryError) if hasattr(torch.cuda, 'OutOfMemoryError') else "out of memory" in error_msg
            is_memory_fault = "page not present" in error_msg or "memory access fault" in error_msg
            is_memory_error = is_oom or is_memory_fault or "memory" in error_msg
            
            if is_memory_error:
                # 检查是否是 ROCm + fp16 的问题
                if is_rocm and precision in ["amp-fp16", "amp-bf16"]:
                    rocm_fp16_failures += 1
                    print("\n" + "🔴" * 40)
                    
                    # 区分 OOM 和 Memory Access Fault
                    if is_oom:
                        error_type = "Out of Memory (显存不足)"
                        error_detail = "GPU 显存不足，无法分配所需内存"
                    elif is_memory_fault:
                        error_type = "Memory Access Fault (内存访问错误)"
                        error_detail = "ROCm + fp16 导致的内存访问错误"
                    else:
                        error_type = "Memory Error (内存错误)"
                        error_detail = "GPU 内存相关错误"
                    
                    print(f"🔴 ROCm + fp16 内存错误 (第 {rocm_fp16_failures}/{max_rocm_fp16_failures} 次)")
                    print("🔴" * 40)
                    print(f"\n❌ 错误发生在 batch {batch_idx}")
                    print(f"❌ 错误类型: {error_type}")
                    print(f"❌ 错误详情: {error_detail}")
                    print(f"❌ 当前精度: {precision}")
                    print(f"❌ 当前 epoch: {epoch + 1}")
                    
                    if rocm_fp16_failures >= max_rocm_fp16_failures:
                        print("\n" + "🔴" * 40)
                        print("🔴 训练失败：ROCm + fp16 连续失败")
                        print("🔴" * 40)
                        print(f"\n❌ ROCm + fp16 已连续失败 {max_rocm_fp16_failures} 次")
                        print("\n📋 错误详情：")
                        if is_oom:
                            print("   - GPU 显存不足（可能是 fp16 导致的内存碎片）")
                            print("   - ROCm + fp16 可能导致内存管理问题")
                        else:
                            print("   - ROCm 对混合精度的支持不稳定")
                            print("   - 导致 GPU 内存访问错误")
                        print("   - 无法继续训练")
                        print("\n✅ 解决方案：")
                        print("   方案 1（推荐）：切换到 fp32")
                        print("     1. 修改 config.json")
                        print("     2. 将 \"precision\": \"amp-fp16\" 改为 \"precision\": \"fp32\"")
                        print("     3. 重新运行训练")
                        if is_oom:
                            print("\n   方案 2：减少显存占用（如果必须使用 fp16）")
                            print("     1. 减少 batch_size: 16 → 8 或 4")
                            print("     2. 减少 max_length: 256 → 192 或 128")
                            print("     3. 使用梯度累积: gradient_accumulation_steps = 2 或 4")
                        print("\n💡 说明：")
                        print("   - fp32 虽然速度慢，但可以稳定完成训练")
                        print("   - 这是 ROCm 平台的已知限制")
                        print("   - 等待 ROCm 对混合精度的支持改进")
                        print("\n" + "🔴" * 40)
                        raise RuntimeError(
                            f"\n{'='*60}\n"
                            f"ROCm + fp16 训练失败\n"
                            f"{'='*60}\n"
                            f"已连续失败 {max_rocm_fp16_failures} 次\n"
                            f"错误类型: {error_type}\n"
                            f"错误详情: {error_detail}\n"
                            f"当前 batch: {batch_idx}\n"
                            f"当前 epoch: {epoch + 1}\n"
                            f"\n解决方案:\n"
                            f"请修改 config.json 中 \"precision\": \"fp32\" 后重试\n"
                            f"{'='*60}\n"
                        )
                    else:
                        print(f"\n⚠️  清理内存并跳过这个 batch...")
                        print(f"⚠️  剩余尝试次数: {max_rocm_fp16_failures - rocm_fp16_failures}")
                        print(f"⚠️  如果继续失败，将自动停止训练")
                        if is_oom:
                            print(f"⚠️  提示: 如果频繁出现 OOM，建议切换到 fp32 或减少 batch_size")
                        print("🔴" * 40 + "\n")
                else:
                    # 非 ROCm + fp16 的内存错误
                    if is_oom:
                        print(f"\n⚠️  GPU 显存不足在 batch {batch_idx}")
                        print("   建议: 减少 batch_size 或 max_length")
                    else:
                        print(f"\n⚠️  GPU 内存错误在 batch {batch_idx}，清理内存并跳过...")
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                # 跳过这个 batch
                continue
            else:
                # 其他运行时错误，重新抛出
                raise
        except Exception as e:
            # 捕获所有其他异常（包括 SIGABRT 等）
            error_msg = str(e).lower()
            if "memory" in error_msg or "page not present" in error_msg or "abort" in error_msg:
                if is_rocm and precision in ["amp-fp16", "amp-bf16"]:
                    rocm_fp16_failures += 1
                    print("\n" + "🔴" * 40)
                    print(f"🔴 ROCm + fp16 严重错误 (第 {rocm_fp16_failures}/{max_rocm_fp16_failures} 次)")
                    print("🔴" * 40)
                    print(f"\n❌ 错误类型: {type(e).__name__}")
                    print(f"❌ 错误信息: {str(e)[:200]}")
                    print(f"❌ 错误发生在 batch {batch_idx}")
                    print(f"❌ 当前精度: {precision}")
                    print(f"❌ 当前 epoch: {epoch + 1}")
                    
                    if rocm_fp16_failures >= max_rocm_fp16_failures:
                        print("\n" + "🔴" * 40)
                        print("🔴 训练失败：ROCm + fp16 连续失败")
                        print("🔴" * 40)
                        print(f"\n❌ ROCm + fp16 已连续失败 {max_rocm_fp16_failures} 次")
                        print("\n📋 错误详情：")
                        print(f"   - 错误类型: {type(e).__name__}")
                        print("   - ROCm 对混合精度的支持不稳定")
                        print("   - 导致 GPU 内存访问错误或程序崩溃")
                        print("   - 无法继续训练")
                        print("\n✅ 解决方案：")
                        print("   1. 修改 config.json")
                        print("   2. 将 \"precision\": \"amp-fp16\" 改为 \"precision\": \"fp32\"")
                        print("   3. 重新运行训练")
                        print("\n💡 说明：")
                        print("   - fp32 虽然速度慢，但可以稳定完成训练")
                        print("   - 这是 ROCm 平台的已知限制")
                        print("   - 等待 ROCm 对混合精度的支持改进")
                        print("\n" + "🔴" * 40)
                        raise RuntimeError(
                            f"\n{'='*60}\n"
                            f"ROCm + fp16 训练失败\n"
                            f"{'='*60}\n"
                            f"已连续失败 {max_rocm_fp16_failures} 次\n"
                            f"错误类型: {type(e).__name__}\n"
                            f"错误信息: {str(e)[:200]}\n"
                            f"当前 batch: {batch_idx}\n"
                            f"当前 epoch: {epoch + 1}\n"
                            f"\n解决方案:\n"
                            f"请修改 config.json 中 \"precision\": \"fp32\" 后重试\n"
                            f"{'='*60}\n"
                        )
                    else:
                        print(f"\n⚠️  尝试继续...")
                        print(f"⚠️  剩余尝试次数: {max_rocm_fp16_failures - rocm_fp16_failures}")
                        print(f"⚠️  如果继续失败，将自动停止训练")
                        print("🔴" * 40 + "\n")
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        continue
                else:
                    raise
            else:
                raise
    
    # 处理剩余的累积梯度（如果最后一个 batch 没有达到累积步数）
    if accumulated_steps > 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(
    model: RouterModel,
    dataloader: DataLoader,
    device: torch.device,
    precision: str = "fp32",
    temperature: float = 1.0,
) -> Dict[str, float]:
    """
    评估模型（相似度匹配模式）
    
    Args:
        model: RouterModel
        dataloader: 数据加载器
        device: 设备
        precision: 精度
    
    Returns:
        评估指标字典
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    autocast_dtype = torch.float16 if precision == "amp-fp16" else torch.bfloat16 if precision == "amp-bf16" else None
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            query_input_ids = batch["query_input_ids"].to(device)
            query_attention_mask = batch.get("query_attention_mask")
            if query_attention_mask is not None:
                query_attention_mask = query_attention_mask.to(device)
            
            labels = batch["label"].to(device)
            
            # 相似度匹配模式：需要工具描述
            if "tool_input_ids" not in batch:
                raise ValueError("工具匹配模式需要 tool_input_ids，请确保数据集包含工具描述")
            
            tool_input_ids = batch["tool_input_ids"].to(device)
            tool_attention_mask = batch.get("tool_attention_mask")
            if tool_attention_mask is not None:
                tool_attention_mask = tool_attention_mask.to(device)
            
            # 前向传播
            if autocast_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    h_query = model.encode_query(query_input_ids, query_attention_mask)
                    h_tool = model.encode_tool(tool_input_ids, tool_attention_mask)
                    
                    # L2 归一化（与训练时保持一致）
                    eps = 1e-8
                    h_query_norm = h_query.norm(p=2, dim=-1, keepdim=True)
                    h_tool_norm = h_tool.norm(p=2, dim=-1, keepdim=True)
                    h_query = h_query / (h_query_norm + eps)
                    h_tool = h_tool / (h_tool_norm + eps)
                    
                    scores = (h_query * h_tool).sum(dim=-1)
                    # 应用温度缩放（与训练时保持一致）
                    if temperature != 1.0:
                        scores = scores / temperature
                    # 数值稳定性：限制 scores 范围（防止溢出）
                    scores = torch.clamp(scores, min=-100.0, max=100.0)
                    loss = F.binary_cross_entropy_with_logits(scores, labels.float())
                    predictions = (scores > 0).long()
            else:
                h_query = model.encode_query(query_input_ids, query_attention_mask)
                h_tool = model.encode_tool(tool_input_ids, tool_attention_mask)
                
                # L2 归一化（与训练时保持一致）
                eps = 1e-8
                h_query_norm = h_query.norm(p=2, dim=-1, keepdim=True)
                h_tool_norm = h_tool.norm(p=2, dim=-1, keepdim=True)
                h_query = h_query / (h_query_norm + eps)
                h_tool = h_tool / (h_tool_norm + eps)
                
                scores = (h_query * h_tool).sum(dim=-1)
                # 应用温度缩放（与训练时保持一致）
                if temperature != 1.0:
                    scores = scores / temperature
                # 数值稳定性：限制 scores 范围（防止溢出）
                scores = torch.clamp(scores, min=-100.0, max=100.0)
                loss = F.binary_cross_entropy_with_logits(scores, labels.float())
                predictions = (scores > 0).long()
            
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            total_loss += loss.item()
            num_batches += 1
    
    # 合并所有预测和标签
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 计算准确率
    accuracy = (all_predictions == all_labels).float().mean().item()
    
    metrics = {
        "loss": total_loss / num_batches if num_batches > 0 else 0.0,
        "accuracy": accuracy,
    }
    
    return metrics


def simple_inference(model: RouterModel, tokenizer: AutoTokenizer, text: str, config: Dict[str, Any]):
    """
    简单推理：将文本转换为 embedding
    
    Args:
        model: RouterModel
        tokenizer: tokenizer
        text: 输入文本
        config: 配置字典
    
    Returns:
        embedding 向量
    """
    training_config = config.get("training", {})
    
    # Tokenize
    enc = tokenizer(
        text,
        max_length=training_config.get("max_length", 2048),
        truncation=training_config.get("truncation", True),
        padding=training_config.get("padding", "max_length"),
        return_tensors="pt"
    )
    
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")
    
    # 前向传播
    with torch.no_grad():
        h_query = model.encode_query(input_ids, attention_mask=attention_mask)
    
    return h_query


def main():
    """主函数：训练和评估 Router 模型"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Router 模型训练和评估")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    parser.add_argument("--train-size", type=int, default=None, help="训练样本数量（None=使用配置文件中的值）")
    parser.add_argument("--dataset", type=str, default=None, help="数据集名称（None=使用配置文件中的值）")
    parser.add_argument("--test-text", type=str, default=None, help="测试文本（None=使用配置文件中的值或数据集样本）")
    parser.add_argument("--learning-rate", type=float, default=None, help="学习率（None=使用配置文件中的值）")
    parser.add_argument("--precision", type=str, default=None, choices=["fp32", "amp-fp16", "amp-bf16"], help="精度（None=使用配置文件中的值）")
    parser.add_argument("--resume", action="store_true", help="从checkpoint恢复训练（覆盖配置文件）")
    parser.add_argument("--no-resume", action="store_true", help="不从checkpoint恢复，从头开始训练（覆盖配置文件）")
    parser.add_argument("--mode", type=str, default=None, choices=["train", "eval", "inference"], help="运行模式：train=训练, eval=评估, inference=推理")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "rocm", "cpu"], help="设备选择：auto=自动检测, cuda=强制使用GPU(CUDA), rocm=强制使用GPU(ROCm), cpu=强制使用CPU")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 从配置文件读取默认值，命令行参数优先
    dataset_name = args.dataset if args.dataset is not None else config.get("dataset", {}).get("name", "snips_built_in_intents")
    train_size = args.train_size if args.train_size is not None else config.get("training", {}).get("train_size")
    test_text = args.test_text if args.test_text is not None else config.get("inference", {}).get("test_text")
    learning_rate = args.learning_rate if args.learning_rate is not None else config.get("training", {}).get("learning_rate", 1e-4)
    precision = args.precision if args.precision is not None else config.get("training", {}).get("precision", "fp32")
    
    # ROCm 环境检测和混合精度警告（不强制回退）
    is_rocm = hasattr(torch.version, 'hip') if hasattr(torch, 'version') else False
    if is_rocm and precision in ["amp-fp16", "amp-bf16"]:
        print("\n" + "🔴" * 30)
        print("🔴" * 30)
        print("⚠️  ⚠️  ⚠️  严重警告：ROCm + 混合精度 ⚠️  ⚠️  ⚠️")
        print("🔴" * 30)
        print("🔴" * 30)
        print(f"\n❌ 检测到 ROCm 环境且配置使用混合精度 ({precision})")
        print("\n⚠️  已知问题：")
        print("   - ROCm 对 fp16/bf16 混合精度的支持非常不稳定")
        print("   - 可能导致 'Memory access fault' 错误")
        print("   - 可能导致训练过程中断或崩溃")
        print("   - 可能导致 GPU core dump")
        print("\n💡 强烈建议：")
        print("   修改 config.json 中 \"precision\": \"fp32\"")
        print("   fp32 虽然慢，但可以稳定完成训练")
        print("\n⚠️  如果继续使用 fp16，可能会遇到：")
        print("   - 训练过程中频繁崩溃")
        print("   - 内存访问错误")
        print("   - 数据丢失（未保存的 checkpoint）")
        print("\n" + "🔴" * 30)
        print("🔴" * 30 + "\n")
        
        # 给用户 3 秒时间考虑是否继续
        import time
        import sys
        print("将在 5 秒后继续（按 Ctrl+C 取消）...")
        try:
            for i in range(5, 0, -1):
                print(f"  {i}...", end='\r', flush=True)
                time.sleep(1)
            print("  继续训练...\n")
        except KeyboardInterrupt:
            print("\n\n❌ 用户取消，退出程序")
            print("请修改 config.json 中 \"precision\": \"fp32\" 后重试")
            sys.exit(1)
    num_epochs = config.get("training", {}).get("num_epochs", 10)
    batch_size = config.get("training", {}).get("batch_size", 32)
    gradient_accumulation_steps = config.get("training", {}).get("gradient_accumulation_steps", 1)
    gradient_clip_norm = config.get("training", {}).get("gradient_clip_norm", None)
    temperature = config.get("training", {}).get("temperature", 1.0)
    eval_interval = config.get("training", {}).get("eval_interval", 1)
    top_k = config.get("evaluation", {}).get("top_k", 5)
    # 工具匹配模式（固定）
    
    # 处理 resume 参数：命令行优先
    if args.no_resume:
        resume = False
    elif args.resume:
        resume = True
    else:
        resume = config.get("training", {}).get("resume", True)
    
    # 运行模式
    run_mode = args.mode if args.mode is not None else "train"
    
    # Checkpoint 配置
    checkpoint_dir = Path(config.get("training", {}).get("checkpoint_dir", "./checkpoints"))
    checkpoint_name = config.get("training", {}).get("checkpoint_name", "router_model")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载 tokenizer
    tokenizer = load_tokenizer_from_config(config)
    
    # 创建 TokenEmbedding
    token_embedding = create_model_from_config(config, tokenizer=tokenizer)
    
    # 获取 Mamba 和 Router 配置
    mamba_config = config.get("model", {}).get("mamba", {})
    d_model = mamba_config.get("d_model", 768)
    n_layers = mamba_config.get("n_layers", 4)
    d_state = mamba_config.get("d_state", 16)
    expand = mamba_config.get("expand", 2)
    share_encoder = mamba_config.get("share_encoder", False)
    
    # 创建 RouterModel（相似度匹配模式）
    router_model = RouterModel(
        token_embedding=token_embedding,
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        expand=expand,
        share_encoder=share_encoder,
    )
    
    # 设备配置
    device_config = args.device if args.device is not None else config.get("training", {}).get("device", "auto")
    
    # 检查 ROCm 支持
    rocm_available = False
    try:
        import subprocess
        result = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=2)
        rocm_available = result.returncode == 0
    except:
        pass
    
    if device_config == "cuda" or device_config == "rocm":
        if not torch.cuda.is_available():
            device_type = "ROCm" if device_config == "rocm" else "CUDA"
            print(f"⚠️  警告：强制使用 {device_type}，但 GPU 不可用，将回退到 CPU")
            if device_config == "rocm" and not rocm_available:
                print("   提示：检测到系统有 ROCm，但 PyTorch 可能未安装 ROCm 版本")
                print("   请安装 PyTorch ROCm 版本：pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
            if device_config == "rocm":
                print("✓ 使用 ROCm GPU 加速")
    elif device_config == "cpu":
        device = torch.device("cpu")
    else:  # auto
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # 检测是否是 ROCm
            if rocm_available:
                print("✓ 自动检测到 ROCm GPU，使用 GPU 加速")
            else:
                print("✓ 自动检测到 CUDA GPU，使用 GPU 加速")
        else:
            device = torch.device("cpu")
            print("⚠️  注意：未检测到 GPU，使用 CPU 模式")
            print(f"   PyTorch 版本: {torch.__version__}")
            if rocm_available:
                print("   ✓ 检测到系统有 ROCm")
                print("   ⚠️  但 PyTorch 未检测到 GPU 支持")
                print("   建议：安装 PyTorch ROCm 版本")
                print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0")
            elif hasattr(torch.version, 'cuda') and torch.version.cuda:
                print(f"   PyTorch 编译时支持 CUDA: {torch.version.cuda}")
                print("   可能原因：CUDA 驱动未安装或版本不匹配")
    
    router_model = router_model.to(device)
    
    print("\n" + "=" * 60)
    print("模型信息")
    print("=" * 60)
    print(f"Token Embedding:")
    print(f"  Model name: {token_embedding.model_name}")
    print(f"  Vocab size: {token_embedding.vocab_size}")
    print(f"  Embedding dim: {token_embedding.embedding_dim}")
    print(f"Mamba Encoder:")
    print(f"  d_model: {d_model}")
    print(f"  n_layers: {n_layers}")
    print(f"  d_state: {d_state}")
    print(f"  expand: {expand}")
    print(f"  share_encoder: {share_encoder}")
    print(f"Router:")
    print(f"  模式: 相似度匹配模式")
    print(f"Device: {device}")
    print(f"Precision: {precision}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Gradient clip norm: {gradient_clip_norm if gradient_clip_norm is not None else 'None (不裁剪)'}")
    print(f"Temperature: {temperature}")
    print("=" * 60)
    
    # 加载数据集
    print("\n" + "=" * 60)
    print("加载数据集")
    print("=" * 60)
    print(f"数据集: {dataset_name}")
    print(f"训练样本数量: {train_size if train_size is not None else '全部'}")
    
    raw_dataset = load_training_dataset(
        dataset_name=dataset_name,
        train_size=train_size
    )
    
    # 显示数据集信息
    if len(raw_dataset) > 0:
        sample = raw_dataset[0]
        print(f"\n数据集字段: {list(sample.keys())}")
        print(f"示例样本:")
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}...")
            else:
                print(f"  {key}: {value}")
    
    # 更新 raw_dataset（如果被分割了）
    # 注意：如果从训练集分割了验证集，raw_dataset 已经被更新
    
    # 创建数据集和 DataLoader（工具匹配模式）
    training_config = config.get("training", {})
    train_dataset = RouterDataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        max_length=training_config.get("max_length", 2048),
        truncation=training_config.get("truncation", True),
        padding=training_config.get("padding", "max_length"),
        precompute_tokens=training_config.get("precompute_tokens", False),
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # 数据加载并行数（增加以加速数据加载）
        pin_memory=True if device.type == 'cuda' else False,  # 加速数据传输
        persistent_workers=True if device.type == 'cuda' else False,  # 保持 worker 进程
    )
    
    # 创建验证集（如果有）
    eval_dataset = None
    eval_dataloader = None
    try:
        # 检查是否是本地文件
        dataset_path = Path(dataset_name)
        eval_raw = None
        
        if dataset_path.exists() and dataset_path.suffix == ".json":
            # 本地文件：尝试从同一目录加载验证集
            val_path = dataset_path.parent / dataset_path.stem.replace("train", "val").replace("_train", "_val")
            if not val_path.exists():
                val_path = dataset_path.parent / (dataset_path.stem + "_val.json")
            
            if val_path.exists():
                print(f"\n找到验证集文件: {val_path}")
                from datasets import Dataset
                with open(val_path, "r", encoding="utf-8") as f:
                    val_data = json.load(f)
                eval_raw = Dataset.from_list(val_data)
            else:
                # 如果没有验证集文件，从训练集中分割一部分作为验证集
                print("\n未找到验证集文件，从训练集中分割 10% 作为验证集")
                split_idx = int(len(raw_dataset) * 0.9)
                eval_raw = raw_dataset.select(range(split_idx, len(raw_dataset)))
                raw_dataset = raw_dataset.select(range(split_idx))
                # 重新创建训练集 Dataset
                train_dataset = RouterDataset(
                    dataset=raw_dataset,
                    tokenizer=tokenizer,
                    max_length=training_config.get("max_length", 2048),
                    truncation=training_config.get("truncation", True),
                    padding=training_config.get("padding", "max_length"),
                )
                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                )
                print(f"训练集: {len(raw_dataset)}, 验证集: {len(eval_raw)}")
        else:
            # HuggingFace 数据集
            full_dataset = load_dataset(dataset_name, cache_dir="./datasets")
            if "validation" in full_dataset or "val" in full_dataset:
                eval_split = "validation" if "validation" in full_dataset else "val"
                eval_raw = full_dataset[eval_split]
            else:
                # 如果没有验证集，从训练集中分割
                print("\n未找到验证集，从训练集中分割 10% 作为验证集")
                split_idx = int(len(raw_dataset) * 0.9)
                eval_raw = raw_dataset.select(range(split_idx, len(raw_dataset)))
                raw_dataset = raw_dataset.select(range(split_idx))
                # 重新创建训练集 Dataset
                train_dataset = RouterDataset(
                    dataset=raw_dataset,
                    tokenizer=tokenizer,
                    max_length=training_config.get("max_length", 2048),
                    truncation=training_config.get("truncation", True),
                    padding=training_config.get("padding", "max_length"),
                )
                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                )
                print(f"训练集: {len(raw_dataset)}, 验证集: {len(eval_raw)}")
        
        # 创建验证集 Dataset
        if eval_raw is not None:
            eval_dataset = RouterDataset(
                dataset=eval_raw,
                tokenizer=tokenizer,
                max_length=training_config.get("max_length", 2048),
                truncation=training_config.get("truncation", True),
                padding=training_config.get("padding", "max_length"),
                precompute_tokens=training_config.get("precompute_tokens", False),
            )
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
            print(f"\n验证集大小: {len(eval_dataset)}")
    except Exception as e:
        print(f"\n无法加载验证集: {e}")
        import traceback
        traceback.print_exc()
    
    # 创建优化器（支持 8-bit 优化器以节省显存）
    # 注意: bitsandbytes 不支持 ROCm，在 ROCm 环境下会自动回退到标准 Adam
    use_8bit_optimizer = config.get("training", {}).get("use_8bit_optimizer", False)
    is_rocm = hasattr(torch.version, 'hip') if hasattr(torch, 'version') else False
    
    if use_8bit_optimizer and not is_rocm:
        try:
            from bitsandbytes.optim import Adam8bit
            optimizer = Adam8bit(router_model.parameters(), lr=learning_rate)
            print("✓ 使用 8-bit 优化器（节省显存）")
        except ImportError:
            print("⚠️  未安装 bitsandbytes，回退到标准 Adam 优化器")
            print("   安装命令: pip install bitsandbytes (仅支持 CUDA)")
            optimizer = torch.optim.Adam(router_model.parameters(), lr=learning_rate)
        except Exception as e:
            print(f"⚠️  8-bit 优化器初始化失败: {e}")
            print("   回退到标准 Adam 优化器")
            optimizer = torch.optim.Adam(router_model.parameters(), lr=learning_rate)
    else:
        if use_8bit_optimizer and is_rocm:
            print("⚠️  8-bit 优化器不支持 ROCm，使用标准 Adam 优化器")
        optimizer = torch.optim.Adam(router_model.parameters(), lr=learning_rate)
    
    # 创建学习率调度器（如果启用）
    scheduler = None
    if run_mode == "train":
        # 计算总训练步数
        num_training_steps = len(train_dataloader) * num_epochs
        scheduler = create_lr_scheduler(optimizer, config, num_training_steps)
    
    # 尝试从 checkpoint 恢复
    start_epoch = 0
    start_step = 0
    best_accuracy = 0.0
    
    # 用于跟踪epoch损失趋势（用于诊断每轮损失上升问题）
    epoch_loss_history = []
    
    if resume and run_mode == "train":
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir, checkpoint_name)
        if latest_checkpoint:
            try:
                start_epoch, start_step, loaded_config, best_metric = load_checkpoint(
                    latest_checkpoint, router_model, optimizer, device=str(device), scheduler=scheduler
                )
                if best_metric is not None:
                    best_accuracy = best_metric
                print(f"✅ 从 checkpoint 恢复训练: epoch={start_epoch}, step={start_step}, best_accuracy={best_accuracy:.4f}")
            except Exception as e:
                print(f"⚠️  加载 checkpoint 失败: {e}")
                print("从头开始训练...")
        else:
            print("未找到 checkpoint，从头开始训练...")
    else:
        if run_mode == "train":
            print("从头开始训练...")
    
    # 训练循环
    if run_mode == "train":
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)
        
        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # 训练
            train_loss = train_one_epoch(
                model=router_model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                device=device,
                precision=precision,
                epoch=epoch,
                scheduler=scheduler,
                gradient_accumulation_steps=gradient_accumulation_steps,
                gradient_clip_norm=gradient_clip_norm,
                temperature=temperature,
            )
            
            print(f"训练损失: {train_loss:.4f}")
            
            # 跟踪epoch损失趋势（用于诊断每轮损失上升问题）
            epoch_loss_history.append(train_loss)
            if len(epoch_loss_history) > 1:
                # 检查损失趋势
                if len(epoch_loss_history) >= 2:
                    prev_loss = epoch_loss_history[-2]
                    curr_loss = epoch_loss_history[-1]
                    loss_change = curr_loss - prev_loss
                    loss_change_pct = (loss_change / prev_loss) * 100 if prev_loss > 0 else 0
                    
                    if loss_change > 0:
                        print(f"⚠️  损失上升: {prev_loss:.4f} → {curr_loss:.4f} (+{loss_change:.4f}, +{loss_change_pct:.1f}%)")
                        if len(epoch_loss_history) >= 3:
                            # 检查是否连续上升
                            if all(epoch_loss_history[i] < epoch_loss_history[i+1] for i in range(len(epoch_loss_history)-2, len(epoch_loss_history)-1)):
                                print(f"🔴 警告: 损失连续上升！")
                                print(f"   建议: 1) 进一步降低学习率 2) 检查数据质量 3) 检查模型初始化")
                    else:
                        print(f"✅ 损失下降: {prev_loss:.4f} → {curr_loss:.4f} ({loss_change:.4f}, {loss_change_pct:.1f}%)")
            
            # 评估
            if (epoch + 1) % eval_interval == 0 and eval_dataloader is not None:
                eval_metrics = evaluate(
                    model=router_model,
                    dataloader=eval_dataloader,
                    device=device,
                    precision=precision,
                    temperature=temperature,
                )
                print(f"验证损失: {eval_metrics['loss']:.4f}")
                print(f"验证准确率: {eval_metrics['accuracy']:.4f}")
                
                # 保存最佳模型
                if eval_metrics['accuracy'] > best_accuracy:
                    best_accuracy = eval_metrics['accuracy']
                    print(f"✅ 新的最佳准确率: {best_accuracy:.4f}")
            
            # 保存 checkpoint
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                checkpoint_name=checkpoint_name,
                model=router_model,
                optimizer=optimizer,
                epoch=epoch + 1,
                step=start_step + (epoch + 1) * len(train_dataloader),
                config=config,
                best_metric=best_accuracy,
                scheduler=scheduler,
            )
        
        print("\n" + "=" * 60)
        print("训练完成")
        print("=" * 60)
    
    # 评估模式
    elif run_mode == "eval":
        print("\n" + "=" * 60)
        print("评估模型")
        print("=" * 60)
        
        if eval_dataloader is None:
            print("使用训练集进行评估...")
            eval_dataloader = train_dataloader
        
        eval_metrics = evaluate(
            model=router_model,
            dataloader=eval_dataloader,
            device=device,
            precision=precision,
            temperature=temperature,
        )
        
        print(f"\n评估结果:")
        for key, value in eval_metrics.items():
            print(f"  {key}: {value:.4f}")
    
    # 推理模式
    elif run_mode == "inference":
        print("\n" + "=" * 60)
        print("推理测试")
        print("=" * 60)
        
        if test_text:
            print(f"测试文本: {test_text}")
        else:
            if len(raw_dataset) > 0:
                test_text = raw_dataset[0].get("text", str(raw_dataset[0]))
                print(f"使用数据集中的样本: {test_text}")
            else:
                test_text = "Share my location with Hillary's sister"
                print(f"使用默认测试文本: {test_text}")
        
        # 执行推理
        h_query = simple_inference(router_model, tokenizer, test_text, config)
        print(f"\n推理结果:")
        print(f"  查询向量 shape: {h_query.shape}")
        print(f"  向量 norm: {h_query.norm().item():.4f}")
        print(f"  向量样本 (前10维): {h_query[0, :10].tolist()}")
        
        print(f"\n注意：推理模式仅返回查询向量，需要配合工具向量进行相似度匹配")


if __name__ == "__main__":
    main()
