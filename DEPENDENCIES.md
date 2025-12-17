# 项目依赖说明

## 核心依赖

### 必需依赖

- **torch** (>=2.0.0): PyTorch 深度学习框架
- **transformers** (>=4.40.0,<4.50.0): HuggingFace 模型和 tokenizer 库
- **tokenizers** (>=0.19.0,<0.20.0): 分词器库（transformers 的依赖）
- **datasets** (>=4.4.1): HuggingFace 数据集库
- **safetensors** (>=0.4.0): 安全的模型序列化格式
- **tqdm** (>=4.67.1): 进度条显示
- **sentencepiece** (>=0.1.99): 某些 tokenizer 需要（如 Qwen）

### 可选依赖

- **torchvision** (>=0.15.0): 图像处理（某些模型可能需要）
- **modelscope** (>=1.9.0): 模型下载（主要用于中国用户）
- **bitsandbytes** (>=0.49.0): 8-bit 优化器（仅 CUDA，ROCm 不支持）

## 使用 uv 管理依赖

### 安装 uv

```bash
# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用 pip
pip install uv
```

### 基本使用

```bash
# 1. 创建虚拟环境
uv venv

# 2. 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate      # Windows

# 3. 安装项目依赖
uv pip install -e .

# 4. 安装可选依赖
uv pip install -e '.[bitsandbytes]'  # 8-bit 优化器
uv pip install -e '.[all]'            # 所有可选依赖
uv pip install -e '.[dev]'            # 开发依赖
```

### 同步依赖

```bash
# 从 pyproject.toml 同步依赖
uv pip sync
```

### 添加新依赖

```bash
# 安装新包
uv pip install <package>

# 然后手动添加到 pyproject.toml 的 dependencies 中
```

## 版本兼容性说明

### transformers 和 tokenizers

- `transformers>=4.40.0,<4.50.0`: 需要明确指定版本范围
- `tokenizers>=0.19.0,<0.20.0`: 与 transformers 4.40.0 兼容

### ROCm 支持

- **bitsandbytes**: 不支持 ROCm，仅在 CUDA 环境下可用
- **torch**: 需要安装 ROCm 版本的 PyTorch

## 开发依赖

开发时可选安装：

- **pytest** (>=7.0.0): 测试框架
- **black** (>=23.0.0): 代码格式化
- **ruff** (>=0.1.0): 代码检查
- **mypy** (>=1.0.0): 类型检查

安装开发依赖：

```bash
uv pip install -e '.[dev]'
```

## 依赖分析

项目主要使用以下第三方库：

1. **torch**: 深度学习框架
2. **transformers**: 模型和 tokenizer
3. **datasets**: 数据集加载
4. **safetensors**: 模型序列化
5. **tqdm**: 进度显示
6. **sentencepiece**: 分词器支持

## 故障排除

### 版本冲突

如果遇到版本冲突，可以：

1. 检查 `pyproject.toml` 中的版本约束
2. 使用 `uv pip list` 查看已安装的版本
3. 使用 `uv pip install --upgrade <package>` 升级包

### ROCm 环境

在 ROCm 环境下：

1. 不要安装 `bitsandbytes`
2. 使用 `precision: fp32` 而不是 `amp-fp16`
3. 确保安装了 ROCm 版本的 PyTorch

