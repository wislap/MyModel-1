# uv 下载加速指南

## 🚀 快速开始

### 方法 1: 使用镜像源（推荐）

```bash
# 设置 PyPI 镜像（用于包下载）
export UV_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/simple

# 安装 Python
uv python install 3.10
```

### 方法 2: 永久配置

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
echo 'export UV_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/simple' >> ~/.bashrc
source ~/.bashrc
```

---

## 📋 详细方法

### 1. 配置 PyPI 镜像源

uv 使用 PyPI 镜像来加速 Python 包的下载。虽然 Python 解释器本身从官方下载，但依赖包可以通过镜像加速。

#### 国内镜像源

| 镜像源 | URL |
|--------|-----|
| 清华大学 | `https://mirrors.tuna.tsinghua.edu.cn/pypi/simple` |
| 阿里云 | `https://mirrors.aliyun.com/pypi/simple` |
| 中科大 | `https://mirrors.ustc.edu.cn/pypi/simple` |
| 豆瓣 | `https://pypi.douban.com/simple` |

#### 使用方法

```bash
# 临时使用（当前终端会话）
export UV_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/simple
uv python install 3.10

# 永久配置（推荐）
echo 'export UV_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/simple' >> ~/.bashrc
source ~/.bashrc
```

#### 配置多个镜像源

```bash
# 主镜像
export UV_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/simple

# 备用镜像（如果主镜像失败）
export UV_EXTRA_INDEX_URL=https://pypi.org/simple
```

---

### 2. 使用 uv 配置文件

创建 `uv.toml` 配置文件（如果 uv 支持）：

```toml
[index]
url = "https://mirrors.tuna.tsinghua.edu.cn/pypi/simple"

[extra-index]
url = "https://pypi.org/simple"
```

**注意**: uv 的配置文件格式可能不同，请参考官方文档。

---

### 3. 优化并发下载

uv 默认使用并行下载，可以调整并发数：

```bash
# 设置并发下载数（默认通常是 4-8）
export UV_CONCURRENT_DOWNLOADS=10

# 安装 Python
uv python install 3.10
```

---

### 4. 使用缓存

uv 会自动缓存下载的文件，可以：

```bash
# 查看缓存位置
uv cache dir

# 清理缓存（如果需要）
uv cache clean

# 查看缓存大小
du -sh $(uv cache dir)
```

---

### 5. 使用代理（如果网络受限）

```bash
# 设置 HTTP 代理
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080

# 安装 Python
uv python install 3.10

# 取消代理
unset HTTP_PROXY
unset HTTPS_PROXY
```

---

### 6. 预下载 Python（离线安装）

如果需要离线安装：

```bash
# 在有网络的环境下载
uv python install 3.10

# 复制缓存到离线环境
# uv 的 Python 缓存通常在 ~/.local/share/uv/python/
```

---

## 🔧 针对不同场景的配置

### 场景 1: 中国用户（推荐配置）

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
export UV_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/simple
export UV_EXTRA_INDEX_URL=https://pypi.org/simple
export UV_CONCURRENT_DOWNLOADS=10
```

### 场景 2: 企业网络（需要代理）

```bash
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
export UV_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/simple
```

### 场景 3: 最大化速度

```bash
export UV_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/simple
export UV_CONCURRENT_DOWNLOADS=16
export UV_NETWORK_TIMEOUT=30
```

---

## 📊 性能优化建议

### 1. 选择合适的镜像源

- **国内用户**: 使用清华大学或阿里云镜像
- **国外用户**: 使用官方 PyPI 或就近的镜像
- **企业用户**: 使用内网镜像（如果有）

### 2. 调整并发数

- **网络带宽充足**: 增加并发数（10-16）
- **网络带宽有限**: 减少并发数（2-4）
- **默认**: 通常 4-8 个并发即可

### 3. 使用缓存

- uv 会自动缓存，无需手动管理
- 定期清理缓存可以释放空间
- 缓存可以加速重复安装

---

## 🐛 故障排除

### 问题 1: 镜像源不可用

```bash
# 尝试其他镜像
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple

# 或使用官方源
export UV_INDEX_URL=https://pypi.org/simple
```

### 问题 2: 下载速度慢

```bash
# 检查网络连接
ping mirrors.tuna.tsinghua.edu.cn

# 尝试不同的镜像
export UV_INDEX_URL=https://mirrors.ustc.edu.cn/pypi/simple

# 增加并发数
export UV_CONCURRENT_DOWNLOADS=16
```

### 问题 3: 代理配置问题

```bash
# 检查代理设置
echo $HTTP_PROXY
echo $HTTPS_PROXY

# 测试代理
curl -x $HTTP_PROXY https://pypi.org/simple
```

---

## 💡 最佳实践

### 推荐配置（中国用户）

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
cat >> ~/.bashrc << 'EOF'

# uv 加速配置
export UV_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/simple
export UV_EXTRA_INDEX_URL=https://pypi.org/simple
export UV_CONCURRENT_DOWNLOADS=10
EOF

source ~/.bashrc
```

### 验证配置

```bash
# 检查环境变量
echo $UV_INDEX_URL

# 测试安装
uv python install 3.10
```

---

## 📝 注意事项

1. **Python 解释器下载**: 
   - Python 解释器本身从官方下载，无法通过镜像加速
   - 但依赖包可以通过镜像加速

2. **镜像源选择**:
   - 选择距离近、稳定的镜像
   - 定期检查镜像是否可用

3. **并发数设置**:
   - 不要设置过高，可能导致服务器限流
   - 根据网络情况调整

4. **缓存管理**:
   - 定期清理缓存释放空间
   - 缓存可以加速重复操作

---

## 🔗 相关资源

- [uv 官方文档](https://github.com/astral-sh/uv)
- [PyPI 镜像列表](https://www.pypi.org/mirrors/)
- [清华大学镜像站](https://mirrors.tuna.tsinghua.edu.cn/)

---

**最后更新**: 2024-12-17

