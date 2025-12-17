"""
设备诊断脚本：检查 CUDA/ROCm 可用性和配置
"""
import torch
import sys
import subprocess

print("=" * 60)
print("PyTorch 设备诊断")
print("=" * 60)

print(f"\nPyTorch 版本: {torch.__version__}")

# 检查 CUDA/ROCm 编译支持
if hasattr(torch.version, 'cuda'):
    cuda_version = torch.version.cuda
    print(f"PyTorch 编译时 CUDA/ROCm 版本: {cuda_version if cuda_version else 'N/A'}")
else:
    print("PyTorch 编译时 CUDA/ROCm 版本: N/A")

# 检查 ROCm
rocm_available = False
rocm_info = None
try:
    result = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=2)
    rocm_available = result.returncode == 0
    if rocm_available:
        rocm_info = result.stdout
except:
    pass

print(f"\nROCm 系统检测: {'✓ 可用' if rocm_available else '✗ 不可用'}")

# 检查 GPU 运行时可用性（ROCm 版本的 PyTorch 也使用 torch.cuda API）
gpu_available = torch.cuda.is_available()
print(f"GPU 运行时可用 (torch.cuda.is_available()): {gpu_available}")

if gpu_available:
    print(f"\n✓ GPU 设备信息:")
    print(f"  设备数量: {torch.cuda.device_count()}")
    print(f"  当前设备: {torch.cuda.current_device()}")
    print(f"  设备名称: {torch.cuda.get_device_name(0)}")
    try:
        print(f"  计算能力: {torch.cuda.get_device_capability(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU 内存: {props.total_memory / 1024**3:.2f} GB")
    except Exception as e:
        print(f"  无法获取详细信息: {e}")
    
    # 判断是 CUDA 还是 ROCm
    if rocm_available:
        print(f"\n  → 检测到 ROCm 系统，PyTorch 可能使用 ROCm 后端")
    else:
        print(f"\n  → 可能是 NVIDIA CUDA GPU")
else:
    print("\n⚠️  GPU 不可用的可能原因：")
    if rocm_available:
        print("1. ✓ 系统有 ROCm，但 PyTorch 未安装 ROCm 版本")
        print("2. PyTorch 安装的是 CUDA 版本（当前版本）")
        print("\n建议：")
        print("  安装 PyTorch ROCm 版本：")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0")
        print("  或")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7")
    else:
        print("1. 系统没有 GPU")
        print("2. GPU 驱动未安装或版本过旧")
        print("3. PyTorch 安装的是 CPU 版本")
        print("\n建议检查：")
        print("- 运行 'nvidia-smi' 检查 NVIDIA GPU")
        print("- 运行 'rocminfo' 检查 AMD GPU")
        print("- 如果确实没有 GPU，可以在 config.json 中设置 'device': 'cpu'")

print("\n" + "=" * 60)
print("当前推荐设备配置")
print("=" * 60)
if gpu_available:
    if rocm_available:
        print("推荐: device = 'auto' 或 'rocm'")
    else:
        print("推荐: device = 'auto' 或 'cuda'")
    print("将使用 GPU 加速")
elif rocm_available:
    print("推荐: 先安装 PyTorch ROCm 版本，然后使用 device = 'rocm'")
    print("当前: device = 'cpu'（将使用 CPU，可能较慢）")
else:
    print("推荐: device = 'cpu'")
    print("将使用 CPU（可能较慢）")

