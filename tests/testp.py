import torch

# 打印 PyTorch 版本
print(f"PyTorch 版本: {torch.__version__}")

# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA 是否可用: {cuda_available}")

if cuda_available:
    # 打印 CUDA 版本（由 PyTorch 编译时使用的 CUDA 版本）
    print(f"PyTorch 编译时使用的 CUDA 版本: {torch.version.cuda}")

    # 打印当前 GPU 数量
    print(f"GPU 数量: {torch.cuda.device_count()}")

    # 打印当前 GPU 名称
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # 设置默认 GPU（可选）
    device = torch.device("cuda:0")
    print(f"当前使用的设备: {device}")

    # 简单张量运算测试
    x = torch.rand(3, 3).to(device)
    y = torch.rand(3, 3).to(device)
    z = torch.mm(x, y)
    print("GPU 上的矩阵乘法成功执行！")
else:
    print("CUDA 不可用，将使用 CPU。")
    device = torch.device("cpu")