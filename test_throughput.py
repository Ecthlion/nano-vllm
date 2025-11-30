import torch
import time
import numpy as np
import psutil
import sys

def check_system_memory():
    """检查系统内存是否足够"""
    available_memory = psutil.virtual_memory().available / (1024**3)  # GB
    print(f"系统可用内存: {available_memory:.2f} GB")
    return available_memory

def test_large_data_transfer(data_size_gb=100):
    """测试大数据量传输"""
    if not torch.cuda.is_available():
        print("CUDA 不可用")
        return
    
    # 检查内存是否足够
    available_memory = check_system_memory()
    if available_memory < data_size_gb * 1.2:  # 留20%余量
        print(f"警告: 可用内存({available_memory:.2f}GB)不足，无法测试{data_size_gb}GB数据")
        return
    
    device = torch.device('cuda')
    
    print(f"\n开始测试 {data_size_gb}GB 数据传输")
    print("=" * 60)
    
    # 计算需要的元素数量 (使用 float32，每个元素4字节)
    element_size = 4  # float32 字节数
    num_elements = int(data_size_gb * 1024**3 / element_size)
    
    print(f"数据大小: {data_size_gb} GB")
    print(f"元素数量: {num_elements:,}")
    print(f"数据类型: float32")
    
    # 分块测试以避免内存碎片问题
    chunk_size_gb = 10  # 每次传输10GB
    chunk_elements = int(chunk_size_gb * 1024**3 / element_size)
    num_chunks = data_size_gb // chunk_size_gb
    
    print(f"分块传输: {num_chunks} 个 {chunk_size_gb}GB 块")
    
    # 测试 CPU -> GPU
    print("\n测试 CPU -> GPU 传输:")
    print("-" * 40)
    
    total_transfer_time = 0
    total_data_transferred = 0
    
    for chunk in range(num_chunks):
        print(f"传输块 {chunk + 1}/{num_chunks}...")
        
        # 创建 CPU 数据
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated()
        
        cpu_data = torch.randn(chunk_elements, dtype=torch.float32)
        
        # 预热和同步
        torch.cuda.synchronize()
        start_time = time.time()
        
        # 传输到 GPU
        gpu_data = cpu_data.to(device)
        torch.cuda.synchronize()
        
        end_time = time.time()
        end_mem = torch.cuda.memory_allocated()
        
        chunk_time = end_time - start_time
        chunk_gb = chunk_size_gb
        chunk_throughput = chunk_gb / chunk_time
        
        total_transfer_time += chunk_time
        total_data_transferred += chunk_gb
        
        print(f"  块 {chunk + 1}: {chunk_time:.3f}s, {chunk_throughput:.2f} GB/s")
        
        # 清理内存
        del cpu_data, gpu_data
        torch.cuda.empty_cache()
    
    avg_throughput = total_data_transferred / total_transfer_time
    print(f"\nCPU->GPU 平均吞吐量: {avg_throughput:.2f} GB/s")
    print(f"总传输时间: {total_transfer_time:.3f} 秒")
    print(f"总数据量: {total_data_transferred:.1f} GB")

def test_single_large_transfer(data_size_gb=100):
    """单次大传输测试"""
    if not torch.cuda.is_available():
        return
    
    print(f"\n单次 {data_size_gb}GB 传输测试")
    print("=" * 60)
    
    device = torch.device('cuda')
    element_size = 4  # float32
    num_elements = int(data_size_gb * 1024**3 / element_size)
    
    try:
        # 创建大数组
        print("创建 CPU 数据...")
        cpu_data = torch.randn(num_elements, dtype=torch.float32)
        
        # 预热
        _ = torch.randn(1000, device=device)
        torch.cuda.synchronize()
        
        print("开始传输...")
        start_time = time.time()
        
        # 传输到 GPU
        gpu_data = cpu_data.to(device)
        torch.cuda.synchronize()
        
        end_time = time.time()
        
        transfer_time = end_time - start_time
        throughput = data_size_gb / transfer_time
        
        print(f"单次传输结果:")
        print(f"  传输时间: {transfer_time:.3f} 秒")
        print(f"  吞吐量: {throughput:.2f} GB/s")
        
        # 测试 GPU -> CPU
        print("\n测试 GPU -> CPU 传输...")
        start_time = time.time()
        
        cpu_data_back = gpu_data.cpu()
        torch.cuda.synchronize()
        
        end_time = time.time()
        
        transfer_time_back = end_time - start_time
        throughput_back = data_size_gb / transfer_time_back
        
        print(f"GPU->CPU 结果:")
        print(f"  传输时间: {transfer_time_back:.3f} 秒")
        print(f"  吞吐量: {throughput_back:.2f} GB/s")
        
    except RuntimeError as e:
        print(f"内存不足: {e}")
        return

def test_different_batch_sizes():
    """测试不同批次大小的性能"""
    if not torch.cuda.is_available():
        return
    
    print(f"\n不同批次大小性能测试")
    print("=" * 60)
    
    device = torch.device('cuda')
    total_size_gb = 50  # 总测试数据量
    element_size = 4
    
    batch_sizes_gb = [1, 2, 5, 10, 20]  # 不同的批次大小
    
    for batch_gb in batch_sizes_gb:
        num_batches = total_size_gb // batch_gb
        batch_elements = int(batch_gb * 1024**3 / element_size)
        
        print(f"\n批次大小: {batch_gb}GB, 批次数量: {num_batches}")
        print("-" * 40)
        
        total_time = 0
        
        for batch in range(num_batches):
            # 创建和传输数据
            cpu_data = torch.randn(batch_elements, dtype=torch.float32)
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            gpu_data = cpu_data.to(device)
            torch.cuda.synchronize()
            
            batch_time = time.time() - start_time
            total_time += batch_time
            
            # 清理
            del cpu_data, gpu_data
        
        avg_throughput = total_size_gb / total_time
        print(f"平均吞吐量: {avg_throughput:.2f} GB/s")
        print(f"总时间: {total_time:.3f} 秒")

if __name__ == "__main__":
    print("100GB 大数据传输测试")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 检查系统资源
        check_system_memory()
        
        # 运行测试
        test_large_data_transfer(100)  # 100GB 测试
        
        # 可以根据系统内存调整测试大小
        # test_single_large_transfer(50)  # 如果100GB太大，测试50GB
        
        # test_different_batch_sizes()
        
    else:
        print("未检测到 GPU")
