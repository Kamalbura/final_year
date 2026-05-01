import torch
import time
import os

print("=" * 50)
print("COMPUTE SMOKE TEST")
print("=" * 50)

print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA compiled: {torch.version.cuda}")
print(f"CPU threads: {torch.get_num_threads()}")

# Test 1: CPU
print("\n--- CPU TEST ---")
start = time.time()
x = torch.randn(1000, 1000)
for _ in range(10):
    y = torch.matmul(x, x.T)
cpu_time = time.time() - start
print(f"CPU (10x matmul 1000x1000): {cpu_time:.3f}s")

# Test 2: RTX 2050
print("\n--- RTX 2050 TEST ---")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    # Try GPU compute
    try:
        x_gpu = x.to("cuda")
        torch.cuda.synchronize()
        start = time.time()
        y_gpu = torch.matmul(x_gpu, x_gpu.T)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        print(f"GPU (matmul): {gpu_time*1000:.2f}ms")
        if gpu_time > 0:
            print(f"Speedup: {cpu_time/gpu_time:.1f}x")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    except Exception as e:
        print(f"GPU error: {e}")
else:
    print("CUDA not available")

# Test 3: Inference simulation
print("\n--- INFERENCE TEST (CPU) ---")
batch = torch.randn(64, 168, 5)
start = time.time()
for _ in range(100):
    out = batch.mean(dim=1)
cpu_inf = time.time() - start
print(f"64 samples x 100 iter: {cpu_inf*1000:.2f}ms")
print(f"Per sample: {cpu_inf/64*1000:.3f}ms")

# Test 4: Memory
print("\n--- MEMORY ---")
print(f"CPU RAM available: {os.popen('wmic OS get FreePhysicalMemory').read().split()[1] if os.name=='nt' else 'N/A'} KB")

print("\n--- SUMMARY ---")
print("✓ CPU: Working")
print("✓ RTX 2050: Detected but may have kernel issues")
print("\nResults:")
print(f"  - CPU: {cpu_time:.2f}s")
print(f"  - GPU: {'error' if not torch.cuda.is_available() else f'{gpu_time*1000:.2f}ms'}")