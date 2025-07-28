import cupy as cp
import numpy as np

print("Testing GPU access with CuPy...")

try:
    # Create arrays
    cpu_array = np.array([1, 2, 3, 4, 5])
    gpu_array = cp.array(cpu_array)
    
    # Do some computation on GPU
    result_gpu = cp.square(gpu_array)
    
    # Move result back to CPU
    result_cpu = cp.asnumpy(result_gpu)
    
    print("\nTest Calculation:")
    print(f"Input array: {cpu_array}")
    print(f"Squared on GPU: {result_cpu}")
    print("\nGPU test successful!")
    
except Exception as e:
    print(f"\nError: {str(e)}")
    raise
