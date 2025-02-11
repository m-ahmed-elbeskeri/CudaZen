# **CudaZenTranspiler: A DSL-to-C++ CUDA Transpiler**  

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
[![CUDA](https://img.shields.io/badge/CUDA-Supported-green.svg)](https://developer.nvidia.com/cuda-toolkit)  

---

## **Overview**  
**CudaZenTranspiler** is a powerful **Domain-Specific Language (DSL) to CUDA C++ transpiler** that simplifies GPU programming by providing a high-level abstraction over CUDA memory management, kernel launches, and graph execution.  

This transpiler converts easy-to-read DSL code into optimized CUDA C++ code while automatically handling allocations, memory transfers, kernel launches, and CUDA graphs.  

### **Key Features**
✅ **Function Modifiers**: Define functions as `kernel`, `gpu_only`, `cpu_only`, or `gpu_and_cpu`  
✅ **Memory Management**: Supports **CPU, GPU, Pinned, and Unified** memory allocations with optional `manual_delete`  
✅ **Automatic Memory Deallocation**: Scoped memory management ensures proper cleanup  
✅ **Async Memory Copies**: `copy_to_gpu_async()` and `copy_to_cpu_async()` with optional stream usage  
✅ **Kernel Launches**: Simple syntax to launch GPU kernels  
✅ **Graph Capturing & Execution**: Define graphs with `graph(name, stream) {}` and execute them via `launch_graph(name);`  
✅ **Thread Synchronization**: `synchronize;` for easy device synchronization  
✅ **Thread Indexing**: `global_thread_index(x|y|z)` converts to CUDA thread calculations  
✅ **Out-of-Bounds Protection**: `if_in_bounds()` macros for safe execution  

---

## **Installation & Usage**  
### **Prerequisites**
- CUDA Toolkit **11.0+**  
- C++ Compiler with CUDA support (e.g., `nvcc`)  
- Python **3.x** (for transpiling DSL code)  

### **Installation**
Clone this repository:  
```bash
git clone https://github.com/your-username/CudaZenTranspiler.git
cd CudaZenTranspiler
```

---

## **Getting Started**
### **Example DSL Code**
Write CUDA in a more readable **DSL**:
```cpp
debug_mode(true);
set_device(0);

#N = 1000;

cpu_only vector_add_cpu(float* a, float* b, float* c, int n){
  for(int i = 0; i < n; i++){
    c[i] = a[i] + b[i];
  }
}

kernel vector_add_gpu(float* a, float* b, float* c, int n){
  int i = global_thread_index(x);
  if(i < n){
    c[i] = a[i] + b[i];
  }
}

int main() {
  cpu float* h_a = alloc_cpu<float>(N);
  gpu float* d_a = alloc_gpu<float>(N);

  launch vector_add_gpu(d_a, h_a, d_a, N) with { threads:256, blocks:10 };
  synchronize;

  return 0;
}
```

### **Transpile the DSL Code**
```python
from CudaZenTranspiler import CudaZenTranspiler

dsl_code = """(Insert DSL Code Here)"""
transpiler = CudaZenTranspiler(dsl_code)
cuda_code = transpiler.transpile()
print(cuda_code)
```

---

## **Features Explained**
### **1. Function Modifiers**
Define functions with different execution properties:
| DSL Keyword | CUDA Equivalent | Execution Location |
|-------------|----------------|--------------------|
| `cpu_only` | `__host__` | CPU only |
| `gpu_only` | `__device__` | GPU only |
| `kernel` | `__global__` | GPU Kernel |
| `gpu_and_cpu` | `__host__ __device__` | Runs on both CPU & GPU |

Example:
```cpp
gpu_and_cpu void multiply(float* arr, int N) {
    for(int i = 0; i < N; i++) {
        arr[i] *= 2;
    }
}
```

---

### **2. Memory Allocations**
Allocate memory with easy-to-read syntax:
```cpp
cpu float* h_data = alloc_cpu<float>(N);
gpu float* d_data = alloc_gpu<float>(N);
pinned float* pinnedBuf = alloc_pinned<float>(256);
unified float* sharedBuf = alloc_unified<float>(N);
```
**Automatic Cleanup:** Non-`manual_delete` variables are automatically deallocated at scope exit.

Manual deallocation:
```cpp
deallocate(d_data);
```

---

### **3. Async Memory Copies**
Transfer data efficiently using streams:
```cpp
copy_to_gpu_async(d_data, h_data, N) on stream1;
copy_to_cpu_async(h_data, d_data, N);
```

---

### **4. Kernel Launches**
Launch a kernel in one line:
```cpp
launch vector_add_gpu(d_a, h_a, d_c, N) with { threads:256, blocks:10 };
```
Optional stream and shared memory:
```cpp
launch vector_add_gpu(d_a, d_b, d_c, N) with { threads:256, blocks:10, shared_mem:1024, stream:stream1 };
```

---

### **5. CUDA Graphs**
Capture and execute a CUDA graph:
```cpp
graph(myGraph, stream1) {
    launch vector_add_gpu(d_a, d_b, d_c, N) with { threads:256, blocks:10 };
}
launch_graph(myGraph);
```

---

### **6. Synchronization**
Ensure all operations complete before continuing:
```cpp
synchronize;
```

---

### **7. Thread Indexing**
Use `global_thread_index(x|y|z)` instead of writing complex CUDA expressions:
```cpp
int i = global_thread_index(x);  // Expands to blockIdx.x * blockDim.x + threadIdx.x
```

---

### **8. Safe Execution with Bounds Checking**
Prevent out-of-bounds errors:
```cpp
if_in_bounds(i, 0, N) {
    c[i] = a[i] + b[i];
}
```
Supports **1D, 2D, and 3D** bounds checking:
```cpp
if_in_bounds2D(x, y, 0, width, 0, height) { }
if_in_bounds3D(x, y, z, 0, width, 0, height, 0, depth) { }
```

---

## **Performance & Debugging**
✅ **Debug Mode:** `debug_mode(true);` enables CUDA error checking.  
✅ **Optimized Kernel Launches:** Minimizes unnecessary synchronization.  
✅ **Scoped Memory Management:** Prevents memory leaks.  
✅ **Graph Execution:** Speeds up repeated workloads.  

---

## **Why Use CudaZenTranspiler?**
✅ **Simplicity** – No need to write complex CUDA boilerplate.  
✅ **Safety** – Prevents memory leaks with scoped deallocation.  
✅ **Performance** – Supports CUDA graphs & async operations.  
✅ **Productivity** – Write less code, achieve more!  

---

## **Contributing**
Want to contribute?  
1. Fork this repository  
2. Create a feature branch (`git checkout -b feature-name`)  
3. Commit changes (`git commit -m "Add new feature"`)  
4. Push to branch (`git push origin feature-name`)  
5. Open a Pull Request  

---

## **License**
This project is **MIT Licensed** – free for both commercial & personal use.

---

## **Contact**
🚀 Created by Mohamed Ahmed
📧 Email: Mohamed.ahmed.4894@gmail.com
🔗 GitHub: [Your GitHub Profile](https://github.com/your-username)  

---

⭐ **If you like this project, give it a star on GitHub!** ⭐
