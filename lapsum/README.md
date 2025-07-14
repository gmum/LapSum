# LapSum

**Lapsum** is a high-performance PyTorch extension providing differentiable, GPU-accelerated operations for:
- Sorting
- Ranking
- Top-k selection
- Permutation

Designed for machine learning tasks requiring smooth approximations to discrete operations, Lapsum combines optimized C++/CUDA kernels with Python bindings through PyTorch extensions.

### Key Features

| Feature          | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `topk`          | Differentiable soft top-k and log-soft-top-k operators                      |
| `permute`       | Soft permutation operators and permutation loss                             |
| `sort`          | Differentiable soft sorting                                                |
| `rank`          | Differentiable soft ranking                                                |

## System Requirements

- **Python**: 3.9+
- **PyTorch**: 2.5+
- **Compiler**: C++14+ compatible
- **GPU Support**: CUDA Toolkit (optional but recommended)

### GPU Prerequisites

1. **CUDA Toolkit**: Needed for GPU acceleration. Install the CUDA Toolkit by following the instructions on the [official NVIDIA CUDA website](https://developer.nvidia.com/cuda-toolkit).

   - **Linux/Debian Systems**:
      ```bash
      # Update package lists
      sudo apt update
   
      # Install CUDA Toolkit (latest version)
      sudo apt install nvidia-cuda-toolkit
   
      # OR for specific version (e.g., 12.6)
      sudo apt install cuda-toolkit-12-6
   
      # Verify installation
      nvcc --version
      ```
2. **PyTorch**: Ensure that PyTorch is installed for your Python environment. You can install PyTorch by following the instructions at [PyTorch Installation Guide](https://pytorch.org/get-started/locally/).

---

## Installation Guide

#### Basic Installation
```bash
pip install .
```
#### Development Mode (Editable Install)
```bash
pip install -e .
```
#### Expected Output Files
Successful installation generates these compiled extensions:
```
lapsum/
  ├── topk/extension/
  │   ├── log_soft_topk_cpu.cpython-*.so
  │   ├── log_soft_topk_cuda.cpython-*.so
  │   ├── soft_topk_cpu.cpython-*.so
  │   └── soft_topk_cuda.cpython-*.so
```

#### Full Feature Installation
By default, only topk is installed. To enable all algorithms:
- Uncomment line 59 in [./setup.py](./setup.py#L59)
- Reinstall the package

Additional compiled files will appear in:
```
lapsum/
  ├── permute/extension/
  ├── rank/extension/
  └── sort/extension/
```

### Cleaning installation files 
```bash
rm -rf *.egg-info
find ./lapsum -type f -name "*.so" -delete
```

---

### Verification

```bash
PYTHONPATH=. python ./tests/test_topk.py
```
