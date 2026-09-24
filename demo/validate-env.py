import platform
import torch
import transformers
import vllm

from importlib.metadata import version

import lmcache.cuda_ops

print("architecture:", platform.machine())
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("vLLM:", vllm.__version__)
print("LMCache:", version("lmcache"))
print("Transformers:", transformers.__version__)

# Auto-selected backend [cuda] for accelerator 'cuda' from candidate backends [cuda, rocm]. (__init__.py:148:lmcache.v1.platform)
# architecture: aarch64
# GPU: NVIDIA GH200 480GB
# PyTorch: 2.13.0+cu130
# PyTorch CUDA: 13.0
# vLLM: 0.27.1
# LMCache: 0.5.4
# Transformers: 5.12.1
