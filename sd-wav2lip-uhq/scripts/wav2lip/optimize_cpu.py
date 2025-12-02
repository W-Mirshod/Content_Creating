"""
CPU Optimization settings for maximum multi-core performance.
Import this module at the start of processing to enable all CPU optimizations.

Optimized for AMD EPYC and high-core-count CPUs.
"""
import os
import multiprocessing

# Get the number of CPU cores
NUM_CORES = multiprocessing.cpu_count()
NUM_THREADS = str(NUM_CORES)

# Set environment variables BEFORE importing numpy/torch
# These must be set before numpy is imported
os.environ['OMP_NUM_THREADS'] = NUM_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = NUM_THREADS
os.environ['MKL_NUM_THREADS'] = NUM_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = NUM_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = NUM_THREADS

# OpenMP settings for better scaling on AMD EPYC
os.environ['OMP_DYNAMIC'] = 'FALSE'
os.environ['OMP_SCHEDULE'] = 'STATIC'
os.environ['OMP_PROC_BIND'] = 'SPREAD'  # Spread threads across NUMA nodes
os.environ['OMP_PLACES'] = 'threads'

# GOMP (GCC OpenMP) settings
os.environ['GOMP_CPU_AFFINITY'] = f'0-{NUM_CORES-1}'

# Intel MKL settings (if using)
os.environ['MKL_DYNAMIC'] = 'FALSE'

# For better NUMA performance on EPYC
os.environ['MALLOC_TRIM_THRESHOLD_'] = '128000'

# Now import and configure
import cv2
import torch

# OpenCV optimizations
cv2.setNumThreads(NUM_CORES)
cv2.setUseOptimized(True)

# PyTorch threading (for CPU operations)
# These must be set before any parallel work starts
try:
    torch.set_num_threads(NUM_CORES)
    print(f"[CPU OPTIMIZATION] PyTorch threads set to {NUM_CORES}")
except RuntimeError as e:
    print(f"[CPU OPTIMIZATION] PyTorch threads already configured: {e}")

try:
    torch.set_num_interop_threads(min(NUM_CORES, 8))  # Limit inter-op parallelism
    print(f"[CPU OPTIMIZATION] PyTorch interop threads set to {min(NUM_CORES, 8)}")
except RuntimeError as e:
    if "cannot set" in str(e).lower():
        print(f"[CPU OPTIMIZATION] PyTorch interop threads already configured")
    else:
        print(f"[CPU OPTIMIZATION] Warning setting interop threads: {e}")

# Enable TF32 for better GPU performance if available
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

print(f"[CPU OPTIMIZATION] Enabled {NUM_CORES} cores for parallel processing")
print(f"[CPU OPTIMIZATION] OpenCV threads: {cv2.getNumThreads()}")
print(f"[CPU OPTIMIZATION] PyTorch threads: {torch.get_num_threads()}")
