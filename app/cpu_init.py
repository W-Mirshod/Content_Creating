"""
Early initialization for CPU optimization.
This module must be imported BEFORE torch to set threading parameters.
"""
import os
import multiprocessing

# Get the number of CPU cores
NUM_CORES = multiprocessing.cpu_count()
NUM_THREADS = str(NUM_CORES)

# Set environment variables BEFORE torch is imported anywhere
os.environ['OMP_NUM_THREADS'] = NUM_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = NUM_THREADS
os.environ['MKL_NUM_THREADS'] = NUM_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = NUM_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = NUM_THREADS
os.environ['OMP_DYNAMIC'] = 'FALSE'
os.environ['OMP_SCHEDULE'] = 'STATIC'
os.environ['OMP_PROC_BIND'] = 'SPREAD'
os.environ['OMP_PLACES'] = 'threads'
os.environ['GOMP_CPU_AFFINITY'] = f'0-{NUM_CORES-1}'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['MALLOC_TRIM_THRESHOLD_'] = '128000'

print(f"[CPU_INIT] Environment configured for {NUM_CORES} CPU cores")
