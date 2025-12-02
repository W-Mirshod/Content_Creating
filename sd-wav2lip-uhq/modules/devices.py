"""
Minimal stub implementation of modules.devices for standalone sd-wav2lip-uhq usage.
"""
import torch
import gc


def torch_gc():
    """Clear PyTorch cache and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

