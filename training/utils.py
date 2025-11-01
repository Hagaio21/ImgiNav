"""
Training utility functions.
"""
import torch
import random
import numpy as np


def set_deterministic(seed: int = 42):
    """
    Set random seeds for reproducibility across Python, NumPy, PyTorch, and CUDA.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic algorithms (may be slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

