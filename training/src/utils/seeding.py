"""
Deterministic seeding utilities for reproducibility.
"""
import os
import random
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: If True, enable deterministic algorithms (may slow down training)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For PyTorch >= 1.8
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass
    
    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def worker_init_fn(worker_id: int, base_seed: int = 42) -> None:
    """
    Worker init function for DataLoader to ensure different seeds per worker.
    """
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_rng_state() -> Dict:
    """Get current RNG states for checkpointing."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state: Dict) -> None:
    """Restore RNG states from checkpoint."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])