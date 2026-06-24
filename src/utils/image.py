"""
Image utility functions.
"""

import numpy as np


def tensor_to_np(t):
    """Convert a [C, H, W] float tensor (0-1) to a uint8 numpy array [H, W, C]."""
    return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
