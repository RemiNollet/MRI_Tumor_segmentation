"""
Pre-processing functions identical to the training pipeline.
"""
from typing import List, Tuple

import cv2
import numpy as np

__all__ = ["preprocess_volume", "volume_to_slices"]

TARGET_XY = (240, 240)      # spatial size after resize
CROP_Z    = (3, 147)        # keep slices 3 … 146 → 144 slices


def _resize_slice(slice_: np.ndarray,
                  target_xy: Tuple[int, int] = TARGET_XY) -> np.ndarray:
    """Resize a single axial slice to `target_xy` using bilinear interpolation."""
    return cv2.resize(slice_, target_xy, interpolation=cv2.INTER_LINEAR)


def preprocess_volume(volume: np.ndarray) -> np.ndarray:
    """
    Resize (H, W) to 240 × 240 and crop along Z to 144 slices.

    Returns
    -------
    vol : np.ndarray, shape (240, 240, 144)
    """
    h, w, z = volume.shape
    resized = np.zeros((*TARGET_XY, z), dtype=np.float32)

    for k in range(z):
        resized[:, :, k] = _resize_slice(volume[:, :, k])

    z_start, z_end = CROP_Z
    return resized[:, :, z_start:z_end]


def volume_to_slices(volume: np.ndarray) -> List[np.ndarray]:
    """Convert a 3-D volume to a list of 2-D slices."""
    return [volume[:, :, k] for k in range(volume.shape[2])]