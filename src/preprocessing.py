"""
Pre-processing identical to the training pipeline.
"""
from typing import List, Tuple

import cv2
import numpy as np

TARGET_XY = (240, 240)      # spatial resolution
CROP_Z    = (3, 147)        # keep 144 slices: 3 … 146
__all__   = ["preprocess_volume", "volume_to_slices", "CROP_Z"]


def _resize(slice_: np.ndarray,
            target_xy: Tuple[int, int] = TARGET_XY) -> np.ndarray:
    """Bilinear resize of a single slice."""
    return cv2.resize(slice_, target_xy, interpolation=cv2.INTER_LINEAR)


def preprocess_volume(volume: np.ndarray) -> np.ndarray:
    """
    Resize H×W to 240×240 and crop along Z to 144 slices.
    Returns volume of shape (240, 240, 144)
    """
    h, w, z = volume.shape
    resized = np.zeros((*TARGET_XY, z), dtype=np.float32)
    for k in range(z):
        resized[:, :, k] = _resize(volume[:, :, k])

    z0, z1 = CROP_Z
    return resized[:, :, z0:z1]


def volume_to_slices(volume: np.ndarray) -> List[np.ndarray]:
    """Split 3-D volume into list of 2-D slices."""
    return [volume[:, :, k] for k in range(volume.shape[2])]