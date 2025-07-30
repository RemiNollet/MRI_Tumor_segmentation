"""
Overlay a binary mask on a grayscale slice.
"""
from typing import Union
import numpy as np
import cv2
from matplotlib import cm

__all__ = ["overlay_mask"]


def overlay_mask(img: np.ndarray,
                 mask: np.ndarray,
                 *,
                 alpha: float = 0.4) -> np.ndarray:
    """
    Return RGB image = grayscale slice + colored mask.
    """
    gray = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    cmap = cm.get_cmap("autumn")
    colored = (cmap(mask.astype(float))[:, :, :3] * 255).astype(np.uint8)

    return cv2.addWeighted(gray_rgb, 1.0, colored, alpha, 0)