"""Visualization helpers: overlay binary mask on grayscale slice."""
from typing import Union
import numpy as np
import cv2
from matplotlib import cm

__all__ = ["overlay_mask"]

def overlay_mask(img: np.ndarray, mask: np.ndarray, *, alpha: float = 0.4) -> np.ndarray:
    """Return an RGB image: grayscale slice with colored mask overlay.

    Parameters
    ----------
    img  : np.ndarray (H, W)
    mask : np.ndarray (H, W) uint8 or bool
    alpha: blending coefficient for the mask layer.
    """
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)

    cmap = cm.get_cmap("autumn")
    colored_mask = (cmap(mask.astype(float))[:, :, :3] * 255).astype(np.uint8)

    overlay = cv2.addWeighted(img_rgb, 1.0, colored_mask, alpha, 0)
    return overlay
