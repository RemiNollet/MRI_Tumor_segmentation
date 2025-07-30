"""
Model loading and per-slice prediction logic.
"""
from pathlib import Path
from typing import Tuple, List

import numpy as np
import tensorflow as tf
import keras
from keras.layers import Conv2DTranspose as _KConv2DT

THRESH_DEFAULT = 0.30
MODEL_PATH     = Path(__file__).resolve().parents[1] / "models" / "unet_best.h5"

_model_cache: tf.keras.Model | None = None   # loaded once per session

# Patch the from_config method only once
if not hasattr(_KConv2DT, "_patched_groups"):
    orig_from_config = _KConv2DT.from_config

    def _from_config_ignore_groups(cls, config):
        # Pop unknown "groups" key if present
        config.pop("groups", None)
        return orig_from_config(config)

    _KConv2DT.from_config = classmethod(_from_config_ignore_groups)
    _KConv2DT._patched_groups = True

# ---------------------------------------------------------------------
# Model handling
# ---------------------------------------------------------------------
def get_model() -> tf.keras.Model:
    """Lazy-load and cache the pretrained U-NET."""
    global _model_cache
    if _model_cache is None:
        _model_cache = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model_cache


# ---------------------------------------------------------------------
# Slice-level helpers
# ---------------------------------------------------------------------
def _normalise(slice_: np.ndarray) -> np.ndarray:
    """Min-max normalisation per slice."""
    return slice_ / slice_.max() if slice_.max() > 0 else slice_


def predict_slice(
    model: tf.keras.Model,
    img: np.ndarray,
    *,
    thresh: float = THRESH_DEFAULT
) -> Tuple[np.ndarray, float]:
    """
    Run inference on a single slice.

    Returns
    -------
    mask_bin : np.ndarray (H, W) uint8
        Binary mask after thresholding.
    frac     : float
        Fraction of pixels predicted as tumor.
    """
    inp  = _normalise(img)[None, ..., None]              # (1, H, W, 1)
    pred = model.predict(inp, verbose=0)[0]              # softmax (H, W, 4)

    tumor_prob = pred[..., 1:].sum(axis=-1)              # sum of classes 1-3
    mask_bin   = (tumor_prob > thresh).astype(np.uint8)

    return mask_bin, float(mask_bin.mean())


def find_max_tumor_slice(fracs: List[float]) -> int:
    """Return the index of the slice with the largest tumor fraction."""
    return int(np.argmax(fracs))