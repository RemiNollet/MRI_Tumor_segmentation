"""
Model loading and per-slice prediction.
"""
from pathlib import Path
from typing import Tuple, List

import numpy as np
import tensorflow as tf

THRESH_DEFAULT = 0.30
MODEL_PATH = Path(__file__).resolve().parents[1] / "checkpoints" / "unet_best.h5"

# ---------------------------------------------------------------------
# Patch for TF 2.20 deserialization (remove unsupported 'groups')
# ---------------------------------------------------------------------
from keras.layers import Conv2DTranspose as _KConv2DT

if not hasattr(_KConv2DT, "_patched_groups"):
    _orig_from_config = _KConv2DT.from_config

    def _from_config_no_groups(cls, cfg):   # noqa: N802
        cfg.pop("groups", None)
        return _orig_from_config(cfg)

    _KConv2DT.from_config = classmethod(_from_config_no_groups)
    _KConv2DT._patched_groups = True  # type: ignore

# ---------------------------------------------------------------------
# Cached model
# ---------------------------------------------------------------------
_model_cache: tf.keras.Model | None = None


def get_model() -> tf.keras.Model:
    """Load U-NET weights once per session."""
    global _model_cache
    if _model_cache is None:
        _model_cache = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model_cache


# ---------------------------------------------------------------------
# Slice inference helpers
# ---------------------------------------------------------------------
def _normalise(img: np.ndarray) -> np.ndarray:
    """Min-max per slice."""
    return img / img.max() if img.max() > 0 else img


def predict_slice(
    model: tf.keras.Model,
    img:   np.ndarray,
    *,
    thresh: float = THRESH_DEFAULT
) -> Tuple[np.ndarray, float]:
    """
    Predict one axial slice.
    Returns binary mask and tumor-pixel fraction.
    """
    inp  = _normalise(img)[None, ..., None]          # (1,H,W,1)
    pred = model.predict(inp, verbose=0)[0]          # (H,W,4)

    tumor_prob = pred[..., 1:].sum(axis=-1)
    mask = (tumor_prob > thresh).astype(np.uint8)
    return mask, float(mask.mean())


def find_max_tumor_slice(fracs: List[float]) -> int:
    """Index with max tumor fraction."""
    return int(np.argmax(fracs))