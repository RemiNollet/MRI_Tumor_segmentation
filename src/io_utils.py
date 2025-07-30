"""
I/O utilities – load NIfTI files for Streamlit or disk use.
"""
from pathlib import Path
from typing import Union
import tempfile

import nibabel as nib
import numpy as np
import streamlit as st  # only for type hints

PathLike = Union[str, Path]
__all__ = ["load_nifti"]


def _load(path: PathLike) -> np.ndarray:
    """Low-level loader from filesystem path."""
    img = nib.load(str(path))
    return img.get_fdata().astype(np.float32)


def load_nifti(file: Union[PathLike, "st.runtime.uploaded_file_manager.UploadedFile"]
               ) -> np.ndarray:
    """
    Load a NIfTI from disk *or* a Streamlit UploadedFile.
    Returns a float32 volume (H, W, Z).
    """
    if isinstance(file, (str, Path)):
        return _load(file)

    suffix = ".nii.gz" if file.name.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        tmp.flush()
        return _load(tmp.name)