from pathlib import Path
from typing import Union
import tempfile

import numpy as np
import nibabel as nib
import streamlit as st   # only for type check; no runtime dependency

PathLike = Union[str, Path]

__all__ = ["load_nifti"]


def _load_from_path(path: PathLike) -> np.ndarray:
    """Helper: load NIfTI volume from a filesystem path."""
    img = nib.load(str(path))
    return img.get_fdata().astype(np.float32)


def load_nifti(file: Union[PathLike, "st.runtime.uploaded_file_manager.UploadedFile"]) -> np.ndarray:
    """
    Load a NIfTI file (.nii / .nii.gz) from disk *or* a Streamlit upload.

    Parameters
    ----------
    file : str | pathlib.Path | streamlit.UploadedFile
        Path on disk **or** the object returned by `st.file_uploader`.

    Returns
    -------
    volume : np.ndarray, dtype float32, shape (H, W, Z)
    """
    # Case 1: already a path-like object
    if isinstance(file, (str, Path)):
        return _load_from_path(file)

    # Case 2: Streamlit Upload; write to a temporary file first
    suffix = ".nii.gz" if file.name.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        tmp.flush()
        volume = _load_from_path(tmp.name)

    return volume