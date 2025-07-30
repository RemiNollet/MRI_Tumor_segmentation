# src/io_utils.py
from pathlib import Path
from typing import Union
import tempfile
import nibabel as nib
import numpy as np

PathLike = Union[str, Path]
__all__ = ["load_nifti"]


def _load(path: PathLike) -> np.ndarray:
    img = nib.load(str(path))
    return img.get_fdata().astype(np.float32)


def load_nifti(file: Union[PathLike, "streamlit.uploaded_file_manager.UploadedFile"]
               ) -> np.ndarray:
    if isinstance(file, (str, Path)):
        return _load(file)

    suffix = ".nii.gz" if file.name.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        tmp.flush()
        return _load(tmp.name)