"""
Streamlit front-end for the Brain-MRI U-NET demo.
Run:  streamlit run streamlit_app.py
"""
from pathlib import Path
import streamlit as st
import numpy as np
import cv2

from src.io_utils import load_nifti
from src.preprocessing import preprocess_volume, volume_to_slices
from src.inference import get_model, predict_slice, find_max_tumor_slice
from src.visualization import overlay_mask

THRESHOLD = 0.30  # global probability threshold

st.set_page_config(page_title="Brain MRI Tumor Segmentation", layout="wide")
st.title("Brain MRI Tumor Segmentation – U‑NET Demo")

file = st.file_uploader("Upload a .nii or .nii.gz file", type=["nii", "gz"])

if file is not None:
    with st.spinner("Loading and preprocessing …"):
        raw_volume = load_nifti(file)               # keep original
        proc_volume = preprocess_volume(raw_volume) # resized + cropped
        slices      = volume_to_slices(proc_volume)

    model = get_model()

    with st.spinner("Running inference …"):
        masks, tumor_fracs = [], []
        for sl in slices:
            m, f = predict_slice(model, sl, thresh=THRESHOLD)
            masks.append(m)
            tumor_fracs.append(f)

        idx = find_max_tumor_slice(tumor_fracs)

    st.subheader(f"Slice with largest tumor area (index {idx})")
    col1, col2 = st.columns(2)
    with col1:
        img_norm = cv2.normalize(slices[idx], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)
        st.image(img_rgb,
             caption=f"Original FLAIR (slice {idx})",
             clamp=True)
    with col2:
        st.image(overlay_mask(slices[idx], masks[idx]),
                 caption=f"Predicted mask (threshold {THRESHOLD})")

    st.info(f"Tumor pixels on this slice: {tumor_fracs[idx]*100:.2f} %")
else:
    st.warning("Please upload a .nii or .nii.gz file.")
