"""
Streamlit front-end for the Brain-MRI U-NET demo.
Run:  streamlit run streamlit_app.py
"""
from pathlib import Path
import streamlit as st
import numpy as np

from src.io_utils       import load_nifti
from src.preprocessing  import preprocess_volume, volume_to_slices, CROP_Z
from src.inference      import get_model, predict_slice, find_max_tumor_slice
from src.visualization  import overlay_mask

THRESHOLD = 0.30  # probability threshold for tumor vs background

# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="Brain MRI Tumor Segmentation", layout="wide")
st.title("Brain MRI Tumor Segmentation – U-NET Demo")

file = st.file_uploader("Upload a .nii or .nii.gz file", type=["nii", "gz"])

if file is None:
    st.warning("Please upload a NIfTI file to start.")
    st.stop()

# ---------------------------------------------------------------------
# Load & preprocess
# ---------------------------------------------------------------------
with st.spinner("Loading and preprocessing …"):
    raw_volume  = load_nifti(file)               # original resolution
    proc_volume = preprocess_volume(raw_volume)  # 240×240×144
    slices      = volume_to_slices(proc_volume)

model = get_model()

# ---------------------------------------------------------------------
# Slice-by-slice inference
# ---------------------------------------------------------------------
with st.spinner("Running inference …"):
    masks, fracs = [], []
    for sl in slices:
        m, f = predict_slice(model, sl, thresh=THRESHOLD)
        masks.append(m)
        fracs.append(f)

idx_proc = find_max_tumor_slice(fracs)
idx_raw  = idx_proc + CROP_Z[0]   # map back to raw Z index

# ---------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------
st.subheader(f"Slice with largest tumor area (index {idx_proc})")

col1, col2 = st.columns(2)
with col1:
    st.image(raw_volume[:, :, idx_raw],
             clamp=True,
             caption=f"Original FLAIR (slice {idx_raw})")

with col2:
    st.image(overlay_mask(slices[idx_proc], masks[idx_proc]),
             caption=f"Mask overlay (threshold {THRESHOLD})")

st.info(f"Tumor pixels on this slice: {fracs[idx_proc]*100:.2f} %")