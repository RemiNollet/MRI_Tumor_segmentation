# U-NET Brain MRI Tumor Segmentation (BraTS 2019)

End-to-end segmentation of brain tumors on FLAIR MRI slices using a 2-D U-NET.  
This repository contains a training/evaluation notebook and a lightweight web application for local inference and visualization.

## Project overview

A medical-imaging manufacturer would like to ship software that **automatically detects and segments brain tumors** on MRI scans, providing radiologists with an additional decision-support tool.  
The BraTS 2019 collection offers multi-contrast MRI volumes along with expert, voxel-level tumor annotations.  
The goal is to train an algorithm that, given the axial slices (Z-axis) of a patient scan, returns a mask indicating tumor pixels.

### Performance constraints

Because missing a tumor is clinically critical, **false negatives must be minimised**. We therefore set two global targets (computed pixel-wise over the full validation set):

| Metric | Definition | Required minimum |
| --- | --- | --- |
| **Sensitivity / Recall** | `TP / (TP + FN)` | **95%** |
| **Precision** | `TP / (TP + FP)` | **70%** |

A precision below 70% would not provide sufficient confidence, while recall below 95% would risk overlooking true lesions.

---

## Quick start (Streamlit app)

### 1) Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Download the pretrained model
1.	Download unet_best.h5 from Google Drive
https://drive.google.com/file/d/1HmYfTN_u3WyNwxJLp2943ULw1m2a0Ws4/view

2.	Place it under models/:
```bash
mkdir -p models
mv /path/to/unet_best.h5 models/unet_best.h5
```

### 3) Run
```bash
streamlit run streamlit_app.py
```

Upload a .nii or .nii.gz FLAIR volume. 
The app:
- Resizes to 240×240 and crops Z to [3:147] (144 slices)
- Runs the U-NET slice-by-slice
- Binarizes tumor as sum(prob[class 1..3]) > 0.30
- Finds the slice with the largest tumor area
- Displays original slice, preprocessed slice, and overlay

---

## Training and evaluation notebook

**notebooks/UNET_IRM_for_SageMaker.ipynb** covers dataset preparation, per-slice training, checkpointing, metric computation, and qualitative examples.\
For generalisation checks, the final app was also exercised on BraTS 2020 volumes (external test set example: see notebook notes).

## Probability threshold (0.30)

Initial observation: high precision (> 0.93) but recall ~ 0.90.
Adjustment: consider a pixel as tumor if the sum of non-background class probabilities exceeds 0.30.
- Before: Recall ≈ 0.90 / Precision ≈ 0.93
- After (0.30): Recall ≈ 0.96 / Precision ≈ 0.88 (≥ 70% requirement)

This configuration satisfies the project constraints (recall ≥ 95%, precision ≥ 70%). The Streamlit app applies the same threshold so the visuals match the reported metrics.

---

## Results

Static:
<img width="1436" height="490" alt="image" src="https://github.com/user-attachments/assets/adbaeae6-7778-4e10-884c-74a884d1a039" />

webapp demo:
![App demo](assets/IRM_app_demo.gif)

---

## License and usage
- Dataset: BraTS 2019 (CC BY-NC-SA; check the dataset’s original license).
- Code: MIT.
- Clinical: For research/demo only; not approved for clinical use.

Changelog highlights:
- Added Streamlit app with consistent preprocessing and 0.30 probability threshold
- Added visualization that highlights the slice with the largest tumor area
- Documented local quick start and embedded demo GIF (assets/app_demo.gif)
