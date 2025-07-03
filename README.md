# U-NET Brain MRI Tumor Segmentation (BraTS 2019)

End-to-end segmentation of brain tumors on FLAIR MRI slices using a 2-D U-NET trained on the BraTS 2019 dataset.  
The project is designed for quick deployment on AWS SageMaker and includes a demonstration notebook.

---

## Repository structure
├── UNET_IRM_for_SageMaker.ipynb   ← main notebook \
├── checkpoints/                   ← place the pretrained model here \
├── requirements.txt               ← dependancies to install   \
└── README.md \

---

## Downloading the pretrained model

1. Download **`unet_best.h5`** from Google Drive  \
   https://drive.google.com/file/d/1HmYfTN_u3WyNwxJLp2943ULw1m2a0Ws4/view

2. Create the checkpoints folder if it does not yet exist:

```bash
mkdir -p checkpoints
mv unet_best.h5 checkpoints/unet_best.h5
```

## Notebook to run

UNET_IRM_for_SageMaker.ipynb
- downloads (or mounts) BraTS 2019
- preprocesses the volumes into 2-D slices (240 × 240)
- loads checkpoints/unet_best.h5
- evaluates recall and precision on the test set
- displays predictions with the binary display function that respects the probability threshold

### Probability threshold 0.30
**Initial issue** : High precision (> 0.93) but recall only about 0.90 \
Applied fix : For each pixel, the tumor probability is the sum of class 1 + 2 + 3 probabilities. A pixel is labeled tumor if probability > 0.30. \

**Before** : Recall ≈ 0.90 / Precision ≈ 0.93 \
After threshold 0.30 : Recall ≈ 0.96 / Precision ≈ 0.88 (still above the 0.70 requirement) \
**This satisfies the project constraints: recall ≥ 95 % and precision ≥ 70 %.**

Results:\
![image](https://github.com/user-attachments/assets/962a98b6-7a19-4950-b104-930ad3f9a1bb)

Or binarised:\
![image](https://github.com/user-attachments/assets/a617dc86-d9c6-4fc1-92a8-d87c05db887f)

License and usage
- BraTS 2019: CC BY-NC-SA
- Code: MIT
- This model is for research and demonstration only; it is not approved for clinical use.
