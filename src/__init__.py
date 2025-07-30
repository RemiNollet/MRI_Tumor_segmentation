# Key modules

# io_utils.py: load_nifti(path) -> np.ndarray

# preprocessing.py : resize_volume(vol), crop_z(vol, start=3, end=147), volume_to_slices(vol) -> List[np.ndarray]

# inference.py : load_model(), predict_slice(img, thresh=0.30) -> mask, find_max_tumor_slice(masks) -> idx

# visualization.py : overlay(img, mask, alpha=0.4) -> np.ndarray