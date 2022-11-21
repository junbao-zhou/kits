import numpy as np
import nibabel as nib


def print_nif_img(img: nib.nifti1.Nifti1Image):
    print(f"shape = {img.shape}")
    print(f"dtype = {img.get_data_dtype()}")
    print(f"header = {img.header}")


def nif_img_to_numpy(img: nib.nifti1.Nifti1Image, dtype=np.int32):
    return img.get_fdata(dtype=np.float32).astype(dtype)


def numpy_to_nif_img(np_array: np.ndarray):
    return nib.nifti1.Nifti1Image(np_array, affine=np.eye(4))