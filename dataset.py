from typing import Callable
import kits19.starter_code.utils as data_utils
import nibabel as nib
import numpy as np
import PIL.Image as Im
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import os
import tqdm

def print_nif_img(img: nib.nifti1.Nifti1Image):
    print(f"shape = {img.shape}")
    print(f"dtype = {img.get_data_dtype()}")
    print(f"header = {img.header}")


def nif_img_to_numpy(img: nib.nifti1.Nifti1Image):
    return img.get_fdata()

CLIP_MAX = 512
CLIP_MIN = -256

def normalize_np(array: np.ndarray, hu_min, hu_max):
    cliped = array.clip(hu_min, hu_max)
    cliped -= cliped.min()
    cliped *= 1 / cliped.max()
    return cliped

IMAGING_NP_NAME = "imaging.npy"
SEGMENTATION_NP_NAME = "segmentation.npy"

ALL_CASES = np.arange(300)
TRAINING_CASES = np.arange(210)

def cases_to_numpy():
    def convert_cases(cases_list, save_name):
        for case in tqdm.tqdm(cases_list):
            nif = data_utils.load_volume(case)
            np_array = nif_img_to_numpy(nif)
            np.save(
                os.path.join(data_utils.get_case_path(case), save_name),
                np_array,
                allow_pickle=True,
            )
    print("converting imagings")
    convert_cases(ALL_CASES, IMAGING_NP_NAME)
    print("converting segmentations")
    convert_cases(TRAINING_CASES, SEGMENTATION_NP_NAME)

def load_case_np(case_id, file_name) -> np.ndarray:
    path = os.path.join(data_utils.get_case_path(case_id), file_name)
    return np.load(
        path,
        allow_pickle=True,
    )

def load_imaging_np(case_id):
    return load_case_np(case_id, IMAGING_NP_NAME)

def load_segmentation_np(case_id):
    return load_case_np(case_id, SEGMENTATION_NP_NAME)

class SegmentKits19(datasets.VisionDataset):
    def __init__(
        self,
        cases,
    ) -> None:
        self.cases = cases
        def load_and_concat(load_func: Callable):
            load_list = []
            for case in tqdm.tqdm(self.cases):
                load_list.append(load_func(case))
            return np.concatenate(load_list)
        print(f"loading cases {self.cases} from file")
        print(f"loading imagings")
        self.imagings = load_and_concat(load_imaging_np)
        print(f"loading labels")
        self.labels = load_and_concat(load_segmentation_np)
        print(f"{self.imagings.shape = }")
        print(f"{self.labels.shape = }")

    def __len__(self):
        return self.imagings.shape[0]

    def __getitem__(self, index):
        image, mask = self.dataset[index]
        # print(f"{mask.mode = }")

        image.save(f'{current_file_path}/origin_image.png')
        RGB_mask = mask.convert("RGB")
        RGB_mask.save(f"{current_file_path}/RGB_mask.png")
        print(f"RGB mask have colors : {set_of_RGB_PIL(RGB_mask)}")

        mask = mask.convert("P")
        # mask.save(f"{current_file_path}/P_mask.png")
        # print(f"mask have labels : {util.set_of_GRAY_PIL(mask)}")

        for transform in self.transforms:
            if isinstance(transform, dict):
                image = transform['image'](image)
                mask = transform['mask'](mask)
            elif isinstance(transform, torch.nn.Module):
                image_mask = torch.concat((image, mask))
                image_mask = transform(image_mask)
                image = image_mask[:3]
                mask = image_mask[3:]
            else:
                raise TypeError("transform must bee dict or transforms")

        return image, mask.squeeze().long()


if __name__ == '__main__':
    # volume, segmentation = data_utils.load_case(1)
    # print_nif_img(volume)
    # volume_np = nif_img_to_numpy(volume)
    # print(f"{volume_np.shape = }")
    # print(f"{volume_np.min() = }")
    # print(f"{volume_np.max() = }")
    # volume1 = volume_np[216]
    # normalized_volume = (255.0 * normalize_np(volume1, CLIP_MIN, CLIP_MAX)).astype(np.uint8)
    # img = Im.fromarray(normalized_volume)
    # img.save("test.png")
    # print_nif_img(segmentation)


    # cases_to_numpy()

    dataset = SegmentKits19(np.arange(6))
    print(f"{dataset.__len__()}")
