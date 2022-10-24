import random
from typing import Callable
import kits19.starter_code.utils as data_utils
import nibabel as nib
import numpy as np
import PIL.Image as Im
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import os
import tqdm
import torch


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

ALL_CASES = list(np.concatenate((np.arange(160), np.arange(161, 300))))
TRAINING_CASES = list(np.concatenate((np.arange(160), np.arange(161, 210))))


def cases_to_numpy():
    def convert_cases(cases_list: list, load_func, save_name: str):
        for case in tqdm.tqdm(cases_list):
            nif = load_func(case)
            np_array = nif_img_to_numpy(nif)
            np.save(
                os.path.join(data_utils.get_case_path(case), save_name),
                np_array.astype(np.int16),
                allow_pickle=True,
            )
    print("converting imagings")
    convert_cases(ALL_CASES, data_utils.load_volume, IMAGING_NP_NAME)
    print("converting segmentations")
    convert_cases(
        TRAINING_CASES, data_utils.load_segmentation, SEGMENTATION_NP_NAME)


def load_case_np(case_id: int, file_name: str) -> np.ndarray:
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
        print(f"{self.imagings.shape = }")
        print(f"{self.imagings.dtype = }")
        print(f"loading labels")
        self.labels = load_and_concat(load_segmentation_np)
        print(f"{self.labels.shape = }")
        print(f"{self.labels.dtype = }")

    def __len__(self):
        return self.imagings.shape[0]

    def get_np(self, index):
        return self.imagings[index], self.labels[index]

    def get_transformed_np(self, index):
        img_np, label_np = self.get_np(index)
        img_np = normalize_np(img_np.astype(np.float32), CLIP_MIN, CLIP_MAX)
        return img_np, label_np

    def save_np(self, index):
        img_np, label_np = self.get_transformed_np(index)
        img = Im.fromarray((255.0 * img_np).astype(np.uint8))
        img.save("imaging.png")
        label = Im.fromarray((127.0 * label_np).astype(np.uint8))
        label.save("label.png")

    def __getitem__(self, index):
        img_np, label_np = self.get_transformed_np(index)
        img = torch.from_numpy(img_np)
        label = torch.from_numpy(label_np).long()
        img.unsqueeze_(0)
        # label.unsqueeze_(0)

        return img, label


def split_cases(cases: list, split_num):
    random_cases = random.sample(cases, len(cases))
    list1 = random_cases[:split_num]
    list1.sort()
    list2 = random_cases[split_num:]
    list2.sort()
    return list1, list2


def data_loader(
        train_cases: list, valid_cases: list, batch_size: int, is_normalize: bool = True, is_augment: bool = False):
    train_data = SegmentKits19(
        train_cases
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
    )

    val_data = SegmentKits19(
        valid_cases
    )
    valid_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
    )

    return train_loader, valid_loader


if __name__ == '__main__':
    cases_to_numpy()

    # dataset = SegmentKits19(TRAINING_CASES)
    # print(f"{dataset.__len__()}")
    # dataset.save_np(313)
