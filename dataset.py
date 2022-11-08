import random
from typing import Any, Callable, Dict, List
import kits19.starter_code.utils as data_utils
import numpy as np
import PIL.Image as Im
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as transformF
import os
import tqdm
import torch
import config
from nif_img import nif_img_to_numpy


CLIP_MAX = 512
CLIP_MIN = -256


def normalize_np(array: np.ndarray, hu_min, hu_max):
    cliped = array.clip(hu_min, hu_max)
    cliped -= cliped.min()
    cliped *= 1 / cliped.max()
    return cliped


IMAGING_NP_NAME = "imaging.npy"
SEGMENTATION_NP_NAME = "segmentation.npy"


def cases_to_numpy():
    def convert_cases(cases_list: list, load_func, save_name: str, dtype):
        for case in tqdm.tqdm(cases_list):
            nif = load_func(case)
            np_array = nif_img_to_numpy(nif, dtype)
            np.save(
                os.path.join(data_utils.get_case_path(case), save_name),
                np_array,
                allow_pickle=True,
            )
    print("converting imagings")
    convert_cases(config.ALL_CASES, data_utils.load_volume,
                  IMAGING_NP_NAME, np.int16)
    print("converting segmentations")
    convert_cases(
        config.TRAINING_CASES, data_utils.load_segmentation, SEGMENTATION_NP_NAME, np.int8)


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


def dict_of_np(d: Dict[Any, np.ndarray]):
    return {
        key: (array.shape, array.dtype)
        for key, array in d.items()
    }


class RandomRotationMasked(transforms.RandomRotation):
    def __init__(
        self, degrees, interpolation=transforms.InterpolationMode.NEAREST, expand=False, center=None, fill=0, resample=None
    ):
        super(RandomRotationMasked, self).__init__(
            degrees, interpolation, expand, center, fill, resample)

    def forward(self, img):
        angle = self.get_params(self.degrees)
        img_out = transformF.rotate(
            img[:-1], angle, self.resample, self.expand, self.center)
        mask_out = transformF.rotate(
            img[-1:], angle, transforms.InterpolationMode.NEAREST, self.expand, self.center)
        return torch.concat((img_out, mask_out))


class RandomResizedCropMasked(transforms.RandomResizedCrop):
    def __init__(
        self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=transforms.InterpolationMode.BILINEAR,
    ):
        print(
            f"{type(self).__name__} : {size = }  {scale = }  {ratio = }  {interpolation = }")
        super(RandomResizedCropMasked, self).__init__(
            size, scale, ratio, interpolation)

    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img_out = transformF.resized_crop(
            img[:-1], i, j, h, w, self.size, self.interpolation)
        mask_out = transformF.resized_crop(
            img[-1:], i, j, h, w, self.size, transforms.InterpolationMode.NEAREST)
        return torch.concat((img_out, mask_out))


class augmentation:
    image_size = 512
    image_area = image_size ** 2
    pad_size = 12
    padded_image_area = (image_size + pad_size) ** 2
    transforms = [
        RandomRotationMasked(
            degrees=9, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(12),
        RandomResizedCropMasked(
            size=image_size,
            scale=(
                (image_size - 10) ** 2 / padded_image_area,
                (image_size + 10) ** 2 / padded_image_area,
            ),
            ratio=(96. / 100., 100. / 96.),
        ),
    ]


class SegmentKits19(datasets.VisionDataset):
    def __init__(
        self,
        cases: List[int],
        is_augment: bool = False,
    ) -> None:
        self.cases = cases
        self.is_augment = is_augment
        print(f"{type(self).__name__} {self.is_augment = }")

        def load(load_func: Callable):
            load_dict = {}
            cases_index = []
            image_index = []
            for case in tqdm.tqdm(self.cases):
                loaded = load_func(case)
                cases_index += [case] * loaded.shape[0]
                image_index += list(range(loaded.shape[0]))
                load_dict[case] = loaded
                # load_list.append(load_func(case))
            return load_dict, cases_index, image_index
        print(f"{type(self).__name__} loading cases {self.cases} from file")
        print(f"loading imagings")
        self.imagings, _, _ = load(load_imaging_np)
        print(
            f"{type(self).__name__}.imagings {dict_of_np(self.imagings)}")
        print(f"loading labels")
        self.labels, self.cases_index, self.image_index = load(
            load_segmentation_np)
        print(f"{type(self).__name__}.labels {dict_of_np(self.labels)}")
        # print(f"{self.cases_index = }")
        # print(f"{self.image_index = }")

    def __len__(self):
        return len(self.cases_index)

    def get_np(self, index):
        case = self.cases_index[index]
        img_id = self.image_index[index]
        return self.imagings[case][img_id], self.labels[case][img_id]

    def get_np_3(self, index):
        case = self.cases_index[index]
        img_1_id = self.image_index[index]
        img_0_id = img_1_id if img_1_id == 0 else (img_1_id-1)
        img_2_id = img_1_id if img_1_id == (
            self.imagings[case].shape[0]-1) else (img_1_id+1)
        imgs_3 = np.stack(
            (self.imagings[case][img_0_id], self.imagings[case]
             [img_1_id], self.imagings[case][img_2_id]),
        )
        labels = self.labels[case][img_1_id]
        return imgs_3, labels

    def get_transformed_np(self, index):
        img_np, label_np = self.get_np_3(index)
        img_np = normalize_np(img_np.astype(np.float32), CLIP_MIN, CLIP_MAX)
        return img_np, label_np

    def save_np(self, index):
        img_np, label_np = self.get_transformed_np(index)
        if len(img_np.shape) == 3:
            img_np = np.moveaxis(img_np, 0, 2)
        img = Im.fromarray((255.0 * img_np).astype(np.uint8))
        img.save("imaging.png")
        label = Im.fromarray((127.0 * label_np).astype(np.uint8))
        label.save("label.png")

    def __getitem__(self, index):
        img_np, label_np = self.get_transformed_np(index)
        img = torch.from_numpy(img_np)
        label = torch.from_numpy(label_np).float()
        if len(img.shape) == 2:
            img.unsqueeze_(0)
        if len(label.shape) == 2:
            label.unsqueeze_(0)

        if self.is_augment:
            image_mask = torch.concat((img, label))
            for transform in augmentation.transforms:
                if isinstance(transform, torch.nn.Module):
                    image_mask = transform(image_mask)
                else:
                    raise TypeError("transform must be transforms")
            img = image_mask[:-1]
            label = image_mask[-1:]

        return img, label.squeeze().long()

    def save_torch(self, index):
        img_torch, label_torch = self.__getitem__(index)
        img_np = img_torch.numpy()
        label_np = label_torch.numpy()
        if len(img_np.shape) == 3:
            img_np = np.moveaxis(img_np, 0, 2)
        img = Im.fromarray((255.0 * img_np).astype(np.uint8))
        img.save("imaging.png")
        label = Im.fromarray((127.0 * label_np).astype(np.uint8))
        label.save("label.png")

    def get_mindpore(self, index):
        import mindspore
        img_np, label_np = self.get_transformed_np(index)
        img = mindspore.Tensor(img_np)
        label = mindspore.Tensor(label_np).long()

        return img, label


def split_cases(cases: list, split_num):
    random_cases = random.sample(cases, len(cases))
    list1 = random_cases[:split_num]
    list1.sort()
    list2 = random_cases[split_num:]
    list2.sort()
    return list1, list2


def data_loader(
        train_cases: list, valid_cases: list, train_batch_size: int, valid_batch_size: int, is_normalize: bool = True, is_augment: bool = False):
    print(f"""Data loader:
{train_cases = }
{valid_cases = }
""")

    train_data = SegmentKits19(
        train_cases,
        is_augment=is_augment,
    )
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=6,
    )

    val_data = SegmentKits19(
        valid_cases
    )
    valid_loader = DataLoader(
        dataset=val_data,
        batch_size=valid_batch_size,
        num_workers=6,
    )

    return train_loader, valid_loader


if __name__ == '__main__':
    # cases_to_numpy()

    dataset = SegmentKits19(config.TRAINING_CASES[40:45], is_augment=True)
    print(f"{dataset.__len__()}")
    # dataset.save_np(100)
    dataset.save_torch(67)
