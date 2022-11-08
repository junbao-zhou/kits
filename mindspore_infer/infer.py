from glob import glob
from dataset import CLIP_MAX, CLIP_MIN, normalize_np, DataLoader
from model_run import test
import argparse
import nibabel as nib
import numpy as np
from tqdm import tqdm
from models import Res_U_Net
import mindspore
from nif_img import nif_img_to_numpy, numpy_to_nif_img
import config
import os
from mindspore import context

BATCH_SIZE = 50
# os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_S
context.set_context(device_target="Ascend")

parser = argparse.ArgumentParser()
parser.add_argument(
    '--inputpath', '-i',
    type=str,
    required=True,
    help='Directory storing input data'
)
parser.add_argument(
    '--outputpath', '-o',
    type=str,
    required=True,
    help='Directory storing output data'
)
args, unparsed = parser.parse_known_args()
input_path = args.inputpath
output_path = args.outputpath


def nii_to_numpy(nii_path: str):
    nii_file_list = glob(f"{nii_path}/*.nii.gz")
    print(f"converting nii files {nii_file_list}")
    for nii_file in tqdm(nii_file_list):
        img = nib.load(nii_file)
        np_array = nif_img_to_numpy(img, np.int16)
        np.save(
            f"{nii_file}.npy",
            np_array,
            allow_pickle=True,
        )


class SegmentKits19:
    def __init__(
        self,
        np_path: str,
        np_file_name: str,
    ) -> None:
        self.np_path = np_path
        self.np_file_name = np_file_name
        self.imgs_np = np.load(
            os.path.join(self.np_path, self.np_file_name),
            allow_pickle=True,
        )
        print(f"{np_path} : self.imgs_np.shape {self.imgs_np.shape}")

    def __len__(self):
        return self.imgs_np.shape[0]

    def get_np(self, index):
        return self.imgs_np[index]

    def get_np_3(self, index):
        img_1_id = index
        img_0_id = img_1_id if img_1_id == 0 else (img_1_id-1)
        img_2_id = img_1_id if img_1_id == (
            self.__len__()-1) else (img_1_id+1)
        imgs_3 = np.stack(
            (self.get_np(img_0_id), self.get_np(img_1_id), self.get_np(img_2_id)),
        )
        return imgs_3

    def get_transformed_np(self, index):
        img_np = self.get_np_3(index)
        img_np = normalize_np(img_np.astype(np.float32), CLIP_MIN, CLIP_MAX)
        return img_np

    def __getitem__(self, index):
        img_np = self.get_transformed_np(index)

        return img_np


nii_to_numpy(input_path)

np_file_list = [
    os.path.basename(np_file)
    for np_file in glob(f"{input_path}/*.npy")]
data_loaders = [
    DataLoader(
        SegmentKits19(
            np_path=input_path,
            np_file_name=np_file,
        ),
        batch_size=BATCH_SIZE,
        has_label=False,
    )
    for np_file in np_file_list
]

N_CLASSES = 3
model = Res_U_Net(3, N_CLASSES)
print(f"loading model from {type(model).__name__}.ckpt")
mindspore.load_checkpoint(
    f"{type(model).__name__}.ckpt",
    net=model,
    strict_load=True,
)
test(data_loaders, model, output_path)


def numpy_to_nii(np_path: str):
    np_file_list = glob(f"{np_path}/*.npy")
    print(f"converting np files {np_file_list}")
    for np_file in tqdm(np_file_list):
        np_array = np.load(
            np_file,
            allow_pickle=True,
        )
        nif_img = numpy_to_nif_img(np_array)
        nib.save(
            nif_img,
            os.path.join(
                np_path,
                os.path.basename(np_file).replace(".npy", ""))
        )


numpy_to_nii(output_path)
