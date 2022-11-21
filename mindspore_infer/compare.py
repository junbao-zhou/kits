import numpy as np
import os

files = [
    "test_9.nii.gz",
    "test_13.nii.gz",
    ]

output_dir_1 = "output_data"
# output_dir_2 = "ascend_data/output_data"
output_dir_2 = "../output_data"

for file in files:
    a = np.load(
        file=os.path.join(output_dir_1, f"{file}.npy"),
        allow_pickle=True,
    )
    b = np.load(
        file=os.path.join(output_dir_2, f"{file}.npy"),
        allow_pickle=True,
    )

    delta = a-b
    delta_abs = np.abs(delta)
    print(f"{delta_abs.max() = }")
    print(f"{delta_abs.sum() = }")
    print(f"{(delta_abs != 0).sum() = }")