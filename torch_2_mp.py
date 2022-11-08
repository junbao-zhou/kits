import os
from typing import Callable, Tuple, Type
import torch
import torch.nn as nn
import numpy as np
import re
import mindspore.nn
import mindspore.ops
import mindspore
import config

from utils import load_model

OUTPUT_DIR = 'mindspore_infer'


def code_torch_2_mp(torch_codes: str):
    mp_codes = torch_codes.\
        replace('import torch', 'import mindspore').\
        replace('from torch import nn', 'from mindspore import nn').\
        replace('nn.Module', 'nn.Cell').\
        replace('padding_mode', 'pad_mode').\
        replace("pad_mode: str = 'zeros'", "pad_mode: str = 'pad'").\
        replace("pad_mode='zeros'", "pad_mode='pad'").\
        replace('def forward(', 'def construct(').\
        replace('torch.Tensor', 'mindspore.Tensor').\
        replace('nn.Sequential', 'nn.SequentialCell').\
        replace('nn.CellDict', 'CellDict').\
        replace('nn.ConvTranspose2d', 'nn.Conv2dTranspose').\
        replace('nn.Conv2dTranspose(', 'nn.Conv2dTranspose(has_bias=True,').\
        replace('nn.Conv2d(', 'nn.Conv2d(has_bias=True,').\
        replace('.state_dict(', '.parameters_dict(')
    mp_codes = re.sub(
        r"torch\.concat\(([\s\S]*?), dim=([0-9]+)\)", r"ops.Concat(axis=\2)(\1)", mp_codes, re.S)
    mp_codes = re.sub(r"torch\.cat\(([\s\S]*?), dim=([0-9]+)\)",
                      r"ops.Concat(axis=\2)(\1)", mp_codes, re.S)
    mp_codes = re.sub(
        r"nn\.PixelShuffle\(([0-9]+)\)", r"PixelShuffle(\1)", mp_codes, re.S)
    mp_codes = re.sub(
        r"nn\.LeakyReLU\(negative_slope=([0-9\.]+)\)", r"nn.LeakyReLU(alpha=\1)", mp_codes, re.S)
    mp_codes = mp_codes.\
        replace('torch.rand', 'ops.UniformReal()')
    mp_codes = """
from models_mp import CellDict
from models_mp import PixelShuffle
from mindspore import ops
""" + mp_codes
    return mp_codes


def code_torch_2_mp_main():
    from pathlib import Path
    torch_codes = Path('models.py').read_text()
    mp_codes = code_torch_2_mp(torch_codes)
    Path(os.path.join(OUTPUT_DIR, 'models.py')).write_text(mp_codes)


def torch_2_np(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


def np_2_mindspore(np_tensor: np.ndarray):
    return mindspore.Tensor(np_tensor)


def mindspore_2_np(mindspore_tensor: mindspore.Tensor):
    return mindspore_tensor.asnumpy()


def torch_2_mindspore(tensor: torch.Tensor):
    return np_2_mindspore(torch_2_np(tensor))


def compare_torch_minspore_model(torch_model: torch.nn.Module, mindspore_model: mindspore.nn.Cell, input_shape: Tuple[int, ...]):
    input_t = torch.rand(input_shape)
    # input_t = torch.randint(0, 3, input_shape).float()
    output_t = torch_model(input_t)

    input_mp = torch_2_mindspore(input_t)
    mindspore_model.set_train(False)
    output_mp = mindspore_model(input_mp)

    delta = torch_2_mindspore(output_t) - output_mp
    delta_abs = delta.abs()
    print(f"{delta_abs.sum() = }")
    print(f"{delta_abs.max() = }")


def torch_model_2_mindspore(torch_model: torch.nn.Module, mindspore_module_class: Callable):
    torch_state_dict = torch_model.state_dict()
    keys = list(torch_state_dict.keys())
    # print(f"{keys = }")
    # print(f"{len(keys) = }")
    for name, child_module in torch_model.named_modules():
        if type(child_module) == torch.nn.BatchNorm2d:
            torch_state_dict[f"{name}.moving_mean"] = torch_state_dict.pop(
                f"{name}.running_mean")
            torch_state_dict[f"{name}.moving_variance"] = torch_state_dict.pop(
                f"{name}.running_var")
            torch_state_dict[f"{name}.gamma"] = torch_state_dict.pop(
                f"{name}.weight")
            torch_state_dict[f"{name}.beta"] = torch_state_dict.pop(
                f"{name}.bias")
    mp_model = mindspore_module_class()
    mp_model.set_train(False)
    # print(f"{mp_model.parameters_dict().keys() = }")
    # print(f"{len(mp_model.parameters_dict().keys()) = }")
    new_dict = {
        key: mindspore.Parameter(torch_2_mindspore(state.data))
        for key, state in torch_state_dict.items()
    }
    mindspore.load_param_into_net(mp_model, new_dict)
    return mp_model


def torch_model_2_mindspore_main():
    import models
    with torch.no_grad():
        torch_model = models.Res_U_Net(3, 3)
        torch_model.eval()
        load_model(
            model=torch_model,
            load_dir=config.BASE_MODEL_PATH,
            suffice="valid_best",
            map_location=torch.device('cpu')
        )
        import mindspore_infer.models
        mp_model = torch_model_2_mindspore(
            torch_model, lambda: mindspore_infer.models.Res_U_Net(3, 3))
        compare_torch_minspore_model(torch_model, mp_model, (8, 3, 512, 512))
        mindspore.save_checkpoint(mp_model, os.path.join(
            OUTPUT_DIR, f"{type(mp_model).__name__}.ckpt"))


if __name__ == '__main__':
    torch_model_2_mindspore_main()
    # code_torch_2_mp_main()
    # up_torch = torch.nn.PixelShuffle(2)
    # up_mindspore = mindspore.ops.DepthToSpace(2)
    # compare_torch_minspore_model(up_torch, up_mindspore, (1, 8, 5, 5))
