from typing import Dict, List
from mindspore import nn
import mindspore.numpy as np
import mindspore
from mindspore.nn.layer.container import _get_prefix_and_index


class CellDict(nn.Cell):
    def __init__(
            self, cell_dict: Dict[str, nn.Cell], auto_prefix=True, flags=None):
        super(CellDict, self).__init__(auto_prefix, flags)
        for key, cell in cell_dict.items():
            if self._auto_prefix:
                prefix, _ = _get_prefix_and_index(self._cells)
                cell.update_parameters_name(prefix + key + ".")
            self._cells[key] = cell


class PixelShuffle(nn.Cell):
    def __init__(self, scale_factor: int, auto_prefix=True, flags=None):
        super(PixelShuffle, self).__init__(auto_prefix, flags)
        self.scale_factor = scale_factor

    def construct(self, x: mindspore.Tensor):
        """
        Implementation of pixel shuffle using numpy

        Parameters:
        -----------
        tensor: input tensor, shape is [N, C, H, W]
        scale_factor: scale factor to up-sample tensor

        Returns:
        --------
        tensor: tensor after pixel shuffle, shape is [N, C/(s*s), s*H, s*W],
            where s refers to scale factor
        """
        num, ch, height, width = x.shape
        if ch % (self.scale_factor * self.scale_factor) != 0:
            raise ValueError('channel of tensor must be divisible by '
                             '(scale_factor * scale_factor).')

        new_ch = ch // (self.scale_factor * self.scale_factor)
        new_height = height * self.scale_factor
        new_width = width * self.scale_factor

        tensor = x.reshape(
            [num, new_ch, self.scale_factor, self.scale_factor, height, width])
        # new axis: [num, new_ch, height, scale_factor, width, scale_factor]
        tensor = tensor.transpose([0, 1, 4, 2, 5, 3])
        tensor = tensor.reshape([num, new_ch, new_height, new_width])
        return tensor


class Conv2dBNAct(nn.Cell):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            actfunc=nn.ReLU()):
        super(Conv2dBNAct, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding)
        self.batchnorm2d = nn.BatchNorm2d(out_channels)
        self.actfunc = actfunc

    def construct(self, x):
        x = self.conv2d(x)
        x = self.batchnorm2d(x)
        x = self.actfunc(x)
        return x


class Conv2dBNActSeq(nn.SequentialCell):
    def __init__(
            self,
            channels: List[int],
            kernel_size=3,
            stride=1,
            padding=1,
            actfunc=nn.ReLU()):
        layers = []
        for idx, channel in enumerate(channels[:-1]):
            layers.append(
                Conv2dBNAct(
                    channel, channels[idx+1], kernel_size=kernel_size, stride=stride, padding=padding, actfunc=actfunc)
            )
        super(Conv2dBNActSeq, self).__init__(*layers)


class U_Net(nn.Cell):
    def __init__(self, num_classes):
        super(U_Net, self).__init__()
        self.num_classes = num_classes

        # Encoder
        self.convseq1 = Conv2dBNActSeq([1, 32, 32])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convseq2 = Conv2dBNActSeq([32, 64, 64])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convseq3 = Conv2dBNActSeq([64, 128, 128])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convseq4 = Conv2dBNActSeq([128, 256, 256])

        # Conector
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convseq5 = Conv2dBNActSeq([256, 512, 512])
        self.upsample1 = nn.Conv2dTranspose(
            in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Decoder
        self.convseq6 = Conv2dBNActSeq([512, 256, 256])
        self.upsample2 = nn.Conv2dTranspose(
            in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convseq7 = Conv2dBNActSeq([256, 128, 128])
        self.upsample3 = nn.Conv2dTranspose(
            in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convseq8 = Conv2dBNActSeq([128, 64, 64])
        self.upsample4 = nn.Conv2dTranspose(
            in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convseq9 = Conv2dBNActSeq([64, 32, 32])
        self.final = nn.Conv2d(
            in_channels=32, out_channels=self.num_classes, kernel_size=1)

    def construct(self, x):
        downfeat1 = self.convseq1(x)
        x = self.maxpool1(downfeat1)
        downfeat2 = self.convseq2(x)
        x = self.maxpool2(downfeat2)
        downfeat3 = self.convseq3(x)
        x = self.maxpool3(downfeat3)
        downfeat4 = self.convseq4(x)

        x = self.maxpool4(downfeat4)
        downfeat5 = self.convseq5(x)
        x = self.upsample1(downfeat5)

        x = np.concatenate((downfeat4, x), axis=1)
        x = self.convseq6(x)
        x = self.upsample2(x)
        x = np.concatenate((downfeat3, x), axis=1)
        x = self.convseq7(x)
        x = self.upsample3(x)
        x = np.concatenate((downfeat2, x), axis=1)
        x = self.convseq8(x)
        x = self.upsample4(x)
        x = np.concatenate((downfeat1, x), axis=1)
        x = self.convseq9(x)
        x = self.final(x)

        return x


if __name__ == "__main__":
    u_net = U_Net(3)
    print(u_net.parameters_dict().keys())
