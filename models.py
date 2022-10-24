from torch import nn
import torch
from torchvision import models
import torch.nn.functional as F
from typing import List


class Conv2dBatchNorm(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            padding_mode: str = 'zeros',
    ):
        super(Conv2dBatchNorm, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            padding_mode=padding_mode
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv2dBatchNormActivion(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            padding_mode: str = 'zeros',
            activation: nn.Module = nn.ReLU()):
        super(Conv2dBatchNormActivion, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            padding_mode=padding_mode
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Conv2dBatchNormActivionSequential(nn.Sequential):
    def __init__(
            self,
            channels: List[int],
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode: str = 'zeros',
            activation: nn.Module = nn.ReLU()):
        layers = []
        for idx, channel in enumerate(channels[:-1]):
            layers.append(
                Conv2dBatchNormActivion(
                    channel, channels[idx+1], kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, activation=activation))
        super(Conv2dBatchNormActivionSequential, self).__init__(*layers)


class VGGBlockBatchNorm(nn.Module):
    def __init__(
            self,
            num_convs: int,
            in_channels: int,
            out_channels: int,
    ) -> None:
        super(VGGBlockBatchNorm, self).__init__()

        self.conv2ds = Conv2dBatchNormActivionSequential(
            [in_channels] + [out_channels] * num_convs, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2ds(x)
        x = self.pool(x)
        return x


class VGG16(models.VGG):
    def __init__(
            self,
            num_classes: int = 1000, init_weights: bool = True) -> None:
        features = nn.Sequential(
            VGGBlockBatchNorm(2, in_channels=3, out_channels=64),
            VGGBlockBatchNorm(2, in_channels=64, out_channels=128),
            VGGBlockBatchNorm(3, in_channels=128, out_channels=256),
            VGGBlockBatchNorm(3, in_channels=256, out_channels=512),
            VGGBlockBatchNorm(3, in_channels=512, out_channels=512),
        )
        super(VGG16, self).__init__(features, num_classes, init_weights)


# class FCN8(nn.Module):

#     def __init__(self, num_classes):
#         super(FCN8, self).__init__()

#         vgg16 = VGG16()
#         create_model(vgg16, '', is_load_model=True)

#         self.feature_extractors = nn.ModuleDict({
#             '2': vgg16.features[0],
#             '4': vgg16.features[1],
#             '8': vgg16.features[2],
#             '16': vgg16.features[3],
#             '32': vgg16.features[4],
#         })

#         self.predict = nn.Sequential(
#             Conv2dBatchNormActivion(
#                 in_channels=512, out_channels=1024,
#                 kernel_size=3, padding=1
#             ),
#             Conv2dBatchNormActivion(
#                 in_channels=1024, out_channels=num_classes,
#                 kernel_size=1
#             ),
#         )

#         self.features_skip = nn.ModuleDict({
#             '8': Conv2dBatchNorm(in_channels=256, out_channels=num_classes, kernel_size=1),
#             '16': Conv2dBatchNorm(in_channels=512, out_channels=num_classes, kernel_size=1),
#         })

#         self.upsample = nn.ModuleDict({
#             '8': nn.ConvTranspose2d(
#                 in_channels=num_classes, out_channels=num_classes, kernel_size=3, stride=2, padding=1),
#             '16': nn.ConvTranspose2d(
#                 in_channels=num_classes, out_channels=num_classes, kernel_size=3, stride=2, padding=1),
#         })

#         self.final_upsample = nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_channels=num_classes, out_channels=num_classes, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ConvTranspose2d(
#                 in_channels=num_classes, out_channels=num_classes, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ConvTranspose2d(
#                 in_channels=num_classes, out_channels=num_classes, kernel_size=3, stride=2, padding=1, output_padding=1),
#         )

#     def forward(self, x):
#         input_size = x.size()

#         # Size of input=1,num_classes,256,256
#         f = {}
#         for downsample_rate, layer in self.feature_extractors.items():
#             f[downsample_rate] = layer(x)
#             x = f[downsample_rate]

#         predict = self.predict(x)

#         features_skip_out = {
#             '8': self.features_skip['8'](f['8']),
#             '16': self.features_skip['16'](f['16']),
#         }

#         predict = self.upsample['16'](
#             predict, features_skip_out['16'].size()[2:])
#         predict += features_skip_out['16']
#         predict = self.upsample['8'](
#             predict, features_skip_out['8'].size()[2:])
#         predict += features_skip_out['8']
#         output = self.final_upsample(
#             predict)

#         return output


class SegNet(nn.Module):

    def __init__(self, num_classes):
        super(SegNet, self).__init__()
        self.num_classes = num_classes

        self.encoders = nn.ModuleDict({
            '2': Conv2dBatchNormActivionSequential([3, 64, 64]),
            '4': Conv2dBatchNormActivionSequential([64, 128, 128]),
            '8': Conv2dBatchNormActivionSequential([128, 256, 256, 256]),
            '16': Conv2dBatchNormActivionSequential([256, 512, 512, 512]),
            '32': Conv2dBatchNormActivionSequential([512, 512, 512, 512]),
        })

        self.pool = nn.MaxPool2d(2, stride=2, dilation=1, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, stride=2)

        self.decoders = nn.ModuleDict({
            '32': Conv2dBatchNormActivionSequential([512, 512, 512, 512]),
            '16': Conv2dBatchNormActivionSequential([512, 512, 512, 256]),
            '8': Conv2dBatchNormActivionSequential([256, 256, 256, 128]),
            '4': Conv2dBatchNormActivionSequential([128, 128, 64]),
            '2': Conv2dBatchNormActivionSequential([64, 64, num_classes]),
        })

    def forward(self, x):
        encodes = {}
        pool_indices = {}
        for downsample_rate, layer in self.encoders.items():
            encodes[downsample_rate] = layer(x)
            x, pool_indices[downsample_rate] = self.pool(
                encodes[downsample_rate])

        unpools = {}
        for upsample_rate, layer in self.decoders.items():
            unpools[upsample_rate] = self.unpool(
                x, pool_indices[upsample_rate])
            x = layer(unpools[upsample_rate])
        return x


class U_Net(nn.Module):
    def __init__(self, num_classes) -> None:
        super(U_Net, self).__init__()
        self.num_classes = num_classes

        self.encoders = nn.ModuleDict({
            '2': Conv2dBatchNormActivionSequential([1, 32, 32]),
            '4': nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                Conv2dBatchNormActivionSequential([32, 64, 64])),
            '8': nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                Conv2dBatchNormActivionSequential([64, 128, 128])),
            '16': nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                Conv2dBatchNormActivionSequential([128, 256, 256])),
        })

        self.connector = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dBatchNormActivionSequential([256, 512, 512]),
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256,
                kernel_size=3, stride=2, padding=1, output_padding=1),
        )

        self.decoders = nn.ModuleDict({
            '16': nn.Sequential(
                Conv2dBatchNormActivionSequential([512, 256, 256]),
                nn.ConvTranspose2d(
                    in_channels=256, out_channels=128,
                    kernel_size=3, stride=2, padding=1, output_padding=1),
            ),
            '8': nn.Sequential(
                Conv2dBatchNormActivionSequential([256, 128, 128]),
                nn.ConvTranspose2d(
                    in_channels=128, out_channels=64,
                    kernel_size=3, stride=2, padding=1, output_padding=1),
            ),
            '4': nn.Sequential(
                Conv2dBatchNormActivionSequential([128, 64, 64]),
                nn.ConvTranspose2d(
                    in_channels=64, out_channels=32,
                    kernel_size=3, stride=2, padding=1, output_padding=1),
            ),
            '2': nn.Sequential(
                Conv2dBatchNormActivionSequential([64, 32, 32]),
                nn.Conv2d(in_channels=32,
                          out_channels=self.num_classes, kernel_size=1)
            ),
        })

    def forward(self, x):
        encode_results = {}
        for downsample_rate, layer in self.encoders.items():
            encode_results[downsample_rate] = layer(x)
            x = encode_results[downsample_rate]
        x = self.connector(x)
        for upsample_rate, layer in self.decoders.items():
            concat_map = torch.concat(
                (x, encode_results[upsample_rate]), dim=1)
            x = layer(concat_map)
        return x
