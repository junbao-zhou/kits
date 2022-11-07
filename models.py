from torch import nn
import torch
from torchvision import models
from typing import List, Optional, Union


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
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
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
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
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
                    channel, channels[idx+1],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    padding_mode=padding_mode,
                    activation=activation,
                ))
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
            [in_channels] + [out_channels] * num_convs,
            kernel_size=3,
            stride=1,
            padding=1)
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


# class SegNet(nn.Module):

#     def __init__(self, num_classes):
#         super(SegNet, self).__init__()
#         self.num_classes = num_classes

#         self.encoders = nn.ModuleDict({
#             '2': Conv2dBatchNormActivionSequential([3, 64, 64]),
#             '4': Conv2dBatchNormActivionSequential([64, 128, 128]),
#             '8': Conv2dBatchNormActivionSequential([128, 256, 256, 256]),
#             '16': Conv2dBatchNormActivionSequential([256, 512, 512, 512]),
#             '32': Conv2dBatchNormActivionSequential([512, 512, 512, 512]),
#         })

#         self.pool = nn.MaxPool2d(2, stride=2, dilation=1, return_indices=True)
#         self.unpool = nn.MaxUnpool2d(2, stride=2)

#         self.decoders = nn.ModuleDict({
#             '32': Conv2dBatchNormActivionSequential([512, 512, 512, 512]),
#             '16': Conv2dBatchNormActivionSequential([512, 512, 512, 256]),
#             '8': Conv2dBatchNormActivionSequential([256, 256, 256, 128]),
#             '4': Conv2dBatchNormActivionSequential([128, 128, 64]),
#             '2': Conv2dBatchNormActivionSequential([64, 64, num_classes]),
#         })

#     def forward(self, x):
#         encodes = {}
#         pool_indices = {}
#         for downsample_rate, layer in self.encoders.items():
#             encodes[downsample_rate] = layer(x)
#             x, pool_indices[downsample_rate] = self.pool(
#                 encodes[downsample_rate])

#         unpools = {}
#         for upsample_rate, layer in self.decoders.items():
#             unpools[upsample_rate] = self.unpool(
#                 x, pool_indices[upsample_rate])
#             x = layer(unpools[upsample_rate])
#         return x


class U_Net(nn.Module):
    def __init__(self, in_channels, num_classes) -> None:
        super(U_Net, self).__init__()
        self.num_classes = num_classes

        self.encoders = nn.ModuleDict({
            '2': Conv2dBatchNormActivionSequential([in_channels, 32, 32]),
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
                kernel_size=2, stride=2),
        )

        self.decoders = nn.ModuleDict({
            '16': nn.Sequential(
                Conv2dBatchNormActivionSequential([512, 256, 256]),
                nn.ConvTranspose2d(
                    in_channels=256, out_channels=128,
                    kernel_size=2, stride=2),
            ),
            '8': nn.Sequential(
                Conv2dBatchNormActivionSequential([256, 128, 128]),
                nn.ConvTranspose2d(
                    in_channels=128, out_channels=64,
                    kernel_size=2, stride=2),
            ),
            '4': nn.Sequential(
                Conv2dBatchNormActivionSequential([128, 64, 64]),
                nn.ConvTranspose2d(
                    in_channels=64, out_channels=32,
                    kernel_size=2, stride=2),
            ),
            '2': nn.Sequential(
                Conv2dBatchNormActivionSequential([64, 32, 32]),
                nn.Conv2d(
                    in_channels=32, out_channels=self.num_classes, kernel_size=1)
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


class ResBlockDownsaple(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super(ResBlockDownsaple, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='zeros',
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=0.01)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='zeros',
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(negative_slope=0.01)
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode='zeros',
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(x)
        out += identity
        out = self.act2(out)
        return out


class UpsampleResBlock(nn.Module):
    def __init__(
        self,
        # in_channels: int,
        out_channels: int,
    ) -> None:
        super(UpsampleResBlock, self).__init__()
        self.upsample = nn.PixelShuffle(2)
        self.conv1 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='zeros',
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=0.01)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='zeros',
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x, y):
        out = self.upsample(x)
        identity = torch.cat((out, y), dim=1)
        out = self.conv1(identity)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.act2(out)
        return out


def conv1x1(in_channels: int, out_channels: int):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
    )


class Res_U_Net(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super(Res_U_Net, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode='zeros',
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.encoders = nn.ModuleList([
            ResBlockDownsaple(in_channels=32, out_channels=64),
            ResBlockDownsaple(in_channels=64, out_channels=128),
            ResBlockDownsaple(in_channels=128, out_channels=256),
            ResBlockDownsaple(in_channels=256, out_channels=512),
        ])
        self.connector = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode='zeros',
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.decoders = nn.ModuleList([
            UpsampleResBlock(out_channels=64),
            UpsampleResBlock(out_channels=128),
            UpsampleResBlock(out_channels=256),
            UpsampleResBlock(out_channels=512),
        ])
        self.shortcuts = nn.ModuleList([
            conv1x1(in_channels=64, out_channels=32),
            conv1x1(in_channels=128, out_channels=64),
            conv1x1(in_channels=256, out_channels=128),
            conv1x1(in_channels=512, out_channels=256),
        ])
        self.upsample = nn.PixelShuffle(2)
        self.conv_last = conv1x1(in_channels=32+16, out_channels=out_channels)

    def forward(self, x):
        ident = self.conv0(x)
        out = ident
        encode_results = []
        for layer in self.encoders:
            out = layer(out)
            encode_results.append(out)
        out = self.connector(out)
        for i in range(len(self.decoders)-1, -1, -1):
            short = self.shortcuts[i](
                encode_results[i])
            out = self.decoders[i](
                out,
                short)
        out = self.upsample(out)
        out = self.conv_last(
            torch.cat(
                (out, ident), dim=1)
        )
        return out


if __name__ == "__main__":
    u_net = Res_U_Net(3, 3)
    # import torchvision
    # torchvision.models.resnet.resnet18
    print(u_net.state_dict().keys())
    input_t = torch.rand((8, 3, 512, 512))
    output_t = u_net(input_t)
    print(output_t.shape)
