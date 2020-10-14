from typing import Sequence

from torch import Tensor, nn


def affine(x: Tensor, affine_scale: Tensor, affine_shift: Tensor):
    return x * affine_scale + affine_shift


class UpsamplingFirstMainBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        first_dilation_size: int,
        second_dilation_size: int,
        scale: int,
    ):
        super().__init__()

        self.first_layers = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Upsample(scale_factor=scale, mode="nearest"),
            nn.Conv1d(
                in_channels=input_size,
                out_channels=output_size,
                kernel_size=3,
                padding=first_dilation_size,
                dilation=first_dilation_size,
            ),
        )
        self.second_layers = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(
                in_channels=output_size,
                out_channels=output_size,
                kernel_size=3,
                padding=second_dilation_size,
                dilation=second_dilation_size,
            ),
        )

    def forward(self, x: Tensor, affine_scale: Tensor, affine_shift: Tensor):
        h = self.first_layers(x)
        h = affine(h, affine_scale=affine_scale, affine_shift=affine_shift)
        h = self.second_layers(h)
        return h


class UpsamplingSecondSubBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        first_dilation_size: int,
        second_dilation_size: int,
    ):
        super().__init__()

        self.first_layers = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=3,
                padding=first_dilation_size,
                dilation=first_dilation_size,
            ),
        )
        self.second_layers = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=3,
                padding=second_dilation_size,
                dilation=second_dilation_size,
            ),
        )

    def forward(self, x: Tensor, affine_scale: Tensor, affine_shift: Tensor):
        h = affine(x, affine_scale=affine_scale, affine_shift=affine_shift)
        h = self.first_layers(h)
        h = affine(h, affine_scale=affine_scale, affine_shift=affine_shift)
        h = self.second_layers(h)
        return h


class UpsamplingBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        dilation_size_list: Sequence[int],
        scale: int,
    ):
        super().__init__()

        self.first_main_block = UpsamplingFirstMainBlock(
            input_size=input_size,
            output_size=output_size,
            first_dilation_size=dilation_size_list[0],
            second_dilation_size=dilation_size_list[1],
            scale=scale,
        )

        self.first_sub_block = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode="nearest"),
            nn.Conv1d(
                in_channels=input_size,
                out_channels=output_size,
                kernel_size=1,
            ),
        )

        self.second_sub_block = UpsamplingSecondSubBlock(
            hidden_size=output_size,
            first_dilation_size=dilation_size_list[2],
            second_dilation_size=dilation_size_list[3],
        )

    def forward(self, x: Tensor, affine_scale: Tensor, affine_shift: Tensor):
        h = self.first_main_block(
            x, affine_scale=affine_scale, affine_shift=affine_shift
        ) + self.first_sub_block(x)
        h = h + self.second_sub_block(
            h, affine_scale=affine_scale, affine_shift=affine_shift
        )
        return h
