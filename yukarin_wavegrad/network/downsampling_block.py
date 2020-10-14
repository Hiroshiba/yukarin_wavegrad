from typing import Sequence

from torch import Tensor, nn


class DownsamplingBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        dilation_size_list: Sequence[int],
        scale: int,
    ):
        super().__init__()

        self.main_block = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=input_size,
                kernel_size=scale,
                stride=scale,
                groups=input_size,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(
                in_channels=input_size,
                out_channels=output_size,
                kernel_size=3,
                padding=dilation_size_list[0],
                dilation=dilation_size_list[0],
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(
                in_channels=output_size,
                out_channels=output_size,
                kernel_size=3,
                padding=dilation_size_list[1],
                dilation=dilation_size_list[1],
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(
                in_channels=output_size,
                out_channels=output_size,
                kernel_size=3,
                padding=dilation_size_list[2],
                dilation=dilation_size_list[2],
            ),
        )

        self.sub_block = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=output_size,
                kernel_size=1,
            ),
            nn.Conv1d(
                in_channels=output_size,
                out_channels=output_size,
                kernel_size=scale,
                stride=scale,
                groups=output_size,
            ),
        )

    def forward(self, x: Tensor):
        h = self.main_block(x) + self.sub_block(x)
        return h
