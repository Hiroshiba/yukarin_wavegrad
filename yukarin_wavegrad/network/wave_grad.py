from typing import Sequence

from torch import Tensor, nn
from yukarin_wavegrad.network.downsampling_block import DownsamplingBlock
from yukarin_wavegrad.network.feature_wise_linear_modulation import (
    FeatureWiseLinearModulation,
)
from yukarin_wavegrad.network.upsampling_block import UpsamplingBlock


class WaveGrad(nn.Module):
    def __init__(
        self,
        input_size: int,
        scales: Sequence[int],
        upsampling_prev_hidden_size: int,
        upsampling_large_block_num: int,
        upsampling_hidden_sizes: Sequence[int],
        downsampling_prev_hidden_size: int,
        downsampling_hidden_sizes: Sequence[int],
    ):
        super().__init__()

        assert len(upsampling_hidden_sizes) == len(scales)
        assert len(downsampling_hidden_sizes) == len(scales) - 1

        self.prev_conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=upsampling_prev_hidden_size,
            kernel_size=3,
            padding=1,
        )

        self.upsampling_blocks = nn.ModuleList(
            [
                UpsamplingBlock(
                    input_size=(
                        upsampling_hidden_sizes[i - 1]
                        if i > 0
                        else upsampling_prev_hidden_size
                    ),
                    output_size=upsampling_hidden_sizes[i],
                    dilation_size_list=(
                        [1, 2, 4, 8] if i < upsampling_large_block_num else [1, 2, 1, 2]
                    ),
                    scale=scales[i],
                )
                for i in range(len(upsampling_hidden_sizes))
            ]
        )

        self.downsampling_blocks = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=1,
                    out_channels=downsampling_prev_hidden_size,
                    kernel_size=5,
                    padding=2,
                )
            ]
            + [
                DownsamplingBlock(
                    input_size=(
                        downsampling_hidden_sizes[i - 1]
                        if i > 0
                        else downsampling_prev_hidden_size
                    ),
                    output_size=downsampling_hidden_sizes[i],
                    dilation_size_list=[1, 2, 4],
                    scale=scales[-(i + 1)],
                )
                for i in range(len(downsampling_hidden_sizes))
            ]
        )

        self.modulation_blocks = nn.ModuleList(
            [
                FeatureWiseLinearModulation(
                    input_size=input_size,
                    output_size=output_size,
                )
                for input_size, output_size in zip(
                    [downsampling_prev_hidden_size] + downsampling_hidden_sizes,
                    reversed(upsampling_hidden_sizes),
                )
            ]
        )

        self.post_conv = nn.Conv1d(
            in_channels=upsampling_hidden_sizes[-1],
            out_channels=1,
            kernel_size=3,
            padding=1,
        )

    def forward(self, wave: Tensor, local: Tensor, noise_level: Tensor):
        r"""
        wave: (B, T)
        local: (B, C, T)
        noise_level: (B, 1)
        return: (B, T)
        """
        h = wave.unsqueeze(1)
        affine_params = []
        for downsampling_block, modulation_block in zip(
            self.downsampling_blocks, self.modulation_blocks
        ):
            h = downsampling_block(h)
            affine_param = modulation_block(h, noise_level=noise_level)
            affine_params.append(affine_param)

        h = local
        h = self.prev_conv(h)
        for upsampling_block, affine_param in zip(
            self.upsampling_blocks, reversed(affine_params)
        ):
            h = upsampling_block(
                h, affine_scale=affine_param[0], affine_shift=affine_param[1]
            )

        h = self.post_conv(h)
        return h.squeeze(1)
