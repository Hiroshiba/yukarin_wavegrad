import numpy
import torch
from torch import Tensor, nn


class NoiseLevelPositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

        factor = (
            5000
            * torch.exp(
                torch.arange(0, hidden_size, 2).double()
                * (-numpy.log(10000.0) / hidden_size)
            )
        ).float()
        factor = factor.unsqueeze(0)
        self.register_buffer("factor", factor)

    def forward(self, noise_level: Tensor):
        r"""
        noise_level: (B, 1)
        return: (B, C)
        """
        h = torch.matmul(noise_level, self.factor)
        return torch.cat((torch.sin(h), torch.cos(h)), dim=1)


class FeatureWiseLinearModulation(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.main_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=output_size,
                kernel_size=3,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.positional_encoder = NoiseLevelPositionalEncoding(hidden_size=output_size)

        self.affine_scale_conv = nn.Conv1d(
            in_channels=output_size,
            out_channels=output_size,
            kernel_size=3,
            padding=1,
        )
        self.affine_shift_conv = nn.Conv1d(
            in_channels=output_size,
            out_channels=output_size,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: Tensor, noise_level: Tensor):
        r"""
        x: (B, C, T)
        noise_level: (B, 1)
        """
        h = self.main_layers(x)
        h = h + self.positional_encoder(noise_level).unsqueeze(
            2
        )  # (B, C, T) + (B, C, 1)
        affine_scale = self.affine_scale_conv(h)
        affine_shift = self.affine_shift_conv(h)
        return affine_scale, affine_shift
