import torch.nn.functional as F
from torch import nn, Tensor

from yukarin_wavegrad.config import NetworkConfig


class Predictor(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
    ):
        pass


def create_predictor(config: NetworkConfig):
    return Predictor()
