from typing import Callable

import torch
from torch import nn


def init_weights(model: torch.nn.Module, name: str):
    def _init_weights(layer: nn.Module):
        initializer: Callable
        if name == "uniform":
            initializer = torch.nn.init.uniform_
        elif name == "normal":
            initializer = torch.nn.init.normal_
        elif name == "xavier_uniform":
            initializer = torch.nn.init.xavier_uniform_
        elif name == "xavier_normal":
            initializer = torch.nn.init.xavier_normal_
        elif name == "kaiming_uniform":
            initializer = torch.nn.init.kaiming_uniform_
        elif name == "kaiming_normal":
            initializer = torch.nn.init.kaiming_normal_
        elif name == "orthogonal":
            initializer = torch.nn.init.orthogonal_
        elif name == "sparse":
            initializer = torch.nn.init.sparse_
        else:
            raise ValueError(name)

        for key, param in layer.named_parameters():
            if "weight" in key:
                initializer(param)

    model.apply(_init_weights)
