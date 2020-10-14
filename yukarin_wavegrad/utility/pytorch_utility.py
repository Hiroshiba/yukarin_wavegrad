import torch


def init_orthogonal(model: torch.nn.Module):
    def init_weights(layer):
        if isinstance(layer, torch.nn.modules.Conv1d):
            torch.nn.init.orthogonal_(layer.weight)

    model.apply(init_weights)
