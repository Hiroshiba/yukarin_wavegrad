import pytest
import torch
from retry import retry
from yukarin_wavegrad.model import Model
from yukarin_wavegrad.network.predictor import create_predictor
from yukarin_wavegrad.utility.pytorch_utility import init_weights

from tests.utility import (
    create_model_config,
    create_network_config,
    create_sign_wave_dataset,
    train_support,
)

iteration = 500


@retry(tries=3)
@pytest.mark.parametrize("mulaw", [False, True])
def test_train(mulaw: bool):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    predictor = create_predictor(create_network_config())
    model = Model(
        model_config=create_model_config(),
        predictor=predictor,
        local_padding_length=0,
    )
    init_weights(model, "orthogonal")
    model.to(device)

    dataset = create_sign_wave_dataset(mulaw=mulaw)

    def first_hook(o):
        assert o["main/loss"].data > 0.5

    def last_hook(o):
        assert o["main/loss"].data < 0.5

    train_support(
        batch_size=16,
        device=device,
        model=model,
        dataset=dataset,
        iteration=iteration,
        first_hook=first_hook,
        last_hook=last_hook,
        learning_rate=2e-4,
    )

    # save model
    torch.save(
        model.predictor.state_dict(),
        f"/tmp/test_training-mulaw={mulaw}-iteration={iteration}.pth",
    )
