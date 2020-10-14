import torch
from retry import retry
from yukarin_wavegrad.model import Model
from yukarin_wavegrad.network.predictor import create_predictor

from tests.utility import (
    create_model_config,
    create_network_config,
    create_sign_wave_dataset,
    train_support,
)


@retry(tries=3)
def test_train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    predictor = create_predictor(create_network_config()).to(device)
    model = Model(model_config=create_model_config(), predictor=predictor)
    dataset = create_sign_wave_dataset()

    def first_hook(o):
        assert o["main/loss"].data > 0.5

    def last_hook(o):
        assert o["main/loss"].data < 0.3

    iteration = 1000
    train_support(
        batch_size=16,
        device=device,
        model=model,
        dataset=dataset,
        iteration=iteration,
        first_hook=first_hook,
        last_hook=last_hook,
        learning_rate=3e-5,
    )

    # save model
    torch.save(
        model.predictor.state_dict(),
        f"/tmp/test_training-iteration={iteration}.pth",
    )
