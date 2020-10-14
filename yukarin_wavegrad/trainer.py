import warnings
from copy import copy
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from pytorch_trainer.iterators import MultiprocessIterator
from pytorch_trainer.training import Trainer, extensions
from pytorch_trainer.training.updaters import StandardUpdater
from tensorboardX import SummaryWriter
from torch import optim

from yukarin_wavegrad.config import Config
from yukarin_wavegrad.dataset import create_dataset
from yukarin_wavegrad.model import Model
from yukarin_wavegrad.network.predictor import create_predictor
from yukarin_wavegrad.utility.pytorch_utility import init_orthogonal
from yukarin_wavegrad.utility.tensorboard_extension import TensorboardReport


def create_trainer(
    config_dict: Dict[str, Any],
    output: Path,
):
    # config
    config = Config.from_dict(config_dict)
    config.add_git_info()

    output.mkdir(parents=True)
    with (output / "config.yaml").open(mode="w") as f:
        yaml.safe_dump(config.to_dict(), f)

    # model
    device = torch.device("cuda")
    predictor = create_predictor(config.network).to(device)
    model = Model(model_config=config.model, predictor=predictor)
    init_orthogonal(model)

    # dataset
    def _create_iterator(dataset, for_train: bool):
        return MultiprocessIterator(
            dataset,
            config.train.batchsize,
            repeat=for_train,
            shuffle=for_train,
            n_processes=config.train.num_processes,
            dataset_timeout=60 * 15,
        )

    datasets = create_dataset(config.dataset)
    train_iter = _create_iterator(datasets["train"], for_train=True)
    test_iter = _create_iterator(datasets["test"], for_train=False)
    test_eval_iter = _create_iterator(datasets["test_eval"], for_train=False)

    warnings.simplefilter("error", MultiprocessIterator.TimeoutWarning)

    # optimizer
    cp: Dict[str, Any] = copy(config.train.optimizer)
    n = cp.pop("name").lower()

    if n == "adam":
        optimizer = optim.Adam(model.parameters(), **cp)
    elif n == "sgd":
        optimizer = optim.SGD(model.parameters(), **cp)
    else:
        raise ValueError(n)

    # updater
    updater = StandardUpdater(
        iterator=train_iter,
        optimizer=optimizer,
        model=model,
        device=device,
    )

    # trainer
    trigger_log = (config.train.log_iteration, "iteration")
    trigger_snapshot = (config.train.snapshot_iteration, "iteration")
    trigger_stop = (
        (config.train.stop_iteration, "iteration")
        if config.train.stop_iteration is not None
        else None
    )

    trainer = Trainer(updater, stop_trigger=trigger_stop, out=output)
    writer = SummaryWriter(Path(output))

    # # error at randint
    # sample_data = datasets["train"][0]
    # writer.add_graph(
    #     model,
    #     input_to_model=(
    #         sample_data["wave"].unsqueeze(0).to(device),
    #         sample_data["local"].unsqueeze(0).to(device),
    #         sample_data["speaker_id"].unsqueeze(0).to(device)
    #         if predictor.with_speaker
    #         else None,
    #     ),
    # )

    ext = extensions.Evaluator(test_iter, model, device=device)
    trainer.extend(ext, name="test", trigger=trigger_log)
    ext = extensions.Evaluator(test_eval_iter, model, device=device)
    trainer.extend(ext, name="train", trigger=trigger_log)

    ext = extensions.snapshot_object(
        predictor, filename="predictor_{.updater.iteration}.pth"
    )
    trainer.extend(ext, trigger=trigger_snapshot)

    trainer.extend(extensions.FailOnNonNumber(), trigger=trigger_log)
    trainer.extend(extensions.LogReport(trigger=trigger_log))
    trainer.extend(
        extensions.PrintReport(["iteration", "main/loss", "test/main/loss"]),
        trigger=trigger_log,
    )

    ext = TensorboardReport(writer=writer)
    trainer.extend(ext, trigger=trigger_log)

    (output / "struct.txt").write_text(repr(model))

    if trigger_stop is not None:
        trainer.extend(extensions.ProgressBar(trigger_stop))

    return trainer
