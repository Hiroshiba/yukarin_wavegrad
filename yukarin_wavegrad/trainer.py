import warnings
from copy import copy
from functools import partial
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from pytorch_trainer.iterators import MultiprocessIterator
from pytorch_trainer.training import Trainer, extensions
from pytorch_trainer.training.updaters import StandardUpdater
from tensorboardX import SummaryWriter
from torch import optim
from torch.optim.optimizer import Optimizer

from yukarin_wavegrad.config import Config, NoiseScheduleModelConfig
from yukarin_wavegrad.dataset import create_dataset
from yukarin_wavegrad.evaluator import GenerateEvaluator
from yukarin_wavegrad.generator import Generator
from yukarin_wavegrad.model import Model
from yukarin_wavegrad.network.predictor import create_predictor
from yukarin_wavegrad.utility.amp_updater import AmpUpdater
from yukarin_wavegrad.utility.pytorch_utility import init_weights
from yukarin_wavegrad.utility.trainer_extension import TensorboardReport, WandbReport
from yukarin_wavegrad.utility.trainer_utility import LowValueTrigger, create_iterator

try:
    from torch.cuda import amp

    amp_exist = True
except ImportError:
    amp_exist = False


def create_trainer(
    config_dict: Dict[str, Any],
    output: Path,
):
    # config
    config = Config.from_dict(config_dict)
    config.add_git_info()

    output.mkdir(exist_ok=True, parents=True)
    with (output / "config.yaml").open(mode="w") as f:
        yaml.safe_dump(config.to_dict(), f)

    # model
    device = torch.device("cuda")
    predictor = create_predictor(config.network)
    model = Model(
        model_config=config.model,
        predictor=predictor,
        local_padding_length=config.dataset.local_padding_length,
    )
    init_weights(model, "orthogonal")
    model.to(device)

    # dataset
    _create_iterator = partial(
        create_iterator,
        batch_size=config.train.batchsize,
        eval_batch_size=config.train.eval_batchsize,
        num_processes=config.train.num_processes,
        use_multithread=config.train.use_multithread,
    )

    datasets = create_dataset(config.dataset)
    train_iter = _create_iterator(datasets["train"], for_train=True, for_eval=False)
    test_iter = _create_iterator(datasets["test"], for_train=False, for_eval=False)
    eval_iter = _create_iterator(datasets["eval"], for_train=False, for_eval=True)

    warnings.simplefilter("error", MultiprocessIterator.TimeoutWarning)

    # optimizer
    cp: Dict[str, Any] = copy(config.train.optimizer)
    n = cp.pop("name").lower()

    optimizer: Optimizer
    if n == "adam":
        optimizer = optim.Adam(model.parameters(), **cp)
    elif n == "sgd":
        optimizer = optim.SGD(model.parameters(), **cp)
    else:
        raise ValueError(n)

    # updater
    use_amp = config.train.use_amp if config.train.use_amp is not None else amp_exist
    if use_amp:
        updater = AmpUpdater(
            iterator=train_iter,
            optimizer=optimizer,
            model=model,
            device=device,
        )
    else:
        updater = StandardUpdater(
            iterator=train_iter,
            optimizer=optimizer,
            model=model,
            device=device,
        )

    # trainer
    trigger_log = (config.train.log_iteration, "iteration")
    trigger_eval = (config.train.eval_iteration, "iteration")
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

    if config.train.multistep_shift is not None:
        trainer.extend(extensions.MultistepShift(**config.train.multistep_shift))

    ext = extensions.Evaluator(test_iter, model, device=device)
    trainer.extend(ext, name="test", trigger=trigger_log)

    generator = Generator(
        config=config,
        noise_schedule_config=NoiseScheduleModelConfig(start=1e-4, stop=0.05, num=50),
        predictor=predictor,
        sampling_rate=config.dataset.sampling_rate,
        use_gpu=True,
    )
    generate_evaluator = GenerateEvaluator(
        generator=generator,
        local_padding_time_second=config.dataset.evaluate_local_padding_time_second,
    )
    ext = extensions.Evaluator(eval_iter, generate_evaluator, device=device)
    trainer.extend(ext, name="eval", trigger=trigger_eval)

    ext = extensions.snapshot_object(
        predictor,
        filename="predictor_{.updater.iteration}.pth",
        n_retains=5,
    )
    trainer.extend(
        ext,
        trigger=LowValueTrigger("eval/main/mcd", trigger=trigger_eval),
    )

    trainer.extend(extensions.FailOnNonNumber(), trigger=trigger_log)
    trainer.extend(extensions.observe_lr(), trigger=trigger_log)
    trainer.extend(extensions.LogReport(trigger=trigger_log))
    trainer.extend(
        extensions.PrintReport(["iteration", "main/loss", "test/main/loss"]),
        trigger=trigger_log,
    )

    trainer.extend(ext, trigger=TensorboardReport(writer=writer))

    if config.project.category is not None:
        ext = WandbReport(
            config_dict=config.to_dict(),
            project_category=config.project.category,
            project_name=config.project.name,
            output_dir=output.joinpath("wandb"),
        )
        trainer.extend(ext, trigger=trigger_log)

    (output / "struct.txt").write_text(repr(model))

    if trigger_stop is not None:
        trainer.extend(extensions.ProgressBar(trigger_stop))

    ext = extensions.snapshot_object(
        trainer,
        filename="trainer_{.updater.iteration}.pth",
        n_retains=1,
        autoload=True,
    )
    trainer.extend(ext, trigger=trigger_eval)

    return trainer
