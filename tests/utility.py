from pathlib import Path
from pprint import pprint
from typing import Callable, Dict, Sequence, Tuple

import numpy
import torch
from acoustic_feature_extractor.data.sampling_data import SamplingData
from acoustic_feature_extractor.data.wave import Wave
from pytorch_trainer import Reporter
from pytorch_trainer.iterators import SerialIterator
from pytorch_trainer.training.updaters.standard_updater import StandardUpdater
from torch.optim import Adam
from torch.utils.data import Dataset
from yukarin_wavegrad.config import (
    DownsamplingNetworkConfig,
    EncodingNetworkConfig,
    ModelConfig,
    NetworkConfig,
    NoiseScheduleModelConfig,
    UpsamplingNetworkConfig,
)
from yukarin_wavegrad.dataset import BaseWaveDataset
from yukarin_wavegrad.model import Model
from yukarin_wavegrad.utility.dataset_utility import default_convert


def get_data_directory() -> Path:
    return Path(__file__).parent.relative_to(Path.cwd()) / "data"


def get_test_model_path(mulaw, iteration):
    return get_data_directory().joinpath(
        f"test_training-mulaw={mulaw}-iteration={iteration}.pth"
    )


def create_network_config(
    local_size: int = 1,
    scales: Sequence[int] = (5, 4, 4, 2, 2),
    speaker_size=0,
):
    return NetworkConfig(
        local_size=local_size,
        scales=scales,
        latent_size=0,
        speaker_size=speaker_size,
        speaker_embedding_size=4 if speaker_size > 0 else 0,
        encoding=EncodingNetworkConfig(
            hidden_size=0,
            layer_num=0,
        ),
        upsampling=UpsamplingNetworkConfig(
            prev_hidden_size=128,
            large_block_num=1,
            hidden_sizes=[128, 96, 64, 64, 32],
        ),
        downsampling=DownsamplingNetworkConfig(
            prev_hidden_size=32,
            hidden_sizes=[32, 64, 96, 128],
        ),
    )


def create_model_config():
    return ModelConfig(
        noise_schedule=NoiseScheduleModelConfig(start=1e-6, stop=1e-2, num=1000)
    )


class SignWaveDataset(BaseWaveDataset):
    def __init__(
        self,
        sampling_length: int,
        sampling_rate: int,
        mulaw: bool,
        local_scale: int,
        frequency_range: Tuple[float, float],
    ):
        super().__init__(
            sampling_length=sampling_length,
            local_sampling_rate=None,
            local_padding_length=0,
            min_not_silence_length=0,
            mulaw=mulaw,
        )
        self.sampling_rate = sampling_rate
        self.local_scale = local_scale
        self.frequency_range = frequency_range

    def __len__(self):
        return 100

    def __getitem__(self, i: int):
        sampling_rate = self.sampling_rate
        length = self.sampling_length
        frequency = numpy.random.uniform(
            self.frequency_range[0], self.frequency_range[1]
        )
        rand = numpy.random.rand()

        wave = numpy.sin(
            (2 * numpy.pi)
            * (
                numpy.arange(length, dtype=numpy.float32) * frequency / sampling_rate
                + rand
            )
        )

        local = numpy.log(
            numpy.ones(shape=(length // self.local_scale, 1), dtype=numpy.float32)
            * frequency
        )

        silence = numpy.zeros(shape=(length,), dtype=numpy.bool)

        return default_convert(
            self.make_input(
                wave_data=Wave(wave=wave, sampling_rate=sampling_rate),
                silence_data=SamplingData(array=silence, rate=sampling_rate),
                local_data=SamplingData(
                    array=local, rate=sampling_rate // self.local_scale
                ),
            )
        )


def create_sign_wave_dataset(sampling_rate: int = 4800, mulaw: bool = False):
    return SignWaveDataset(
        sampling_length=9600,
        sampling_rate=sampling_rate,
        mulaw=mulaw,
        local_scale=320,
        frequency_range=(220, 880),
    )


def train_support(
    batch_size: int,
    device: torch.device,
    model: Model,
    dataset: Dataset,
    iteration: int,
    first_hook: Callable[[Dict], None] = None,
    last_hook: Callable[[Dict], None] = None,
    learning_rate=2e-4,
):
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_iter = SerialIterator(dataset, batch_size)

    updater = StandardUpdater(
        iterator=train_iter,
        optimizer=optimizer,
        model=model,
        device=device,
    )

    reporter = Reporter()
    reporter.add_observer("main", model)

    observation: Dict = {}
    for i in range(iteration):
        with reporter.scope(observation):
            updater.update()

        if i % 100 == 0:
            print("iteration", i, end=" ")
            pprint(observation)

        if i == 0:
            if first_hook is not None:
                first_hook(observation)

    pprint(observation)
    if last_hook is not None:
        last_hook(observation)
