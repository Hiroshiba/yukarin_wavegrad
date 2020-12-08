from pathlib import Path

import pytest
import torch
from pytorch_trainer.dataset.convert import concat_examples
from yukarin_wavegrad.config import NoiseScheduleModelConfig
from yukarin_wavegrad.generator import Generator

from tests.utility import (
    create_network_config,
    create_sign_wave_dataset,
    get_test_model_path,
)


@pytest.mark.parametrize("mulaw", [False, True])
def test_generator(mulaw: bool):
    iteration = 100000
    sampling_rate = 4800

    dataset_config = type(
        "DatasetConfig",
        (object,),
        dict(
            mulaw=mulaw,
        ),
    )

    config = type(
        "Config",
        (object,),
        dict(
            network=create_network_config(),
            dataset=dataset_config,
        ),
    )

    generator = Generator(
        config=config,
        noise_schedule_config=NoiseScheduleModelConfig(start=1e-4, stop=0.05, num=50),
        predictor=get_test_model_path(mulaw=mulaw, iteration=iteration),
        sampling_rate=sampling_rate,
        use_gpu=torch.cuda.is_available(),
    )

    dataset = create_sign_wave_dataset(sampling_rate=sampling_rate)
    batch = concat_examples([dataset[i] for i in range(4)])
    waves = generator.generate(local=batch["local"])
    for num, wave in enumerate(waves):
        wave.save(
            Path(
                "/tmp/"
                f"test_generator_audio"
                f"-num={num}"
                f"-mulaw={mulaw}"
                f"-iteration={iteration}"
                ".wav"
            )
        )
