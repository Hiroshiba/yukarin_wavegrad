from pathlib import Path

import torch
from pytorch_trainer.dataset.convert import concat_examples
from yukarin_wavegrad.config import NoiseScheduleModelConfig
from yukarin_wavegrad.generator import Generator

from tests.utility import (
    create_network_config,
    create_sign_wave_dataset,
    get_test_model_path,
)


def test_generator():
    iteration = 100000
    sampling_rate = 16000

    generator = Generator(
        network_config=create_network_config(),
        noise_schedule_config=NoiseScheduleModelConfig(start=1e-4, stop=0.05, num=50),
        predictor=get_test_model_path(iteration=iteration),
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
                f"-iteration={iteration}"
                ".wav"
            )
        )
