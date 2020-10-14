from pathlib import Path
from typing import Union

import numpy
import torch
from acoustic_feature_extractor.data.wave import Wave

from yukarin_wavegrad.config import NetworkConfig, NoiseScheduleModelConfig
from yukarin_wavegrad.network.predictor import Predictor, create_predictor


class Generator(object):
    def __init__(
        self,
        network_config: NetworkConfig,
        noise_schedule_config: NoiseScheduleModelConfig,
        predictor: Union[Predictor, Path],
        sampling_rate: int,
        use_gpu: bool = True,
    ):
        self.sampling_rate = sampling_rate
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

        if isinstance(predictor, Path):
            state_dict = torch.load(predictor)
            predictor = create_predictor(network_config)
            predictor.load_state_dict(state_dict)
        self.predictor = predictor.eval().to(self.device)

        noise_schedule = predictor.generate_noise_schedule(
            start=noise_schedule_config.start,
            stop=noise_schedule_config.stop,
            num=noise_schedule_config.num,
        )
        predictor.set_noise_schedule(noise_schedule)

    def generate(
        self,
        local: Union[numpy.ndarray, torch.Tensor],
    ):
        if isinstance(input, numpy.ndarray):
            local = torch.from_numpy(local)
        local = local.to(self.device)

        with torch.no_grad():
            output = self.predictor.inference_forward(local)
        return [
            Wave(wave=wave, sampling_rate=self.sampling_rate)
            for wave in output.cpu().numpy()
        ]
