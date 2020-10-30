from pathlib import Path
from typing import Union

import numpy
import torch
from acoustic_feature_extractor.data.wave import Wave

from yukarin_wavegrad.config import NetworkConfig, NoiseScheduleModelConfig
from yukarin_wavegrad.model import NoiseScheduler
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
        self.scale = numpy.prod(network_config.scales, dtype=int)

        if isinstance(predictor, Path):
            state_dict = torch.load(predictor)
            predictor = create_predictor(network_config)
            predictor.load_state_dict(state_dict)
        self.predictor = predictor.eval().to(self.device)

        self.noise_scheduler = NoiseScheduler(
            NoiseScheduler.generate_noise_schedule(
                start=noise_schedule_config.start,
                stop=noise_schedule_config.stop,
                num=noise_schedule_config.num,
            )
        ).to(self.device)

    def inference_forward(
        self,
        local: torch.Tensor,
        local_padding_length: int = 0,
        speaker_id: torch.Tensor = None,
    ):
        batsh_size = local.shape[0]
        length = local.shape[2] * self.scale - local_padding_length * 2

        wave = self.predictor.generate_noise(batsh_size, length)
        for i in reversed(range(self.noise_scheduler.max_iteration)):
            beta = self.noise_scheduler.beta[i].expand(batsh_size).unsqueeze(1)
            alpha = self.noise_scheduler.alpha[i].expand(batsh_size).unsqueeze(1)
            noise_level = (
                self.noise_scheduler.discrete_noise_level[i + 1]
                .expand(batsh_size)
                .unsqueeze(1)
            )

            diff_wave = self.predictor(
                wave=wave,
                local=local,
                local_padding_length=local_padding_length,
                noise_level=noise_level,
                speaker_id=speaker_id,
            )
            wave = (
                wave - (beta / torch.sqrt(1 - noise_level ** 2) * diff_wave)
            ) / torch.sqrt(alpha)

            if i > 0:
                noise = self.predictor.generate_noise(batsh_size, length)
                before_noise_level = self.noise_scheduler.discrete_noise_level[i]
                wave += (
                    torch.sqrt(beta * (1 - before_noise_level) / (1 - noise_level))
                    * noise
                )
        return wave

    def generate(
        self,
        local: Union[numpy.ndarray, torch.Tensor],
        local_padding_length: int = 0,
        speaker_id: Union[numpy.ndarray, torch.Tensor] = None,
    ):
        if isinstance(input, numpy.ndarray):
            local = torch.from_numpy(local)
        local = local.to(self.device)

        if speaker_id is not None:
            if isinstance(speaker_id, numpy.ndarray):
                speaker_id = torch.from_numpy(speaker_id)
            speaker_id = speaker_id.to(self.device)

        with torch.no_grad():
            output = self.inference_forward(
                local=local,
                local_padding_length=local_padding_length,
                speaker_id=speaker_id,
            )
        return [
            Wave(wave=wave, sampling_rate=self.sampling_rate)
            for wave in output.cpu().numpy()
        ]
