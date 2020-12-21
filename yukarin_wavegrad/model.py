from typing import Sequence

import numpy
import torch
from pytorch_trainer import report
from torch import Tensor, nn

from yukarin_wavegrad.config import ModelConfig
from yukarin_wavegrad.network.predictor import Predictor


class NoiseScheduler(nn.Module):
    def __init__(self, noise_schedule: Sequence[float]):
        super().__init__()

        self.max_iteration = len(noise_schedule)

        beta = numpy.asarray(noise_schedule, dtype=numpy.float64)
        alpha = 1 - beta
        discrete_noise_level = numpy.r_[1, numpy.sqrt(numpy.cumprod(alpha))]

        self.register_buffer("beta", torch.from_numpy(beta).float())
        self.register_buffer("alpha", torch.from_numpy(alpha).float())
        self.register_buffer(
            "discrete_noise_level", torch.from_numpy(discrete_noise_level).float()
        )

    @property
    def device(self):
        return self.beta.device

    @staticmethod
    def generate_noise_schedule(start: float, stop: float, num: int):
        return numpy.linspace(start, stop, num=num)

    def sample_noise_level(self, num: int):
        r"""
        return: (num, 1)
        """
        r = torch.randint(self.max_iteration, size=(num,))
        ma = self.discrete_noise_level[r]
        mi = self.discrete_noise_level[r + 1]
        noise_level = torch.rand(num).float().to(self.device) * (ma - mi) + mi
        return noise_level.unsqueeze(1)


class Model(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        predictor: Predictor,
        local_padding_length: int,
    ):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor
        self.local_padding_length = local_padding_length

        self.l1_loss = nn.L1Loss()

        self.noise_scheduler = NoiseScheduler(
            NoiseScheduler.generate_noise_schedule(
                start=model_config.noise_schedule.start,
                stop=model_config.noise_schedule.stop,
                num=model_config.noise_schedule.num,
            )
        )

    def __call__(
        self,
        wave: Tensor,
        local: Tensor,
        speaker_id: Tensor = None,
    ):
        batch_size = wave.shape[0]

        noise_level = self.noise_scheduler.sample_noise_level(num=batch_size)
        noise = self.predictor.generate_noise(*wave.shape)
        noised_wave = noise_level * wave + torch.sqrt(1 - noise_level ** 2) * noise

        output = self.predictor(
            wave=noised_wave,
            local=local,
            local_padding_length=self.local_padding_length,
            noise_level=noise_level,
            speaker_id=speaker_id,
        )

        loss = self.l1_loss(output, noise)

        # report
        values = dict(loss=loss)
        if not self.training:
            values = {key: (l, batch_size) for key, l in values.items()}  # add weight
        report(values, self)

        return loss
