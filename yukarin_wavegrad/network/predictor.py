from typing import Optional, Sequence

import numpy
import torch
from torch import Tensor, nn
from yukarin_wavegrad.config import NetworkConfig
from yukarin_wavegrad.network.wave_grad import WaveGrad


class Predictor(nn.Module):
    def __init__(
        self,
        wave_grad: WaveGrad,
        scale: int,
        speaker_size: int,
        speaker_embedding_size: int,
    ):
        super().__init__()

        self.wave_grad = wave_grad
        self.scale = scale
        self.speaker_size = speaker_size

        self.speaker_embedder: Optional[nn.Embedding] = None
        if self.with_speaker:
            self.speaker_embedder = nn.Embedding(
                num_embeddings=speaker_size,
                embedding_dim=speaker_embedding_size,
            )

        self.beta: Optional[Tensor] = None  # (N, )
        self.alpha: Optional[Tensor] = None  # (N, )
        self.discrete_noise_level: Optional[Tensor] = None  # (N+1, )
        self.max_iteration: Optional[int] = None

    @property
    def with_speaker(self):
        return self.speaker_size > 0

    @property
    def device(self):
        return self.wave_grad.prev_conv.weight.device

    @staticmethod
    def generate_noise_schedule(start: float, stop: float, num: int):
        return numpy.linspace(start, stop, num=num)

    def generate_noise(self, *shape):
        return torch.randn(*shape).float().to(self.wave_grad.prev_conv.weight.device)

    def set_noise_schedule(self, noise_schedule: Sequence[float]):
        beta = numpy.asarray(noise_schedule, dtype=numpy.float64)
        alpha = 1 - beta
        discrete_noise_level = numpy.r_[1, numpy.sqrt(numpy.cumprod(alpha))]
        self.beta = torch.from_numpy(beta).float().to(self.device)
        self.alpha = torch.from_numpy(alpha).float().to(self.device)
        self.discrete_noise_level = (
            torch.from_numpy(discrete_noise_level).float().to(self.device)
        )
        self.max_iteration = len(noise_schedule)

    def sample_noise_level(self, num: int):
        r"""
        return: (num, 1)
        """
        r = torch.randint(self.max_iteration, size=(num,))
        mi = self.discrete_noise_level[r]
        ma = self.discrete_noise_level[r + 1]
        noise_level = (
            torch.rand(num).float().to(self.discrete_noise_level.device) * (ma - mi)
            + mi
        )
        return noise_level.unsqueeze(1)

    def forward(
        self,
        wave: Tensor,
        local: Tensor,
        noise_level: Tensor,
        speaker_id: Tensor = None,
    ):
        if self.with_speaker:
            speaker = self.speaker_embedder(speaker_id)
            speaker = speaker.unsqueeze(2)
            speaker = speaker.expand(speaker.shape[0], speaker.shape[1], local.shape[2])
            local = torch.cat((local, speaker), dim=1)

        return self.wave_grad(wave=wave, local=local, noise_level=noise_level)

    def train_forward(
        self,
        wave: Tensor,
        noise: Tensor,
        local: Tensor,
        noise_level: Tensor,
        speaker_id: Tensor = None,
    ):
        return self(
            wave=noise_level * wave + torch.sqrt(1 - noise_level ** 2) * noise,
            local=local,
            noise_level=noise_level,
            speaker_id=speaker_id,
        )

    def inference_forward(
        self,
        local: Tensor,
        speaker_id: Optional[Tensor] = None,
    ):
        batsh_size = local.shape[0]
        length = local.shape[2] * self.scale

        wave = self.generate_noise(batsh_size, length)
        for i in reversed(range(self.max_iteration)):
            beta = self.beta[i].expand(batsh_size).unsqueeze(1)
            alpha = self.alpha[i].expand(batsh_size).unsqueeze(1)
            noise_level = (
                self.discrete_noise_level[i + 1].expand(batsh_size).unsqueeze(1)
            )

            diff_wave = self(
                wave=wave, local=local, noise_level=noise_level, speaker_id=speaker_id
            )
            wave = (
                wave - (beta / torch.sqrt(1 - noise_level ** 2) * diff_wave)
            ) / torch.sqrt(alpha)

            if i > 0:
                noise = self.generate_noise(batsh_size, length)
                before_noise_level = self.discrete_noise_level[i]
                wave += (
                    torch.sqrt(beta * (1 - before_noise_level) / (1 - noise_level))
                    * noise
                )
        return wave


def create_predictor(config: NetworkConfig):
    return Predictor(
        wave_grad=WaveGrad(
            input_size=config.local_size + config.speaker_embedding_size,
            scales=config.scales,
            upsampling_prev_hidden_size=config.upsampling.prev_hidden_size,
            upsampling_large_block_num=config.upsampling.large_block_num,
            upsampling_hidden_sizes=config.upsampling.hidden_sizes,
            downsampling_prev_hidden_size=config.downsampling.prev_hidden_size,
            downsampling_hidden_sizes=config.downsampling.hidden_sizes,
        ),
        scale=numpy.prod(config.scales, dtype=int),
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
    )
