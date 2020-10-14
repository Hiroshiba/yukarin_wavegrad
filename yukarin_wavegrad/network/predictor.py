from typing import Optional

import torch
from torch import Tensor, nn
from yukarin_wavegrad.config import NetworkConfig
from yukarin_wavegrad.network.wave_grad import WaveGrad


class Predictor(nn.Module):
    def __init__(
        self,
        wave_grad: WaveGrad,
        speaker_size: int,
        speaker_embedding_size: int,
    ):
        super().__init__()

        self.wave_grad = wave_grad
        self.speaker_size = speaker_size

        self.speaker_embedder: Optional[nn.Embedding] = None
        if self.with_speaker:
            self.speaker_embedder = nn.Embedding(
                num_embeddings=speaker_size,
                embedding_dim=speaker_embedding_size,
            )

    @property
    def with_speaker(self):
        return self.speaker_size > 0

    @property
    def device(self):
        return self.wave_grad.prev_conv.weight.device

    def generate_noise(self, *shape):
        return torch.randn(*shape).float().to(self.device)

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
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
    )
