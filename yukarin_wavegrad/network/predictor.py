from typing import Optional

import numpy
import torch
from torch import Tensor, nn
from yukarin_wavegrad.config import NetworkConfig
from yukarin_wavegrad.network.wave_grad import WaveGrad


class Predictor(nn.Module):
    def __init__(
        self,
        local_scale: int,
        speaker_size: int,
        speaker_embedding_size: int,
        encoder: Optional[nn.RNNBase],
        wave_grad: WaveGrad,
    ):
        super().__init__()
        self.local_scale = local_scale

        self.speaker_embedder: Optional[nn.Embedding] = None
        if speaker_size > 0:
            self.speaker_embedder = nn.Embedding(
                num_embeddings=speaker_size,
                embedding_dim=speaker_embedding_size,
            )

        self.encoder = encoder
        self.wave_grad = wave_grad

    @property
    def with_speaker(self):
        return self.speaker_embedder is not None

    @property
    def with_encoder(self):
        return self.encoder is not None

    @property
    def device(self):
        return self.wave_grad.prev_conv.weight.device

    def generate_noise(self, *shape):
        return torch.randn(*shape).float().to(self.device)

    def forward(
        self,
        wave: Tensor,
        local: Tensor,
        local_padding_length: int,
        noise_level: Tensor,
        speaker_id: Tensor = None,
    ):
        if self.with_speaker:
            speaker = self.speaker_embedder(speaker_id)
            speaker = speaker.unsqueeze(2)
            speaker = speaker.expand(speaker.shape[0], speaker.shape[1], local.shape[2])
            condition = torch.cat((local, speaker), dim=1)
        else:
            condition = local

        if self.with_encoder:
            condition = condition.transpose(1, 2)
            encoded, _ = self.encoder(condition)
            encoded = encoded.transpose(1, 2)
        else:
            encoded = condition

        if local_padding_length > 0:
            l_pad = local_padding_length // self.local_scale
            encoded = encoded[:, :, l_pad:-l_pad]

        return self.wave_grad(wave=wave, local=encoded, noise_level=noise_level)


def create_predictor(config: NetworkConfig):
    with_encoder = config.encoding.layer_num > 0

    condition_size = (
        config.local_size + config.speaker_embedding_size + config.latent_size
    )
    encoded_size = config.encoding.hidden_size * 2 if with_encoder else condition_size

    encoder: Optional[nn.RNNBase] = None
    if with_encoder:
        encoder = nn.GRU(
            input_size=condition_size,
            hidden_size=config.encoding.hidden_size,
            num_layers=config.encoding.layer_num,
            batch_first=True,
            bidirectional=True,
        )

    return Predictor(
        local_scale=numpy.prod(config.scales),
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
        encoder=encoder,
        wave_grad=WaveGrad(
            input_size=encoded_size,
            scales=config.scales,
            upsampling_prev_hidden_size=config.upsampling.prev_hidden_size,
            upsampling_large_block_num=config.upsampling.large_block_num,
            upsampling_hidden_sizes=config.upsampling.hidden_sizes,
            downsampling_prev_hidden_size=config.downsampling.prev_hidden_size,
            downsampling_hidden_sizes=config.downsampling.hidden_sizes,
        ),
    )
