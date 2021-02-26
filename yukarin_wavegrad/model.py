from typing import Optional, Sequence

import numpy
import torch
import torch.nn.functional as F
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

        self.noise_scheduler = NoiseScheduler(
            NoiseScheduler.generate_noise_schedule(
                start=model_config.noise_schedule.start,
                stop=model_config.noise_schedule.stop,
                num=model_config.noise_schedule.num,
            )
        )

    def one_forward(
        self,
        noise_level: Tensor,
        noise: Tensor,
        wave: Tensor,
        latent: Optional[Tensor],
        local: Tensor,
        speaker_id: Tensor = None,
    ):
        noised_wave = noise_level * wave + torch.sqrt(1 - noise_level ** 2) * noise

        if latent is not None:
            local = torch.cat((local, latent), dim=1)

        output = self.predictor(
            wave=noised_wave,
            local=local,
            local_padding_length=self.local_padding_length,
            noise_level=noise_level,
            speaker_id=speaker_id,
        )

        loss = F.l1_loss(output, noise, reduction="none")
        return loss

    def forward(
        self,
        wave: Tensor,
        local: Tensor,
        speaker_id: Tensor = None,
    ):
        batch_size = wave.shape[0]
        sample_size = self.model_config.sample_size
        latent_size = self.model_config.latent_size

        noise_level = self.noise_scheduler.sample_noise_level(num=batch_size)
        noise = self.predictor.generate_noise(*wave.shape)

        latent = None
        if sample_size <= 1:
            assert latent_size == 0

        else:
            assert latent_size > 0

            latent_list = []
            for i_data in range(batch_size):
                latent = self.predictor.generate_noise(
                    sample_size, latent_size, local.shape[2]
                )

                with torch.no_grad():
                    loss = self.one_forward(
                        noise_level=noise_level[i_data : i_data + 1].expand(
                            (sample_size,) + noise_level.shape[1:]
                        ),
                        noise=noise[i_data : i_data + 1].expand(
                            (sample_size,) + noise.shape[1:]
                        ),
                        wave=wave[i_data : i_data + 1].expand(
                            (sample_size,) + wave.shape[1:]
                        ),
                        latent=latent,
                        local=local[i_data : i_data + 1].expand(
                            (sample_size,) + local.shape[1:]
                        ),
                        speaker_id=(
                            speaker_id[i_data : i_data + 1].expand(
                                (sample_size,) + speaker_id.shape[1:]
                            )
                            if speaker_id is not None
                            else None
                        ),
                    )

                i_sample = loss.mean(1).argmax(0)
                latent_list.append(latent[i_sample])

            latent = torch.stack(latent_list)

        loss = self.one_forward(
            noise_level=noise_level,
            noise=noise,
            wave=wave,
            latent=latent,
            local=local,
            speaker_id=speaker_id,
        ).mean()

        # report
        values = dict(loss=loss)
        if not self.training:
            values = {key: (l, batch_size) for key, l in values.items()}  # add weight
        report(values, self)

        return loss
