from pytorch_trainer import report
from torch import Tensor, nn

from yukarin_wavegrad.config import ModelConfig
from yukarin_wavegrad.network.predictor import Predictor


class Model(nn.Module):
    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

        self.l1_loss = nn.L1Loss()

        noise_schedule = predictor.generate_noise_schedule(
            start=model_config.noise_schedule.start,
            stop=model_config.noise_schedule.stop,
            num=model_config.noise_schedule.num,
        )
        predictor.set_noise_schedule(noise_schedule)

    def __call__(
        self,
        wave: Tensor,
        local: Tensor,
        speaker_id: Tensor = None,
    ):
        batch_size = wave.shape[0]

        noise_level = self.predictor.sample_noise_level(num=batch_size)

        noise = self.predictor.generate_noise(*wave.shape)
        output = self.predictor.train_forward(
            wave=wave,
            noise=noise,
            local=local,
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
