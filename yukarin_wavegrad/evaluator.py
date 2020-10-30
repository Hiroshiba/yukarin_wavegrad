from pathlib import Path
from typing import Optional

import numpy
from acoustic_feature_extractor.data.spectrogram import to_melcepstrum
from acoustic_feature_extractor.data.wave import Wave
from pytorch_trainer import report
from torch import Tensor, nn

from yukarin_wavegrad.generator import Generator

_logdb_const = 10.0 / numpy.log(10.0) * numpy.sqrt(2.0)


def _mcd(x: numpy.ndarray, y: numpy.ndarray) -> float:
    z = x - y
    r = numpy.sqrt((z * z).sum(axis=1)).mean()
    return _logdb_const * r


def calc_mcd(
    path1: Optional[Path] = None,
    path2: Optional[Path] = None,
    wave1: Optional[Wave] = None,
    wave2: Optional[Wave] = None,
):
    wave1 = Wave.load(path1) if wave1 is None else wave1
    wave2 = Wave.load(path2) if wave2 is None else wave2
    assert wave1.sampling_rate == wave2.sampling_rate

    sampling_rate = wave1.sampling_rate

    min_length = min(len(wave1.wave), len(wave2.wave))
    wave1.wave = wave1.wave[:min_length]
    wave2.wave = wave2.wave[:min_length]

    mc1 = to_melcepstrum(
        x=wave1.wave,
        sampling_rate=sampling_rate,
        n_fft=2048,
        win_length=1024,
        hop_length=256,
        order=24,
    )
    mc2 = to_melcepstrum(
        x=wave2.wave,
        sampling_rate=sampling_rate,
        n_fft=2048,
        win_length=1024,
        hop_length=256,
        order=24,
    )
    return _mcd(mc1, mc2)


class GenerateEvaluator(nn.Module):
    def __init__(self, generator: Generator, local_padding_time_second: float):
        super().__init__()
        self.generator = generator
        self.local_padding_time_second = local_padding_time_second

    def __call__(
        self,
        wave: Tensor,
        local: Tensor,
        speaker_id: Tensor = None,
    ):
        batch_size = len(wave)

        local_padding_length = int(
            self.generator.sampling_rate * self.local_padding_time_second
        )

        output = self.generator.generate(
            local=local,
            local_padding_length=local_padding_length,
            speaker_id=speaker_id,
        )

        mcd_list = []
        for wi, wo in zip(wave.cpu().numpy(), output):
            wi = Wave(wave=wi, sampling_rate=wo.sampling_rate)
            mcd = calc_mcd(wave1=wi, wave2=wo)
            mcd_list.append(mcd)

        scores = {
            "mcd": (numpy.mean(mcd_list), batch_size),
        }

        report(scores, self)
        return scores
