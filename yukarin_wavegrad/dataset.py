import glob
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Union

import librosa
import numpy
from acoustic_feature_extractor.data.sampling_data import SamplingData
from acoustic_feature_extractor.data.wave import Wave
from torch.utils.data import ConcatDataset, Dataset

from yukarin_wavegrad.config import DatasetConfig
from yukarin_wavegrad.utility.dataset_utility import default_convert


@dataclass
class Input:
    wave: Wave
    silence: SamplingData
    local: SamplingData


@dataclass
class LazyInput:
    path_wave: Path
    path_silence: Path
    path_local: Path

    def generate(self):
        return Input(
            wave=Wave.load(self.path_wave),
            silence=SamplingData.load(self.path_silence),
            local=SamplingData.load(self.path_local),
        )


@dataclass
class SpectrogramLazyInput(LazyInput):
    path_local: Path = None

    def generate(self):
        wave = Wave.load(self.path_wave)

        if self.path_local is None or not self.path_local.exists():
            local_rate = 80
            local_array = to_log_melspectrogram(wave=wave, rate=local_rate)
            local = SamplingData(array=local_array, rate=local_rate)

            with NamedTemporaryFile(suffix=".npy", delete=False) as f:
                self.path_local = Path(f.name)
                local.save(self.path_local)
        else:
            local = SamplingData.load(self.path_local)

        return Input(
            wave=wave,
            silence=SamplingData.load(self.path_silence),
            local=local,
        )


def to_log_melspectrogram(wave: Wave, rate: int):
    assert wave.sampling_rate == 24000
    s = librosa.feature.melspectrogram(
        wave.wave,
        sr=wave.sampling_rate,
        n_fft=2048,
        hop_length=wave.sampling_rate // rate,
        win_length=wave.sampling_rate // rate * 4,
        window="hann",
        center=True,
        pad_mode="reflect",
        n_mels=80,
        fmin=20,
        fmax=12000,
    ).T
    s = numpy.log10(numpy.maximum(s, 1e-10))
    return numpy.ascontiguousarray(s).astype(numpy.float32)


class BaseWaveDataset(Dataset):
    def __init__(
        self,
        sampling_length: int,
        local_padding_length: int,
        min_not_silence_length: int,
    ):
        self.sampling_length = sampling_length
        self.local_padding_length = local_padding_length
        self.min_not_silence_length = min_not_silence_length

    @staticmethod
    def extract_input(
        sampling_length: int,
        wave_data: Wave,
        silence_data: SamplingData,
        local_data: SamplingData,
        local_padding_length: int,
        min_not_silence_length: int,
        padding_value=0,
    ):
        """
        :return:
            wave: (sampling_length, )
            silence: (sampling_length, )
            local: (sampling_length // scale + pad, )
        """
        sr = wave_data.sampling_rate
        sl = sampling_length

        assert sr % local_data.rate == 0
        l_scale = int(sr // local_data.rate)

        length = len(local_data.array) * l_scale
        assert (
            abs(length - len(wave_data.wave)) < l_scale * 4
        ), f"{abs(length - len(wave_data.wave))} {l_scale}"

        assert local_padding_length % l_scale == 0
        l_pad = local_padding_length // l_scale

        l_length = length // l_scale
        l_sl = sl // l_scale

        for _ in range(10000):
            if l_length > l_sl:
                l_offset = numpy.random.randint(l_length - l_sl)
            else:
                l_offset = 0
            offset = l_offset * l_scale

            silence = numpy.squeeze(silence_data.resample(sr, index=offset, length=sl))
            if (~silence).sum() >= min_not_silence_length:
                break
        else:
            raise Exception("cannot pick not silence data")

        wave = wave_data.wave[offset : offset + sl]

        # local
        l_start, l_end = l_offset - l_pad, l_offset + l_sl + l_pad
        if l_start < 0 or l_end > l_length:
            shape = list(local_data.array.shape)
            shape[0] = l_sl + l_pad * 2
            local = (
                numpy.ones(shape=shape, dtype=local_data.array.dtype) * padding_value
            )
            if l_start < 0:
                p_start = -l_start
                l_start = 0
            else:
                p_start = 0
            if l_end > l_length:
                p_end = l_sl + l_pad * 2 - (l_end - l_length)
                l_end = l_length
            else:
                p_end = l_sl + l_pad * 2
            local[p_start:p_end] = local_data.array[l_start:l_end]
        else:
            local = local_data.array[l_start:l_end]

        return dict(
            wave=wave,
            local=local.T,  # (C, T)
        )

    def make_input(
        self,
        wave_data: Wave,
        silence_data: SamplingData,
        local_data: SamplingData,
    ):
        return self.extract_input(
            sampling_length=self.sampling_length,
            wave_data=wave_data,
            silence_data=silence_data,
            local_data=local_data,
            local_padding_length=self.local_padding_length,
            min_not_silence_length=self.min_not_silence_length,
        )


class WavesDataset(BaseWaveDataset):
    def __init__(
        self,
        inputs: List[Union[Input, LazyInput]],
        sampling_length: int,
        local_padding_length: int,
        min_not_silence_length: int,
    ):
        super().__init__(
            sampling_length=sampling_length,
            local_padding_length=local_padding_length,
            min_not_silence_length=min_not_silence_length,
        )
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        input = self.inputs[i]
        if isinstance(input, LazyInput):
            input = input.generate()

        return default_convert(
            self.make_input(
                wave_data=input.wave,
                silence_data=input.silence,
                local_data=input.local,
            )
        )


class SpeakerWavesDataset(Dataset):
    def __init__(self, wave_dataset: Dataset, speaker_ids: List[int]):
        assert len(wave_dataset) == len(speaker_ids)
        self.wave_dataset = wave_dataset
        self.speaker_ids = speaker_ids

    def __len__(self):
        return len(self.wave_dataset)

    def __getitem__(self, i):
        d = self.wave_dataset[i]
        d["speaker_id"] = numpy.array(self.speaker_ids[i], dtype=numpy.long)
        return default_convert(d)


def create_dataset(config: DatasetConfig):
    wave_paths = {Path(p).stem: Path(p) for p in glob.glob(config.input_wave_glob)}
    fn_list = sorted(wave_paths.keys())
    assert len(fn_list) > 0

    silence_paths = {
        Path(p).stem: Path(p) for p in glob.glob(config.input_silence_glob)
    }
    assert set(fn_list) == set(silence_paths.keys())

    local_paths = None
    if config.input_local_glob is not None:
        local_paths = {
            Path(p).stem: Path(p) for p in glob.glob(config.input_local_glob)
        }
        assert set(fn_list) == set(local_paths.keys())

    speaker_ids = None
    if config.speaker_dict_path is not None:
        fn_each_speaker: Dict[str, List[str]] = json.load(
            open(config.speaker_dict_path)
        )
        assert config.speaker_size == len(fn_each_speaker)

        speaker_ids = {
            fn: speaker_id
            for speaker_id, (_, fns) in enumerate(fn_each_speaker.items())
            for fn in fns
        }
        assert set(fn_list).issubset(set(speaker_ids.keys()))

    numpy.random.RandomState(config.seed).shuffle(fn_list)

    num_test = config.num_test
    num_train = (
        config.num_train if config.num_train is not None else len(fn_list) - num_test
    )

    trains = fn_list[num_test:][:num_train]
    tests = fn_list[:num_test]

    def make_dataset(fns, for_evaluate=False):
        inputs = [
            LazyInput(
                path_wave=wave_paths[fn],
                path_silence=silence_paths[fn],
                path_local=local_paths[fn],
            )
            if local_paths is not None
            else SpectrogramLazyInput(
                path_wave=wave_paths[fn],
                path_silence=silence_paths[fn],
            )
            for fn in fns
        ]

        if not for_evaluate:
            sampling_length = config.sampling_length
            local_padding_length = config.local_padding_length
        else:
            sampling_length = int(config.evaluate_time_second * config.sampling_rate)
            local_padding_length = int(
                config.evaluate_local_padding_time_second * config.sampling_rate
            )

        dataset = WavesDataset(
            inputs=inputs,
            sampling_length=sampling_length,
            local_padding_length=local_padding_length,
            min_not_silence_length=config.min_not_silence_length,
        )

        if speaker_ids is not None:
            dataset = SpeakerWavesDataset(
                wave_dataset=dataset,
                speaker_ids=[speaker_ids[fn] for fn in fns],
            )

        if for_evaluate:
            dataset = ConcatDataset([dataset] * config.evaluate_times)

        return dataset

    return dict(
        train=make_dataset(trains),
        test=make_dataset(tests),
        eval=make_dataset(tests, for_evaluate=True),
    )
