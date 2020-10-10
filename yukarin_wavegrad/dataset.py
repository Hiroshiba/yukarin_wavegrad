import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy
from torch.utils.data.dataset import ConcatDataset, Dataset

from yukarin_wavegrad.config import DatasetConfig
from yukarin_wavegrad.utility.dataset_utility import default_convert


@dataclass
class DatasetInputData:
    input_path: Path
    target_path: Path


class InputTargetDataset(Dataset):
    def __init__(
        self,
        datas: Sequence[DatasetInputData],
    ):
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        data = self.datas[i]
        input = numpy.load(str(data.input_path), allow_pickle=True)
        target = numpy.load(str(data.target_path), allow_pickle=True)

        return default_convert(
            dict(
                input=input,
                target=target,
            )
        )


def create_dataset(config: DatasetConfig):
    input_paths = {Path(p).stem: Path(p) for p in glob.glob(str(config.input_glob))}
    assert len(input_paths) > 0

    target_paths = {Path(p).stem: Path(p) for p in glob.glob(str(config.target_glob))}
    assert len(input_paths) == len(target_paths)

    inputs = [
        DatasetInputData(
            input_path=input_path,
            target_path=target_path,
        )
        for input_path, target_path in zip(input_paths, target_paths)
    ]

    if config.seed is not None:
        numpy.random.RandomState(config.seed).shuffle(inputs)

    tests, trains = inputs[: config.num_test], inputs[config.num_test :]
    train_tests = trains[: config.num_test]

    def dataset_wrapper(datas, is_test: bool):
        dataset = InputTargetDataset(
            datas=datas,
        )
        if is_test:
            dataset = ConcatDataset([dataset] * config.num_times_test)
        return dataset

    return {
        "train": dataset_wrapper(trains, is_test=False),
        "test": dataset_wrapper(tests, is_test=True),
        "train_test": dataset_wrapper(train_tests, is_test=True),
    }
