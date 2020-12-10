import argparse
import multiprocessing
import warnings
from functools import partial
from pathlib import Path
from typing import Optional

import yaml
from pytorch_trainer.iterators import MultiprocessIterator
from tqdm import tqdm
from yukarin_wavegrad.config import Config
from yukarin_wavegrad.dataset import create_dataset


def _wrapper(index, dataset):
    try:
        dataset[index]
        return index, None
    except Exception as e:
        return index, e


def _check(dataset, desc: str, num_processes: Optional[int], batchsize: int):
    wrapper = partial(_wrapper, dataset=dataset)

    with multiprocessing.Pool(processes=num_processes) as pool:
        it = pool.imap_unordered(wrapper, range(len(dataset)), chunksize=2 ** 10)
        for i, error in tqdm(it, desc=desc, total=len(dataset)):
            if error is not None:
                print(f"error at {i}")
                breakpoint()

    if num_processes != 0:
        it = MultiprocessIterator(
            dataset,
            batchsize,
            repeat=False,
            shuffle=False,
            n_processes=num_processes,
            dataset_timeout=10,
        )
        for i, _ in tqdm(enumerate(it), desc=desc, total=len(dataset) // batchsize):
            pass


def check_dataset(config_yaml_path: Path, trials: int):
    with config_yaml_path.open() as f:
        config_dict = yaml.safe_load(f)

    config = Config.from_dict(config_dict)

    warnings.simplefilter("error", MultiprocessIterator.TimeoutWarning)

    num_processes = config.train.num_processes
    batchsize = config.train.batchsize

    # dataset
    datasets = create_dataset(config.dataset)

    for i in range(trials):
        print(f"try {i}")
        _check(
            datasets["train"],
            desc="train",
            num_processes=num_processes,
            batchsize=batchsize,
        )
        _check(
            datasets["test"],
            desc="test",
            num_processes=num_processes,
            batchsize=batchsize,
        )

        if datasets["eval"] is not None:
            _check(
                datasets["eval"],
                desc="eval",
                num_processes=num_processes,
                batchsize=batchsize,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_yaml_path", type=Path)
    parser.add_argument("--trials", type=int, default=10)
    check_dataset(**vars(parser.parse_args()))
