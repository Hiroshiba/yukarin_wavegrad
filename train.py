import argparse
from pathlib import Path

import yaml

from yukarin_wavegrad.trainer import create_trainer


def train(
    config_yaml_path: Path,
    output: Path,
):
    with config_yaml_path.open() as f:
        d = yaml.safe_load(f)

    trainer = create_trainer(config_dict=d, output=output)
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_yaml_path", type=Path)
    parser.add_argument("output", type=Path)
    train(**vars(parser.parse_args()))
