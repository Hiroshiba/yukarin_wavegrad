from pathlib import Path

import pytest
import yaml
from yaml import SafeLoader

from tests.utility import get_data_directory
from yukarin_wavegrad.config import Config


@pytest.fixture()
def train_config_path():
    return get_data_directory() / "train_config.yaml"


def test_from_dict(train_config_path: Path):
    with train_config_path.open() as f:
        d = yaml.load(f, SafeLoader)
    Config.from_dict(d)


def test_to_dict(train_config_path: Path):
    with train_config_path.open() as f:
        d = yaml.load(f, SafeLoader)
    Config.from_dict(d).to_dict()


def test_equal_base_config_and_reconstructed(train_config_path: Path):
    with train_config_path.open() as f:
        d = yaml.load(f, SafeLoader)
    base = Config.from_dict(d)
    base_re = Config.from_dict(base.to_dict())
    assert base == base_re
