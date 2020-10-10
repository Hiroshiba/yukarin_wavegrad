from pathlib import Path


def get_data_directory() -> Path:
    return Path(__file__).parent.relative_to(Path.cwd()) / "data"
