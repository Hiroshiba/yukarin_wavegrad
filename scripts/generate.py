import argparse
import re
from glob import glob
from pathlib import Path
from typing import Optional

import numpy
import yaml
from acoustic_feature_extractor.data.sampling_data import SamplingData
from more_itertools import chunked
from pytorch_trainer.dataset.convert import concat_examples
from tqdm import tqdm
from utility.save_arguments import save_arguments
from yukarin_wavegrad.config import Config, NoiseScheduleModelConfig
from yukarin_wavegrad.dataset import SpeakerWavesDataset, WavesDataset, create_dataset
from yukarin_wavegrad.generator import Generator


def _extract_number(f):
    s = re.findall(r"\d+", str(f))
    return int(s[-1]) if s else -1


def _get_predictor_model_path(
    model_dir: Path,
    iteration: int = None,
    prefix: str = "predictor_",
):
    if iteration is None:
        paths = model_dir.glob(prefix + "*.pth")
        model_path = list(sorted(paths, key=_extract_number))[-1]
    else:
        model_path = model_dir / (prefix + "{}.pth".format(iteration))
        assert model_path.exists()
    return model_path


def generate(
    model_dir: Path,
    model_iteration: Optional[int],
    model_config: Optional[Path],
    output_dir: Path,
    batch_size: int,
    num_test: int,
    from_train_data: bool,
    time_second: float,
    val_local_glob: str,
    val_speaker_id: Optional[int],
    noise_schedule_start: float,
    noise_schedule_stop: float,
    noise_schedule_num: int,
    use_gpu: bool,
):
    output_dir.mkdir(exist_ok=True)
    save_arguments(output_dir / "arguments.yaml", generate, locals())

    if model_config is None:
        model_config = model_dir / "config.yaml"
    config = Config.from_dict(yaml.safe_load(model_config.open()))

    model_path = _get_predictor_model_path(
        model_dir=model_dir,
        iteration=model_iteration,
    )
    print("model path: ", model_path)
    generator = Generator(
        config=config,
        noise_schedule_config=NoiseScheduleModelConfig(
            start=noise_schedule_start, stop=noise_schedule_stop, num=noise_schedule_num
        ),
        predictor=model_path,
        sampling_rate=config.dataset.sampling_rate,
        use_gpu=use_gpu,
    )

    local_padding_second = 1
    local_padding_length = config.dataset.sampling_rate * local_padding_second

    config.dataset.sampling_length = int(config.dataset.sampling_rate * time_second)
    config.dataset.local_padding_length = local_padding_length
    dataset = create_dataset(config.dataset)["test" if not from_train_data else "train"]

    if isinstance(dataset, SpeakerWavesDataset):
        wave_paths = [
            input.path_wave for input in dataset.wave_dataset.inputs[:num_test]
        ]
    elif isinstance(dataset, WavesDataset):
        wave_paths = [input.path_wave for input in dataset.inputs[:num_test]]
    else:
        raise Exception()

    for data, wave_path in tqdm(
        zip(chunked(dataset, batch_size), chunked(wave_paths, batch_size)),
        desc="generate",
    ):
        data = concat_examples(data)
        output = generator.generate(
            local=data["local"],
            local_padding_length=local_padding_length,
            speaker_id=data["speaker_id"] if "speaker_id" in data else None,
        )

        for wave, p in zip(output, wave_path):
            wave.save(output_dir / (p.stem + ".wav"))

    # validation
    if val_local_glob is not None:
        local_paths = sorted([Path(p) for p in glob(val_local_glob)])
        speaker_ids = [val_speaker_id] * len(local_paths)
        for local_path, speaker_id in zip(
            chunked(local_paths, batch_size), chunked(speaker_ids, batch_size)
        ):
            datas = [SamplingData.load(p) for p in local_path]
            size = int((time_second + local_padding_second * 2) * datas[0].rate)
            local = numpy.stack(
                [
                    (
                        data.array[:size].T
                        if len(data.array) >= size
                        else numpy.pad(
                            data.array,
                            ((0, size - len(data.array)), (0, 0)),
                            mode="edge",
                        ).T
                    )
                    for data in datas
                ]
            )

            output = generator.generate(
                local=local,
                local_padding_length=local_padding_length,
                speaker_id=(
                    numpy.stack(speaker_id) if speaker_id[0] is not None else None
                ),
            )

            for wave, p in zip(output, local_path):
                wave.save(output_dir / (p.stem + ".wav"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=Path)
    parser.add_argument("--model_iteration", type=int)
    parser.add_argument("--model_config", type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_test", type=int, default=10)
    parser.add_argument("--from_train_data", action="store_true")
    parser.add_argument("--time_second", type=float, default=1)
    parser.add_argument("--val_local_glob")
    parser.add_argument("--val_speaker_id", type=int)
    parser.add_argument("--noise_schedule_start", type=float, default=1e-4)
    parser.add_argument("--noise_schedule_stop", type=float, default=0.05)
    parser.add_argument("--noise_schedule_num", type=int, default=50)
    parser.add_argument("--use_gpu", action="store_true")
    generate(**vars(parser.parse_args()))
