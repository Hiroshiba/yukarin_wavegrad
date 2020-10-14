from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from yukarin_wavegrad.utility import dataclass_utility
from yukarin_wavegrad.utility.git_utility import get_branch_name, get_commit_id


@dataclass
class DatasetConfig:
    sampling_rate: int
    sampling_length: int
    input_wave_glob: str
    input_silence_glob: str
    input_local_glob: str
    min_not_silence_length: int
    speaker_dict_path: Optional[str]
    speaker_size: Optional[int]
    seed: int
    num_train: Optional[int]
    num_test: int
    evaluate_times: int
    evaluate_time_second: float


@dataclass
class UpsamplingNetworkConfig:
    prev_hidden_size: int
    large_block_num: int
    hidden_sizes: List[int]


@dataclass
class DownsamplingNetworkConfig:
    prev_hidden_size: int
    hidden_sizes: List[int]


@dataclass
class NetworkConfig:
    local_size: int
    scales: List[int]
    speaker_size: int
    speaker_embedding_size: int
    upsampling: UpsamplingNetworkConfig
    downsampling: DownsamplingNetworkConfig


@dataclass
class NoiseScheduleModelConfig:
    start: float
    stop: float
    num: int


@dataclass
class ModelConfig:
    noise_schedule: NoiseScheduleModelConfig


@dataclass
class TrainConfig:
    batchsize: int
    log_iteration: int
    snapshot_iteration: int
    stop_iteration: int
    num_processes: Optional[int] = None
    optimizer: Dict[str, Any] = field(
        default_factory=dict(
            name="Adam",
        )
    )


@dataclass
class ProjectConfig:
    name: str
    tags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    dataset: DatasetConfig
    network: NetworkConfig
    model: ModelConfig
    train: TrainConfig
    project: ProjectConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        backward_compatible(d)
        return dataclass_utility.convert_from_dict(cls, d)

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_utility.convert_to_dict(self)

    def add_git_info(self):
        self.project.tags["git-commit-id"] = get_commit_id()
        self.project.tags["git-branch-name"] = get_branch_name()


def backward_compatible(d: Dict[str, Any]):
    pass