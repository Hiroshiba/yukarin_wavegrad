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
    input_local_glob: Optional[str]
    mulaw: bool
    local_padding_length: int
    min_not_silence_length: int
    speaker_dict_path: Optional[str]
    speaker_size: Optional[int]
    seed: int
    num_train: Optional[int]
    num_test: int
    evaluate_times: int
    evaluate_time_second: float
    evaluate_local_padding_time_second: float


@dataclass
class EncodingNetworkConfig:
    hidden_size: int
    layer_num: int


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
    encoding: EncodingNetworkConfig
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
    eval_batchsize: Optional[int]
    log_iteration: int
    eval_iteration: int
    stop_iteration: int
    optimizer: Dict[str, Any]
    multistep_shift: Optional[Dict[str, Any]] = None
    num_processes: Optional[int] = None
    use_multithread: bool = False
    use_amp: Optional[bool] = None


@dataclass
class ProjectConfig:
    name: str
    tags: Dict[str, Any] = field(default_factory=dict)
    category: Optional[str] = None


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
    if "eval_iteration" not in d["train"]:
        d["train"]["eval_iteration"] = d["train"]["snapshot_iteration"]

    if "encoding" not in d["network"]:
        d["network"]["encoding"] = {"hidden_size": 0, "layer_num": 0}

    if "local_padding_length" not in d["dataset"]:
        d["dataset"]["local_padding_length"] = 0

    if "evaluate_local_padding_time_second" not in d["dataset"]:
        d["dataset"]["evaluate_local_padding_time_second"] = 0

    if "snapshot_iteration" in d["train"]:
        d["train"].pop("snapshot_iteration")

    if "mulaw" not in d["dataset"]:
        d["dataset"]["mulaw"] = False
