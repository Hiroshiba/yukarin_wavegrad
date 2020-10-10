from pathlib import Path

from pytorch_trainer.training import Extension, Trainer
from tensorboardX import SummaryWriter


class TensorboardReport(Extension):
    def __init__(self, writer: SummaryWriter = None):
        self.writer = writer

    def __call__(self, trainer: Trainer):
        if self.writer is None:
            self.writer = SummaryWriter(Path(trainer.out))

        observations = trainer.observation
        n_iter = trainer.updater.iteration
        for n, v in observations.items():
            self.writer.add_scalar(n, v, n_iter)

    def finalize(self):
        super().finalize()
        self.writer.flush()
