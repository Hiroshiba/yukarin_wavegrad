from pytorch_trainer.dataset import convert
from pytorch_trainer.training.updaters.standard_updater import StandardUpdater

try:
    from torch.cuda import amp
except ImportError:
    pass


class AmpUpdater(StandardUpdater):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = amp.GradScaler()

    def update_core(self):
        iterator = self._iterators["main"]
        batch = iterator.next()
        in_arrays = convert._call_converter(self.converter, batch, self.device)

        optimizer = self._optimizers["main"]
        model = self._models["main"]
        loss_func = self.loss_func or model

        for model in self._models.values():
            model.train()
        optimizer.zero_grad()

        with amp.autocast():
            if isinstance(in_arrays, tuple):
                loss = loss_func(*in_arrays)
            elif isinstance(in_arrays, dict):
                loss = loss_func(**in_arrays)
            else:
                loss = loss_func(in_arrays)

        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["scaler"] = self.scaler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.scaler.load_state_dict(state_dict["scaler"])
