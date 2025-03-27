import torch.nn as nn
from functools import wraps

from deeplazy.core.lazy_model_patcher import LazyModelPatcher
import torch


class LazyModel(nn.Module):
    def __init__(self, config, cls, loader):
        super().__init__()
        self.loader = loader
        self.base_model = cls(config).to(self.loader.device)
        self.patcher = LazyModelPatcher(self.loader)
        self.model = self.patcher.patch(self.base_model)

    def forward(self, *args, **kwargs):
        with torch.inference_mode():
            return self.model(*args, **kwargs)

    def to(self, device):
        self.loader.device = device
        return self
