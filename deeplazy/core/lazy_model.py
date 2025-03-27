import torch.nn as nn
from functools import wraps

from deeplazy.core.pytorch_lazy_model_patcher import PytorchLazyModelPatcher
from deeplazy.core.tensorflow_lazy_model_patcher import TensorflowLazyModelPatcher

from deeplazy.enums.framework_enum import FrameworkType


class LazyModel:
    def __init__(self, config=None, cls=None, loader=None):
        self.loader = loader
        self.framework = loader.framework

        if config is not None:
            self.model_instance = cls(config)
        else:
            self.model_instance = cls()

        if self.framework == FrameworkType.PYTORCH:
            patcher = PytorchLazyModelPatcher(self.loader)
        elif self.framework == FrameworkType.TENSORFLOW:
            patcher = TensorflowLazyModelPatcher(self.loader)

        self.model = patcher.patch(self.model_instance)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def to(self, device):
        self.loader.device = device
        return self
