import torch
import tensorflow as tf
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from deeplazy.core.pytorch_lazy_model_patcher import PytorchLazyModelPatcher
from deeplazy.core.tensorflow_lazy_model_patcher import TensorflowLazyModelPatcher
from deeplazy.enums.framework_enum import FrameworkType


class LazyModel:
    def __init__(self, cls=None, loader=None):
        if loader is None:
            raise ValueError("Loader is required")

        self.loader = loader
        self.framework = loader.framework

        if self.framework == FrameworkType.PYTORCH:
            self.model_instance = self._build_empty_pytorch_model(cls)
            patcher = PytorchLazyModelPatcher(self.loader, is_tied=hasattr(
                self.model_instance, '_tied_weights_keys'))

        elif self.framework == FrameworkType.TENSORFLOW:
            self.model_instance = self._build_empty_tensorflow_model(cls)
            patcher = TensorflowLazyModelPatcher(self.loader, is_tied=hasattr(
                self.model_instance, '_tied_weights_keys'))

        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

        self.model = patcher.patch(self.model_instance)

    def _build_empty_pytorch_model(self, cls):
        model = cls.from_pretrained(
            pretrained_model_name_or_path=self.loader.weights_dir,
            device_map={"": "meta"},
            low_cpu_mem_usage=True
        )
        return model

    def _build_empty_tensorflow_model(self, cls):
        return cls.from_pretrained(self.loader.weights_dir)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def to(self, device):
        self.loader.device = device
        return self

    def generate(self, *args, **kwargs):
        if hasattr(self.model, "generate"):
            return self.model.generate(*args, **kwargs)
        raise AttributeError("O modelo atual não suporta `.generate()`")
