import torch
import tensorflow as tf
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers import AutoConfig

from deeplazy.core.pytorch_lazy_model_patcher import PytorchLazyModelPatcher
from deeplazy.core.tensorflow_lazy_model_patcher import TensorflowLazyModelPatcher
from deeplazy.enums.framework_enum import FrameworkType


class LazyModel:
    def __init__(self, cls=None, loader=None, is_custom=False, **kwargs):
        if loader is None:
            raise ValueError("Loader is required")

        self.loader = loader
        self.framework = loader.framework
        self.is_custom = is_custom
        self.model_args = kwargs

        if self.framework == FrameworkType.PYTORCH:
            self.model_instance = self._build_empty_pytorch_model(cls)
            patcher = PytorchLazyModelPatcher(
                self.loader, is_tied=hasattr(self.model_instance, '_tied_weights_keys'))

        elif self.framework == FrameworkType.TENSORFLOW:
            self.model_instance = self._build_empty_tensorflow_model(cls)
            patcher = TensorflowLazyModelPatcher(
                self.loader, is_tied=hasattr(self.model_instance, '_tied_weights_keys'))

        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

        self.model = patcher.patch(self.model_instance)

    # ========== PYTORCH ==========
    def _patch_torch_register_parameter(self):
        original = torch.nn.Module.register_parameter

        def fake_register(self, name, param):
            if param is not None:
                meta_tensor = torch.empty(
                    param.shape, dtype=param.dtype, device='meta', requires_grad=param.requires_grad)
                param = torch.nn.Parameter(
                    meta_tensor, requires_grad=param.requires_grad)
            return original(self, name, param)

        torch.nn.Module.register_parameter = fake_register
        return original

    def _build_empty_pytorch_model(self, cls):
        original_register_parameter = self._patch_torch_register_parameter()

        try:
            if self.is_custom:
                model = cls(**self.model_args)
            else:
                model = cls.from_pretrained(
                    self.loader.weights_dir, trust_remote_code=True)
        finally:
            torch.nn.Module.register_parameter = original_register_parameter

        return model

    # ========== TENSORFLOW ==========
    def _patch_tf_add_weight(self):
        original = tf.keras.layers.Layer.add_weight

        def fake_add_weight(self, name=None, shape=None, dtype=None, *args, **kwargs):
            dtype = dtype or tf.float32
            shape = shape or (1,)
            # Retorna variável com zeros para simular peso sem erro de dims
            return tf.Variable(tf.zeros(shape, dtype=dtype), trainable=False)

        tf.keras.layers.Layer.add_weight = fake_add_weight
        return original

    def _build_empty_tensorflow_model(self, cls):
        original_add_weight = self._patch_tf_add_weight()

        try:
            if self.is_custom:
                model = cls(**self.model_args)
            else:
                model = cls.from_pretrained(
                    self.loader.weights_dir, trust_remote_code=True)
        finally:
            tf.keras.layers.Layer.add_weight = original_add_weight

        return model

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def to(self, device):
        self.loader.device = device
        return self

    def generate(self, *args, **kwargs):
        if hasattr(self.model, "generate"):
            return self.model.generate(*args, **kwargs)
        raise AttributeError("Current model does not support `.generate()`")
