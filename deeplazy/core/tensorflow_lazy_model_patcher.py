import tensorflow as tf
import gc
from functools import wraps


class TensorflowLazyModelPatcher:
    def __init__(self, loader, is_tied=False):
        self.loader = loader
        self.modules_by_name = {}
        self.base_model_prefix = None
        self.is_tied = is_tied

    def patch(self, model):
        self.base_model_prefix = getattr(model, 'base_model_prefix', None)
        self._annotate_module_names(model)
        self._patch_module_instances()
        return model

    def _annotate_module_names(self, model):
        for layer in model.submodules:
            full_name = layer.name
            normalized = full_name
            if self.base_model_prefix and full_name.startswith(f"{self.base_model_prefix}/"):
                normalized = full_name[len(self.base_model_prefix) + 1:]
            layer._lazy_full_name = full_name
            layer._lazy_normalized_name = normalized
            self.modules_by_name[full_name] = layer

    def _patch_module_instances(self):
        for name, layer in self.modules_by_name.items():
            if hasattr(layer, '_lazy_wrapped'):
                continue
            if isinstance(layer, tf.keras.layers.Layer):
                self._wrap_instance_call(layer)

    def _wrap_instance_call(self, layer):
        orig_call = layer.call
        loader_ref = self.loader
        layer._lazy_wrapped = True

        @wraps(orig_call)
        def wrapped_call(*args, **kwargs):
            full_name = getattr(layer, '_lazy_full_name', '')
            normalized_name = getattr(
                layer, '_lazy_normalized_name', full_name)

            # Handle tied weights
            if normalized_name == "lm_head" and loader_ref.cache.get("lm_head") is None:
                if self.is_tied:
                    if loader_ref.cache.get("wte") is None:
                        loader_ref.load_module("wte")
                    loader_ref.cache.put(
                        "lm_head", loader_ref.cache.get("wte"))
                else:
                    loader_ref.load_module("lm_head")

            if loader_ref.cache.get(normalized_name) is None:
                loader_ref.load_module(normalized_name)

            module_weights = loader_ref.cache.get(normalized_name)
            original_attrs = {}

            if module_weights is not None:
                for name, tensor in module_weights.items():
                    if hasattr(layer, name):
                        original_attrs[name] = getattr(layer, name)
                    setattr(layer, name, tensor)

            output = orig_call(*args, **kwargs)

            for name, original in original_attrs.items():
                setattr(layer, name, original)

            loader_ref.unload_module(normalized_name)

            return output

        layer.call = wrapped_call
