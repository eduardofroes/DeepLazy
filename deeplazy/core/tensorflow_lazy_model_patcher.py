import tensorflow as tf
import gc
from functools import wraps


class TensorflowLazyModelPatcher:
    def __init__(self, loader):
        self.loader = loader
        self.modules_by_name = {}

    def patch(self, model):
        self._annotate_module_names(model)
        self._patch_module_instances()
        return model

    def _annotate_module_names(self, model):
        for layer in model.submodules:
            layer._lazy_full_name = layer.name
            self.modules_by_name[layer.name] = layer

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
            backup = {}

            if loader_ref.cache.get(full_name) is None:
                loader_ref.load_module(full_name)

            module_weights = loader_ref.cache.get(full_name)
            if module_weights is not None:
                for name, tensor in module_weights.items():
                    if hasattr(layer, name):
                        original = getattr(layer, name)
                        backup[name] = original
                        setattr(layer, name, tensor)

            output = orig_call(*args, **kwargs)

            for name, original in backup.items():
                setattr(layer, name, tf.constant(
                    0.0, shape=original.shape, dtype=original.dtype))

            loader_ref.unload_module(full_name)
            return output

        layer.call = wrapped_call
