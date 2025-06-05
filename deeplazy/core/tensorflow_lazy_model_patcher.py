import tensorflow as tf
from functools import wraps


class TensorflowLazyModelPatcher:
    def __init__(self, loader, is_tied=False):
        self.loader = loader
        self.modules_by_name = {}
        self.base_model_prefix = None
        self.is_tied = is_tied

    def patch(self, model):
        self.base_model_prefix = getattr(model, 'base_model_prefix', None)
        self._annotate_layer_names(model)
        self._patch_layers()
        self._detect_and_cache_lm_head(model)
        return model

    def _annotate_layer_names(self, model):
        for layer in model.submodules:
            full_name = layer.name
            normalized = full_name
            if self.base_model_prefix and full_name.startswith(f"{self.base_model_prefix}/"):
                normalized = full_name[len(self.base_model_prefix) + 1:]
            layer._lazy_full_name = full_name
            layer._lazy_normalized_name = normalized
            self.modules_by_name[full_name] = layer

    def _patch_layers(self):
        for name, layer in self.modules_by_name.items():
            if hasattr(layer, '_lazy_wrapped'):
                continue
            if isinstance(layer, tf.keras.layers.Layer):
                self._wrap_layer_call(layer)

    def _wrap_layer_call(self, layer):
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
                        loader_ref.load_module("wte", self.base_model_prefix)
                    loader_ref.cache.put(
                        "lm_head", loader_ref.cache.get("wte"))
                else:
                    loader_ref.load_module("lm_head", self.base_model_prefix)

            if loader_ref.cache.get(normalized_name) is None:
                loader_ref.load_module(normalized_name, self.base_model_prefix)

            module_weights = loader_ref.cache.get(normalized_name)
            original_attrs = {}

            if module_weights is not None:
                for weight_name, weight_value in module_weights.items():
                    if hasattr(layer, weight_name):
                        original_attrs[weight_name] = getattr(
                            layer, weight_name)
                    tensor_on_device = tf.convert_to_tensor(weight_value)
                    setattr(layer, weight_name, tensor_on_device)

            output = orig_call(*args, **kwargs)

            # Libera memória dos pesos
            for name in original_attrs:
                setattr(layer, name, None)

            loader_ref.unload_module(normalized_name)

            return output

        layer.call = wrapped_call

    def _detect_and_cache_lm_head(self, model):
        try:
            output_tensor = model.outputs[0]
            for layer in reversed(model.submodules):
                try:
                    if output_tensor in layer.output if isinstance(layer.output, list) else [layer.output]:
                        name = getattr(
                            layer, "_lazy_normalized_name", layer.name)
                        if self.loader.cache.get(name):
                            print(f"[lazy] detected lm_head candidate: {name}")
                            self.loader.cache.put(
                                "lm_head", self.loader.cache.get(name))
                        break
                except Exception:
                    continue
        except Exception:
            pass
