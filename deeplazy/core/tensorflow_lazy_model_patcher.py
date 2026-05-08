import threading
from functools import wraps


class TensorflowLazyModelPatcher:
    """Patches a meta-built TF/Keras model so each layer loads its
    weights on demand.

    Mirrors the PyTorch patcher: the LRU cache controls eviction, dtype
    is preserved, and a one-step lookahead prefetch hides I/O behind
    compute on subsequent forwards.
    """

    def __init__(self, loader, is_tied=False):
        self.loader = loader
        self.modules_by_name = {}
        self.base_model_prefix = None
        self.is_tied = is_tied
        self._next_module = {}
        self._last_seen_module = None
        self._trace_lock = threading.Lock()

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
        import tensorflow as tf
        for name, layer in self.modules_by_name.items():
            if hasattr(layer, '_lazy_wrapped'):
                continue
            if isinstance(layer, tf.keras.layers.Layer):
                self._wrap_layer_call(layer)

    def _record_trace(self, normalized_name):
        with self._trace_lock:
            prev = self._last_seen_module
            if prev is not None and prev != normalized_name:
                self._next_module[prev] = normalized_name
            self._last_seen_module = normalized_name
            return self._next_module.get(normalized_name)

    def _ensure_loaded(self, normalized_name, loader_ref):
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

    def _wrap_layer_call(self, layer):
        orig_call = layer.call
        loader_ref = self.loader
        layer._lazy_wrapped = True
        patcher = self

        @wraps(orig_call)
        def wrapped_call(*args, **kwargs):
            full_name = getattr(layer, '_lazy_full_name', '')
            normalized_name = getattr(
                layer, '_lazy_normalized_name', full_name)

            next_name = patcher._record_trace(normalized_name)

            patcher._ensure_loaded(normalized_name, loader_ref)

            if next_name is not None:
                loader_ref.prefetch_module(next_name, patcher.base_model_prefix)

            module_weights = loader_ref.cache.get(normalized_name)
            original_attrs = {}

            if module_weights is not None:
                for weight_name, weight_value in module_weights.items():
                    if hasattr(layer, weight_name):
                        original_attrs[weight_name] = getattr(
                            layer, weight_name)
                    setattr(layer, weight_name, weight_value)

            output = orig_call(*args, **kwargs)

            for name in original_attrs:
                setattr(layer, name, None)

            # Eviction is the LRU cache's job — do not call
            # loader.unload_module here.
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
