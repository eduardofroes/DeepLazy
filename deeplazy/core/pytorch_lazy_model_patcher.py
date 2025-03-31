import torch
import torch.nn as nn
from functools import wraps
import asyncio
from concurrent.futures import ThreadPoolExecutor
import gc
import threading


class PytorchLazyModelPatcher:
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
        for name, module in model.named_modules():
            module._lazy_full_name = name
            normalized = name
            if self.base_model_prefix and name.startswith(f"{self.base_model_prefix}."):
                normalized = name[len(self.base_model_prefix) + 1:]
            module._lazy_normalized_name = normalized
            self.modules_by_name[name] = module

    def _patch_module_instances(self):
        for name, module in self.modules_by_name.items():
            if hasattr(module, '_lazy_wrapped'):
                continue
            if isinstance(module, nn.Module) and not isinstance(module, nn.Sequential):
                self._wrap_instance_forward(module)

    def _wrap_instance_forward(self, module):
        orig_forward = module.forward
        loader_ref = self.loader
        module._lazy_wrapped = True

        @wraps(orig_forward)
        def wrapped_forward(*args, **kwargs):
            full_name = getattr(module, '_lazy_full_name', '')
            normalized_name = getattr(
                module, '_lazy_normalized_name', full_name)

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
            original_attrs = []

            module.to_empty(device=loader_ref.device)

            if module_weights is not None:
                for param_name, tensor in module_weights.items():
                    name_parts = param_name.split('.')
                    target = module
                    name = name_parts[-1]

                    if hasattr(target, name):
                        original_attrs.append(name)
                        for attr_dict in [target._parameters, target._buffers, target.__dict__]:
                            if name in attr_dict:
                                attr_dict.pop(name, None)
                        try:
                            delattr(target, name)
                        except Exception:
                            pass

                    tensor = tensor.to(dtype=torch.float32,
                                       device=loader_ref.device)
                    target.register_buffer(name, tensor)

                if hasattr(module, 'masked_bias') and hasattr(module, 'bias'):
                    module.masked_bias = module.bias

            result = orig_forward(*args, **kwargs)

            for name in original_attrs:
                delattr(module, name)
                setattr(module, name, None)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            loader_ref.unload_module(normalized_name)

            return result

        if hasattr(module, 'weight') or hasattr(module, 'bias') or hasattr(module, 'masked_bias'):
            module.forward = wrapped_forward
