import torch
import torch.nn as nn
from functools import wraps
import gc


class PytorchLazyModelPatcher:
    def __init__(self, loader):
        self.loader = loader
        self.modules_by_name = {}

    def patch(self, model):
        self._annotate_module_names(model)
        self._patch_module_instances()
        return model

    def _annotate_module_names(self, model):
        for name, module in model.named_modules():
            module._lazy_full_name = name
            self.modules_by_name[name] = module

    def _patch_module_instances(self):
        for name, module in self.modules_by_name.items():
            if hasattr(module, '_lazy_wrapped'):
                continue  # evita duplicar patch

            if isinstance(module, nn.Module) and not isinstance(module, nn.Sequential):
                self._wrap_instance_forward(module)

    def _wrap_instance_forward(self, module):
        orig_forward = module.forward
        loader_ref = self.loader
        module._lazy_wrapped = True  # marca como patchado

        @wraps(orig_forward)
        def wrapped_forward(*args, **kwargs):
            full_name = getattr(module, '_lazy_full_name', '')
            backup = {}

            if loader_ref.cache.get(full_name) is None:
                loader_ref.load_module(full_name)

            module_weights = loader_ref.cache.get(full_name)
            if module_weights is not None:
                for param_name, tensor in module_weights.items():
                    parts = param_name.split('.')
                    obj = module
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    name = parts[-1]

                    if hasattr(obj, name):
                        orig_attr = getattr(obj, name)
                        backup[(obj, name)] = orig_attr
                        delattr(obj, name)
                        tensor = tensor.to(
                            device=loader_ref.device, dtype=orig_attr.dtype)
                        obj.register_buffer(name, tensor)

            result = orig_forward(*args, **kwargs)

            for (obj, name), original in backup.items():
                if hasattr(obj, name):
                    delattr(obj, name)
                obj.register_buffer(name, torch.empty(0, device="meta"))

            loader_ref.unload_module(full_name)
            return result

        module.forward = wrapped_forward
