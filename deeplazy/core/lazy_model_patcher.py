import torch
import torch.nn as nn
from functools import wraps
import gc


class LazyModelPatcher:
    def __init__(self, loader):
        self.seen_classes = set()
        self.loader = loader
        self.modules_by_name = {}

    def patch(self, model):
        self._annotate_module_names(model)
        self._patch_module_classes()
        return model

    def _annotate_module_names(self, model):
        for name, module in model.named_modules():
            module._lazy_full_name = name
            self.modules_by_name[name] = module

    def _patch_module_classes(self):
        base_classes = (nn.Linear, nn.Conv1d, nn.LayerNorm, nn.Embedding)
        for cls in base_classes:
            self._patch_class(cls)

    def _patch_class(self, cls):
        if cls in self.seen_classes:
            return
        self.seen_classes.add(cls)
        orig_forward = cls.forward
        loader_ref = self.loader

        @wraps(orig_forward)
        def wrapped_forward(self, *args, **kwargs):
            full_name = getattr(self, '_lazy_full_name', '')
            backup = {}

            if loader_ref.cache.get(full_name) is None:
                loader_ref.load_module(full_name)

            module_weights = loader_ref.cache.get(full_name)
            if module_weights is not None:
                for param_name, tensor in module_weights.items():
                    parts = param_name.split('.')
                    obj = self
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
                    else:
                        print(f"Aviso: {name} não encontrado em {full_name}")

            result = orig_forward(self, *args, **kwargs)

            for (obj, name), original in backup.items():
                if hasattr(obj, name):
                    delattr(obj, name)
                obj.register_buffer(name, torch.empty(0, device="meta"))

            loader_ref.unload_module(full_name)
            return result

        cls.forward = wrapped_forward
