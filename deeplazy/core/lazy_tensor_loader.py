
from deeplazy.enums.framework_enum import FrameworkType
from typing import Union
import time
import gc
from safetensors.torch import safe_open


class LazyLoader:
    def __init__(self, weights_path: Union[str, list], device='cpu', cache_backend=None,
                 enable_monitor=False, model_name=None, framework=FrameworkType.PYTORCH):

        self.framework = framework
        self.device = device
        self.cache = cache_backend
        self.monitor = None

        if isinstance(weights_path, str):
            self.weights_paths = [weights_path]
        else:
            self.weights_paths = weights_path

        self.is_safetensors = all(path.endswith('.safetensors')
                                  for path in self.weights_paths)
        self.file_handlers = []

        if enable_monitor:
            from deeplazy.ui.dashboard_monitor import DashboardMonitor
            capacity = getattr(cache_backend, 'capacity', 0)
            cache_type = cache_backend.__class__.__name__ if cache_backend else None
            self.monitor = DashboardMonitor(
                model_name=model_name,
                safetensors_path=weights_path,
                framework=framework.value,
                cache_type=cache_type,
                max_layers_in_memory=capacity
            )
            self.monitor.enable()

        if self.framework == FrameworkType.PYTORCH:
            import torch
            self.device = torch.device(device)
        elif self.framework == FrameworkType.TENSORFLOW:
            import tensorflow as tf
            self.device = device

    def _init_file_handlers(self):
        if self.file_handlers:
            return

        if self.is_safetensors:
            from safetensors import safe_open
            for path in self.weights_paths:
                handler = safe_open(
                    path, framework=self.framework.value, device='cpu')
                self.file_handlers.append(handler)
        else:
            if len(self.weights_paths) > 1:
                raise ValueError(
                    "Only single-file loading supported for non-safetensors format.")
            if self.framework == FrameworkType.PYTORCH:
                import torch
                self.file_handlers = [torch.load(
                    self.weights_paths[0], map_location=self.device, mmap=True)]
            elif self.framework == FrameworkType.TENSORFLOW:
                import tensorflow as tf
                self.file_handlers = [
                    tf.saved_model.load(self.weights_paths[0])]

    def load_module(self, module_name):
        self._init_file_handlers()
        cached = self.cache.get(module_name) if self.cache else None
        if cached:
            return

        module_weights = {}
        import time
        start_time = time.time()

        if self.is_safetensors:
            for handler in self.file_handlers:
                for key in handler.keys():
                    if key.startswith(module_name + ".") and key[len(module_name) + 1:] not in module_weights:
                        tensor = handler.get_tensor(key)
                        if self.framework == FrameworkType.PYTORCH:
                            tensor = tensor.to(self.device)
                        elif self.framework == FrameworkType.TENSORFLOW:
                            import tensorflow as tf
                            tensor = tf.convert_to_tensor(tensor.numpy())
                        module_weights[key[len(module_name)+1:]] = tensor
        else:
            handler = self.file_handlers[0]
            if self.framework == FrameworkType.PYTORCH:
                for key, tensor in handler.items():
                    if key.startswith(module_name + "."):
                        module_weights[key[len(module_name)+1:]
                                       ] = tensor.to(self.device)
            elif self.framework == FrameworkType.TENSORFLOW:
                raise NotImplementedError(
                    "Non-safetensors TensorFlow loading not supported yet")

        if module_weights and self.cache:
            self.cache.put(module_name, module_weights)

        if self.monitor:
            exec_time = time.time() - start_time
            self.monitor.record_layer(module_name, exec_time)

    def unload_module(self, module_name):
        if self.cache:
            self.cache.pop(module_name)
            gc.collect()
            if self.framework == FrameworkType.PYTORCH:
                import torch
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
        self.file_handlers = []
