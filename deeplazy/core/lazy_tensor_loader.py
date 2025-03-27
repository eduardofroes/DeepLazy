import torch
from safetensors import safe_open
import gc
from typing import Optional, Union
import os
import time
from deeplazy.core.lazy_redis_cache import RedisCacheBackend
from deeplazy.ui.dashboard_monitor import DashboardMonitor
from deeplazy.core.lazy_cache import LocalLRUCache


def print_memory(stage=""):
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2
    print(f"{stage}: {mem:.2f} MB")


class LazyLoader:
    def __init__(self, weights_path: Union[str, list], device='cpu', cache_backend=None, enable_monitor=False, model_name=None):

        if enable_monitor:

            capacity = 0
            cache_type = None
            if isinstance(cache_backend, LocalLRUCache):
                capacity = cache_backend.capacity
                cache_type = "LRU"
            elif isinstance(cache_backend, RedisCacheBackend):
                capacity = 0
                cache_type = "Redis"
            else:
                capacity = 0

            self.monitor = DashboardMonitor(
                model_name=model_name, safetensors_path=weights_path, framework="pytorch", cache_type=cache_type, max_layers_in_memory=capacity)
            self.monitor.enable()
        else:
            self.monitor = None

        if isinstance(weights_path, str):
            self.weights_paths = [weights_path]
        else:
            self.weights_paths = weights_path

        self.device = torch.device(device)
        self.cache = cache_backend
        self.file_handlers = []
        self.is_safetensors = all(path.endswith('.safetensors')
                                  for path in self.weights_paths)

    def _init_file_handlers(self):
        if not self.file_handlers:
            if self.is_safetensors:
                for path in self.weights_paths:
                    handler = safe_open(path, framework='pt', device='cpu')
                    self.file_handlers.append(handler)
            else:
                if len(self.weights_paths) > 1:
                    raise ValueError(
                        "Format not supported for multiple files that are not safetensors.")
                self.file_handlers = [
                    torch.load(
                        self.weights_paths[0], map_location=self.device, mmap=True)
                ]

    def load_module(self, module_name):
        self._init_file_handlers()

        cached = self.cache.get(module_name) if self.cache else None
        if cached is not None:
            return

        module_weights = {}
        start_time = time.time()

        if self.is_safetensors:
            for handler in self.file_handlers:
                for key in handler.keys():
                    if key.startswith(module_name + ".") and key[len(module_name) + 1:] not in module_weights:
                        tensor = handler.get_tensor(key).to(self.device)
                        module_weights[key[len(module_name)+1:]] = tensor
        else:
            handler = self.file_handlers[0]
            for key, tensor in handler.items():
                if key.startswith(module_name + "."):
                    module_weights[key[len(module_name)+1:]
                                   ] = tensor.to(self.device)

        if module_weights and self.cache:
            self.cache.put(module_name, module_weights)

        exec_time = time.time() - start_time
        if self.monitor:
            self.monitor.record_layer(module_name, exec_time)

    def unload_module(self, module_name):
        if self.cache:
            self.cache.pop(module_name)
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        self.file_handlers = []
