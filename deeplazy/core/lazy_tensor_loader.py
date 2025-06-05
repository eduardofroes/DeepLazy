import os
import time
import gc
from typing import Union
from safetensors import safe_open
from deeplazy.enums.framework_enum import FrameworkType


class LazyLoader:
    def __init__(self, weights_dir: str, device='cpu', cache_backend=None,
                 enable_monitor=False, model_name=None, framework=FrameworkType.PYTORCH):

        self.framework = framework
        self.device = device
        self.cache = cache_backend
        self.monitor = None
        self.weights_dir = weights_dir

        # Busca todos os arquivos .safetensors no diretório
        self.weights_paths = [
            os.path.join(weights_dir, f)
            for f in os.listdir(weights_dir)
            if f.endswith('.safetensors')
        ]

        if not self.weights_paths:
            raise FileNotFoundError(
                f"No files .safetensors found in {weights_dir}")

        self.is_safetensors = True
        self.file_handlers = []
        self.key_to_handler = {}

        if enable_monitor:
            from deeplazy.ui.dashboard_monitor import DashboardMonitor
            capacity = getattr(cache_backend, 'capacity', 0)
            cache_type = cache_backend.__class__.__name__ if cache_backend else None
            self.monitor = DashboardMonitor(
                model_name=model_name,
                safetensors_path=self.weights_paths,
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

        for path in self.weights_paths:
            handler = safe_open(
                path, framework=self.framework.value, device='cpu')
            self.file_handlers.append(handler)
            for key in handler.keys():
                self.key_to_handler[key] = handler

    def load_module(self, module_name, base_model_prefix=None):
        self._init_file_handlers()
        if self.cache and (cached := self.cache.get(module_name)):
            return

        module_weights = {}
        start_time = time.time()

        # Corrigido para evitar erro de NoneType + str
        prefix = (base_model_prefix + ".") if base_model_prefix else ""

        for key, handler in self.key_to_handler.items():
            if key.replace(prefix, "").startswith(module_name + "."):
                short_key = key[len(module_name):]
                if short_key not in module_weights:
                    tensor = handler.get_tensor(key)
                    if self.framework == FrameworkType.TENSORFLOW:
                        import tensorflow as tf
                        tensor = tf.convert_to_tensor(tensor)
                    module_weights[short_key] = tensor

        if module_weights and self.cache:
            self.cache.put(module_name, module_weights)

        if self.monitor:
            exec_time = time.time() - start_time
            self.monitor.record_layer(module_name, exec_time)

    def unload_module(self, module_name):
        if self.cache:
            self.cache.pop(module_name)
            if self.framework == FrameworkType.PYTORCH:
                import torch
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            elif self.framework == FrameworkType.TENSORFLOW:
                import tensorflow as tf
                try:
                    tf.keras.backend.clear_session()
                except Exception:
                    pass
                try:
                    if hasattr(tf.config.experimental, 'clear_memory'):
                        tf.config.experimental.clear_memory()
                except Exception:
                    pass
                gc.collect()

        self.file_handlers = []
        self.key_to_handler = {}
