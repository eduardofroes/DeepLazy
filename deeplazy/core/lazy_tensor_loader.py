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

        # Busca formatos suportados no diretório
        self.weights_paths = [
            os.path.join(weights_dir, f)
            for f in os.listdir(weights_dir)
            if f.endswith('.safetensors')
        ]
        self.weights_format = 'safetensors'

        if not self.weights_paths:
            if self.framework == FrameworkType.PYTORCH:
                self.weights_paths = [
                    os.path.join(weights_dir, f)
                    for f in os.listdir(weights_dir)
                    if f.endswith('.pth')
                ]
                self.weights_format = 'pth'
            elif self.framework == FrameworkType.TENSORFLOW:
                self.weights_paths = [
                    os.path.join(weights_dir, f)
                    for f in os.listdir(weights_dir)
                    if f.endswith('.ckpt') or f.endswith('.h5')
                ]
                if self.weights_paths:
                    if self.weights_paths[0].endswith('.h5'):
                        self.weights_format = 'h5'
                    else:
                        self.weights_format = 'ckpt'

        if not self.weights_paths:
            raise FileNotFoundError(
                f"No supported weight files found in {weights_dir}")

        self.is_safetensors = self.weights_format == 'safetensors'
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
            if self.weights_format == 'safetensors':
                handler = safe_open(
                    path, framework=self.framework.value, device='cpu')
                self.file_handlers.append(handler)
                for key in handler.keys():
                    self.key_to_handler[key] = handler
            elif self.weights_format == 'pth':
                import torch
                state_dict = torch.load(path, map_location='cpu')
                self.file_handlers.append(state_dict)
                for key in state_dict.keys():
                    self.key_to_handler[key] = state_dict
            elif self.weights_format == 'ckpt':
                import tensorflow as tf
                reader = tf.train.load_checkpoint(path)
                self.file_handlers.append(reader)
                for key, _ in tf.train.list_variables(path):
                    self.key_to_handler[key] = reader
            elif self.weights_format == 'h5':
                import h5py
                f = h5py.File(path, 'r')
                self.file_handlers.append(f)
                def _collect(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        collected.append(name)
                collected = []
                f.visititems(_collect)
                for key in collected:
                    self.key_to_handler[key] = f

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
                    if self.weights_format in ('safetensors', 'ckpt'):
                        tensor = handler.get_tensor(key)
                    elif self.weights_format == 'pth':
                        tensor = handler[key]
                    elif self.weights_format == 'h5':
                        tensor = handler[key][()]
                    else:
                        continue
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
