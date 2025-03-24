import gc
import torch
import psutil
import os
import asyncio
import time


def print_memory(stage=""):
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    print(f"[MEMORY] {stage}: {mem:.2f} MB")


class LazyModel:
    def __init__(self, layers: dict, metadata: dict = None, max_layers_in_memory=10, dashboard=None):
        """
        :param layers: Dict of LazyLayer instances.
        :param metadata: Model metadata (e.g., hidden_size, etc.).
        :param max_layers_in_memory: Number of layers to load in memory at once.
        :param dashboard: Optional DashboardMonitor instance.
        """
        self.layers = layers
        self.layer_sequence = list(layers.keys())
        self.metadata = metadata or {}
        self.max_layers_in_memory = max_layers_in_memory
        self.dashboard = dashboard

    async def forward(self, x, enable_dashboard=False):
        total_layers = len(self.layer_sequence)

        if enable_dashboard:
            self.dashboard.enable()

        for i in range(0, total_layers, self.max_layers_in_memory):
            block_names = self.layer_sequence[i:i + self.max_layers_in_memory]

            all_keys = []
            for name in block_names:
                all_keys.extend(self.layers[name].keys)

            all_weights = await self._load_weights_batch(all_keys)
            await self._build_all_layers_parallel(block_names, all_weights)

            for name in block_names:
                layer = self.layers[name]

                if not layer.is_built:
                    raise RuntimeError(
                        f"Layer {name} not built before forward.")

                start_time = time.time()
                x = layer(x)
                exec_time = time.time() - start_time

                if enable_dashboard:
                    self.dashboard.record_layer(name, exec_time)

                layer.unload()

            gc.collect()
            torch.cuda.empty_cache()

        if enable_dashboard:
            self.dashboard.print_footer()

        return x

    async def _load_weights_batch(self, all_keys):
        loader = self.layers[self.layer_sequence[0]].tensor_loader
        return await loader.load_many_async(all_keys)

    async def _build_all_layers_parallel(self, block_names, all_weights):
        tasks = []
        for name in block_names:
            layer = self.layers[name]
            weights = {k: all_weights[k] for k in layer.keys}
            tasks.append(layer.async_build_layer_from_weights(weights))
        await asyncio.gather(*tasks)
