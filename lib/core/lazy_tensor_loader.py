import asyncio
import logging
import torch
import tensorflow as tf

logging.basicConfig(level=logging.INFO)


class LazyTensorLoader:
    def __init__(self, storage, framework='torch', cache=None):
        self.storage = storage
        self.framework = framework
        self.cache = cache

    def load(self, key):
        if self.cache:
            cached = self.cache.get_weight(key)
            if cached is not None:
                return cached

        tensor = self.storage.load_tensor(key)
        if tensor is None:
            raise ValueError(f"Tensor {key} not found.")

        if self.cache:
            self.cache.put_weight(key, tensor)

        return tensor

    async def load_tensor_async(self, key):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load, key)

    async def load_many_async(self, keys):
        tasks = [self.load_tensor_async(k) for k in keys]
        results = await asyncio.gather(*tasks)
        return dict(zip(keys, results))
