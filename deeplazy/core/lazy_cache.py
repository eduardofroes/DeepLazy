import redis
import io
import torch
from collections import OrderedDict
import gc


class BaseCacheBackend:
    def get(self, key):
        raise NotImplementedError

    def put(self, key, value):
        raise NotImplementedError

    def pop(self, key):
        raise NotImplementedError

    def keys(self):
        raise NotImplementedError


class LocalLRUCache(BaseCacheBackend):
    def __init__(self, capacity=4):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            old_key, old_val = self.cache.popitem(last=False)
            del old_val
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def pop(self, key):
        return self.cache.pop(key, None)

    def keys(self):
        return list(self.cache.keys())
