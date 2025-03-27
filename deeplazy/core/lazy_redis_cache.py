# core/redis_layer_cache.py

import redis
import pickle
import io
import torch
from deeplazy.core.lazy_cache import BaseCacheBackend


class RedisCacheBackend(BaseCacheBackend):
    def __init__(self, redis_url="redis://localhost:6379/0", prefix="lazy_weights"):
        self.r = redis.Redis.from_url(redis_url)
        self.prefix = prefix

    def _key(self, name):
        return f"{self.prefix}:{name}"

    def get(self, key):
        raw = self.r.get(self._key(key))
        if raw:
            buffer = io.BytesIO(raw)
            return torch.load(buffer, map_location="cpu")
        return None

    def put(self, key, value):
        buffer = io.BytesIO()
        torch.save(value, buffer)
        self.r.set(self._key(key), buffer.getvalue())

    def pop(self, key):
        self.r.delete(self._key(key))

    def keys(self):
        return [k.decode() for k in self.r.keys(f"{self.prefix}:*")]
