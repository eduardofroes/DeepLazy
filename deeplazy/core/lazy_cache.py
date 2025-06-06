import redis
import io
import torch
import tensorflow as tf
from collections import OrderedDict
import gc
import pickle
from deeplazy.enums.framework_enum import FrameworkType


class BaseCacheBackend:
    def get(self, key):
        raise NotImplementedError

    def put(self, key, value):
        raise NotImplementedError

    def pop(self, key):
        raise NotImplementedError

    def keys(self):
        raise NotImplementedError


class PytorchLocalLRUCache(BaseCacheBackend):
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def pop(self, key):
        return self.cache.pop(key, None)

    def keys(self):
        return list(self.cache.keys())


class TFLRULazyCache(BaseCacheBackend):
    def __init__(self, capacity=4):
        self.capacity = capacity
        self.cache = {}
        self.access_order = []

    def get(self, key):
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest_key = self.access_order.pop(0)
            old_val = self.cache.pop(oldest_key)
            del old_val
            try:
                tf.keras.backend.clear_session()
                if hasattr(tf.config.experimental, 'clear_memory'):
                    tf.config.experimental.clear_memory()
            except Exception:
                pass
            gc.collect()

        self.cache[key] = value
        self.access_order.append(key)

    def pop(self, key):
        if key in self.cache:
            self.access_order.remove(key)
            val = self.cache.pop(key)
            del val
            try:
                tf.keras.backend.clear_session()
                if hasattr(tf.config.experimental, 'clear_memory'):
                    tf.config.experimental.clear_memory()
            except Exception:
                pass
            gc.collect()
            return val
        return None

    def keys(self):
        return list(self.cache.keys())


class RedisCacheBackend(BaseCacheBackend):
    def __init__(self, redis_url="redis://localhost:6379/0", prefix="lazy_weights", framework=FrameworkType.PYTORCH):
        self.r = redis.Redis.from_url(redis_url)
        self.prefix = prefix
        self.framework = framework

    def _key(self, name):
        return f"{self.prefix}:{name}"

    def get(self, key):
        raw = self.r.get(self._key(key))
        if raw:
            buffer = io.BytesIO(raw)
            if self.framework == FrameworkType.PYTORCH:
                return torch.load(buffer, map_location="cpu")
            elif self.framework == FrameworkType.TENSORFLOW:
                return pickle.load(buffer)
        return None

    def put(self, key, value):
        buffer = io.BytesIO()
        if self.framework == FrameworkType.PYTORCH:
            torch.save(value, buffer)
        elif self.framework == FrameworkType.TENSORFLOW:
            pickle.dump(value, buffer)
        self.r.set(self._key(key), buffer.getvalue())

    def pop(self, key):
        self.r.delete(self._key(key))

    def keys(self):
        return [k.decode() for k in self.r.keys(f"{self.prefix}:*")]
