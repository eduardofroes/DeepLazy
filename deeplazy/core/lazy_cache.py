import io
import gc
import pickle
from collections import OrderedDict

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
    """O(1) LRU cache for PyTorch lazy weights.

    Avoids per-put ``torch.cuda.empty_cache()`` calls — those cost a
    full driver synchronisation. We let CUDA's caching allocator reuse
    freed blocks naturally, which is both correct and dramatically
    faster on hot inference loops.

    ``on_evict``, when set, is called with the evicted module name
    immediately after the entry leaves the cache. The loader uses this
    to issue ``madvise(MADV_DONTNEED)`` on the backing mmap pages,
    which releases them from process RSS while keeping them in the
    kernel page cache for fast re-access.
    """

    def __init__(self, capacity=4, on_evict=None):
        self.capacity = capacity
        self.cache: "OrderedDict[str, dict]" = OrderedDict()
        self.on_evict = on_evict

    def get(self, key):
        value = self.cache.get(key)
        if value is not None:
            self.cache.move_to_end(key)
        return value

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            evicted_key, evicted_value = self.cache.popitem(last=False)
            del evicted_value
            if self.on_evict is not None:
                try:
                    self.on_evict(evicted_key)
                except Exception:
                    pass

    def pop(self, key):
        return self.cache.pop(key, None)

    def keys(self):
        return list(self.cache.keys())


class ByteBudgetLRUCache(PytorchLocalLRUCache):
    """LRU cache bounded by a memory budget in megabytes, not by layer count.

    Motivation: layer sizes in a transformer are not uniform.  An embedding
    table or ``lm_head`` can be 10–100× larger than a LayerNorm.  A
    layer-count LRU wastes slots on tiny layers and underestimates how much
    RAM a slot of large layers costs.  A byte budget is exact.

    Evicts LRU entries until the total in-cache bytes ≤ ``budget_mb * 2²⁰``.
    The ``on_evict`` callback (used for ``madvise(MADV_DONTNEED)``) fires
    for every eviction, exactly like the parent class.
    """

    def __init__(self, budget_mb: float, on_evict=None):
        # capacity=inf so the parent's len-check never fires;
        # eviction is driven entirely by _current_bytes vs budget.
        super().__init__(capacity=2**31, on_evict=on_evict)
        self.budget_bytes: int = int(budget_mb * 1024 * 1024)
        self._current_bytes: int = 0

    @staticmethod
    def _sizeof(weights_dict) -> int:
        if not isinstance(weights_dict, dict):
            return 0
        return sum(
            t.nbytes for t in weights_dict.values() if hasattr(t, 'nbytes'))

    def put(self, key, value):
        if key in self.cache:
            self._current_bytes -= self._sizeof(self.cache[key])
            self.cache.move_to_end(key)
        self.cache[key] = value
        self._current_bytes += self._sizeof(value)

        while self._current_bytes > self.budget_bytes and len(self.cache) > 1:
            evicted_key, evicted_value = self.cache.popitem(last=False)
            self._current_bytes -= self._sizeof(evicted_value)
            del evicted_value
            if self.on_evict is not None:
                try:
                    self.on_evict(evicted_key)
                except Exception:
                    pass

    def pop(self, key):
        value = self.cache.pop(key, None)
        if value is not None:
            self._current_bytes -= self._sizeof(value)
        return value

    @property
    def used_mb(self) -> float:
        return self._current_bytes / (1024 * 1024)

    @property
    def budget_mb(self) -> float:
        return self.budget_bytes / (1024 * 1024)


class TFLRULazyCache(BaseCacheBackend):
    """O(1) LRU cache for TensorFlow lazy weights.

    The previous implementation called ``tf.keras.backend.clear_session``
    on every eviction, which destroys the entire global graph state —
    extremely expensive and routinely incorrect when other layers were
    still mid-build. We now drop the reference and let the Python GC do
    its job.
    """

    def __init__(self, capacity=4):
        self.capacity = capacity
        self.cache: "OrderedDict[str, dict]" = OrderedDict()

    def get(self, key):
        value = self.cache.get(key)
        if value is not None:
            self.cache.move_to_end(key)
        return value

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def pop(self, key):
        return self.cache.pop(key, None)

    def keys(self):
        return list(self.cache.keys())


class RedisCacheBackend(BaseCacheBackend):
    """Redis-backed cache.

    For PyTorch we serialise via ``safetensors`` (fast, zero-copy load,
    no pickle attack surface). For TensorFlow we keep ``pickle`` since
    the value is a dict of TF tensors with no equivalent zero-copy
    format readily available.
    """

    def __init__(self, redis_url="redis://localhost:6379/0", prefix="lazy_weights",
                 framework=FrameworkType.PYTORCH):
        import redis
        self.r = redis.Redis.from_url(redis_url)
        self.prefix = prefix
        self.framework = framework

    def _key(self, name):
        return f"{self.prefix}:{name}"

    def get(self, key):
        raw = self.r.get(self._key(key))
        if not raw:
            return None
        if self.framework == FrameworkType.PYTORCH:
            try:
                from safetensors.torch import load as st_load
                return st_load(raw)
            except Exception:
                import torch
                buffer = io.BytesIO(raw)
                return torch.load(buffer, map_location="cpu")
        buffer = io.BytesIO(raw)
        return pickle.load(buffer)

    def put(self, key, value):
        if self.framework == FrameworkType.PYTORCH:
            try:
                from safetensors.torch import save as st_save
                blob = st_save(value)
            except Exception:
                import torch
                buffer = io.BytesIO()
                torch.save(value, buffer)
                blob = buffer.getvalue()
        else:
            buffer = io.BytesIO()
            pickle.dump(value, buffer)
            blob = buffer.getvalue()
        self.r.set(self._key(key), blob)

    def pop(self, key):
        self.r.delete(self._key(key))

    def keys(self):
        return [k.decode() for k in self.r.keys(f"{self.prefix}:*")]
