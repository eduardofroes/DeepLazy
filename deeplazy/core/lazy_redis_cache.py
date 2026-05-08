"""Backwards-compatible alias.

The Redis backend now lives in :mod:`deeplazy.core.lazy_cache`. This
module re-exports it so existing imports keep working.
"""

from deeplazy.core.lazy_cache import RedisCacheBackend

__all__ = ["RedisCacheBackend"]
