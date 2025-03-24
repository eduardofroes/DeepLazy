# core/redis_layer_cache.py

import redis
import pickle


class RedisLayerCache:
    def __init__(self, redis_host='localhost', redis_port=6379, db=0, prefix="layer_cache"):
        self.redis = redis.Redis(host=redis_host, port=redis_port, db=db)
        self.prefix = prefix

    def _key(self, weight_key):
        return f"{self.prefix}:{weight_key}"

    def get_weight(self, weight_key):
        data = self.redis.get(self._key(weight_key))
        if data:
            return pickle.loads(data)
        return None

    def put_weight(self, weight_key, tensor):
        try:
            self.redis.set(self._key(weight_key), pickle.dumps(tensor))
        except Exception as e:
            print(f"[RedisCache] Failed to store tensor: {e}")
