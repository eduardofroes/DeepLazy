from collections import OrderedDict


class LayerCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        self.layer_names = []

    def put_weight(self, key, tensor):
        if key not in self.cache:
            if len(self.cache) >= self.max_size:
                removed = self.layer_names.pop(0)
                self.cache.pop(removed, None)
            self.layer_names.append(key)
        self.cache[key] = tensor

    def get_weight(self, key):
        return self.cache.get(key, None)

    def is_layer_cached(self, keys):
        return all(k in self.cache for k in keys)
