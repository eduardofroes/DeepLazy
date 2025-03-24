import os
from safetensors import safe_open


def index_safetensors_shards(shards_dir):
    index = {}
    for filename in os.listdir(shards_dir):
        if filename.endswith(".safetensors"):
            path = os.path.join(shards_dir, filename)
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    index[key] = path
    return index
