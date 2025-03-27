from deeplazy.core.lazy_model import LazyModel
from transformers import AutoTokenizer, GPT2Model, GPT2Config
from deeplazy.core.lazy_cache import LocalLRUCache
from deeplazy.core.lazy_tensor_loader import LazyLoader
import torch
import psutil
import os


def print_memory(stage=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2
    print(f"{stage}: {mem:.2f} MB")


if __name__ == "__main__":
    MODEL_PATH = "/opt/repository/gpt2_safetensors/model.safetensors"

    print_memory("Início")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    cache = LocalLRUCache(capacity=10)

    loader = LazyLoader(weights_path=[MODEL_PATH],
                        device='cpu', cache_backend=cache)

    model = LazyModel(
        config=GPT2Config.from_pretrained("gpt2"),
        cls=GPT2Model,
        loader=loader
    )

    inputs = tokenizer("Texto exemplo", return_tensors="pt")

    outputs1 = model(**inputs)
    print(outputs1)
    print_memory("Após primeiro forward")

    outputs2 = model(**inputs)
    print(outputs2)

    print_memory("Após segundo forward")
