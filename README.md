# 🧠 DeepLazy — Lazy Loading Framework for Large Language Models

**DeepLazy** is a modular and extensible Python library that enables **lazy loading of large language models (LLMs)** — only loading the model weights layer-by-layer on-demand during inference, significantly reducing memory usage and startup time.

Ideal for:

- Heavy Transformer-based models (e.g., LLaMA, DeepSeek, Falcon)
- Low-memory environments (edge devices, research clusters)
- Fine-grained execution profiling and system monitoring

---

## 📦 Installation

You can install **DeepLazy** directly from [PyPI](https://pypi.org/project/deeplazy):

```bash
pip install deeplazy
```

> Requires Python ≥ 3.8 and `torch` or `tensorflow`, depending on the framework you intend to use.

---

## 🚀 Quick Example

```python
import asyncio
import torch
from deeplazy.storage.safetensors_loader import SafeTensorStorageManager
from deeplazy.core.lazy_model_builder import LazyModelBuilder

async def main():
    # Step 1: Initialize storage from .safetensors
    storage = SafeTensorStorageManager(
        "./your_model_dir"
    )

    # Step 2: Build the lazy model
    builder = LazyModelBuilder(
        framework='torch',
        storage=storage,
        config_path="./your_model_dir/config.json",
        max_layers_in_memory=20,
        use_cache=True,
        cache_type='memory'  # or 'redis'
    )
    model = builder.build_model()

    # Step 3: Prepare input
    input_dim = model.metadata.get("hidden_size", 768)
    x = torch.randint(low=0, high=input_dim, size=(1, input_dim), dtype=torch.long)

    # Step 4: Run inference
    output = await model.forward(x, enable_dashboard=True)
    print("✅ Output shape:", output.shape)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 📊 Built-in Dashboard (Optional)

Enable a **real-time terminal dashboard** (like `k9s`) with:

- Layer-by-layer execution monitoring
- Memory consumption
- CPU/GPU usage
- Execution time per layer
- Final model stats

Just pass `enable_dashboard=True` in `.forward()`.

---

## 🔧 Cache Support

Choose between:

- `memory` cache (default): in-memory layer weight caching.
- `redis` cache: share cache across processes/machines.

Example:

```python
cache_type='redis',
redis_config={'host': 'localhost', 'port': 6379, 'db': 0, 'prefix': 'layer_cache'}
```

---

## 📁 File Format

- DeepLazy uses **`.safetensors` format with index.json**.
- Compatible with models exported via 🤗 Transformers or custom serialization.

---

## 🤝 Contributing

Pull requests and feature suggestions are welcome.  
Please open an issue first to discuss major changes.

---

## 📜 License

MIT License — Feel free to use, fork, and build on top of it.
