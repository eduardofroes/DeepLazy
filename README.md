# 🧠 DeepLazy — Lazy Loading Framework for Large Language Models (LLMs)

**DeepLazy** is a Python framework designed to efficiently load and run **large-scale language models (LLMs)** using a **lazy-loading mechanism**, drastically reducing memory usage and initialization time.

---

## 🚀 Why DeepLazy?

Traditional loading of large models consumes a huge amount of memory, even when only a few layers are used at a time. **DeepLazy** solves this by:

- Loading only **a few layers at a time**.
- **Unloading layers after execution**.
- Supporting **in-memory or Redis-based caching**.
- Providing a **real-time terminal dashboard** (K9s-like interface) with layer-by-layer tracking.

---

## 📦 Features

- ✅ Lazy layer-by-layer loading
- ✅ Support for **PyTorch** and **TensorFlow**
- ✅ Integration with `.safetensors` format
- ✅ Memory and execution time tracking
- ✅ K9s-style **real-time terminal dashboard**
- ✅ Redis or in-memory weight caching

---

## 📁 Folder Structure

```
deeplazy/
│
├── core/
│   ├── lazy_model.py
│   ├── lazy_model_builder.py
│   ├── lazy_layer.py
│   ├── architecture_parser.py
│   └── layer_cache.py
│
├── adapters/
│   ├── pytorch_adapter.py
│   └── tensorflow_adapter.py
│
├── storage/
│   └── safetensors_loader.py
│
├── ui/
│   └── dashboard_monitor.py
│
├── enums/
│   ├── framework_enum.py
│   └── layer_type_enum.py
│
└── example_loader.py  ← Example usage
```

---

## 🔧 Installation

```bash
pip install -r requirements.txt
```

Dependencies include:

- `torch` or `tensorflow`
- `psutil`
- `safetensors`
- `rich`
- `redis` _(optional)_

---

## 🧪 Example Usage

```python
# example_loader.py
import os
import torch
import psutil
import asyncio
import time

from storage.safetensors_loader import SafeTensorStorageManager
from core.lazy_model_builder import LazyModelBuilder


def print_memory(stage=""):
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    print(f"[MEMORY] {stage}: {mem:.2f} MB")


async def main():
    # Step 1 - Initialize SafeTensor Storage
    storage = SafeTensorStorageManager(
        shards_dir="/path/to/model_directory",
        index_path="/path/to/model_directory/model.safetensors.index.json"
    )

    # Step 2 - Build Lazy Model
    builder = LazyModelBuilder(
        framework='torch',                             # or 'tensorflow'
        config_path="/path/to/model_directory/config.json",
        index_path="/path/to/model_directory/model.safetensors.index.json",
        storage=storage,
        max_layers_in_memory=30,
        use_cache=True,
        cache_type='memory',                           # 'memory' or 'redis'
        redis_config=None
    )
    model = builder.build_model()

    # Step 3 - Prepare Dummy Input
    input_dim = model.metadata.get("hidden_size", 768)
    dummy_input = torch.randint(0, input_dim, size=(1, input_dim), dtype=torch.long)

    print_memory("Before Forward Pass")

    # Step 4 - Run Inference (with dashboard)
    start_time = time.time()
    output = await model.forward(dummy_input, enable_dashboard=True)
    end_time = time.time()

    print_memory("After Forward Pass")

    print(f"✅ Inference Time: {end_time - start_time:.4f} seconds")
    print("Output Shape:", output.shape)


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 📊 Real-Time Terminal Dashboard

When `enable_dashboard=True` is passed to `.forward()`, DeepLazy launches a real-time K9s-style dashboard with:

- Layer-by-layer execution status
- Execution time per layer
- Live memory usage
- CPU and GPU usage
- Framework and cache configuration

📸 Example Output:

```
🧠 Lazy LLM Execution Dashboard — Model: deepseek-v2
Layer                                | Built | Time (s) | Memory (MB)
------------------------------------|-------|----------|-------------
model.layer.0.mlp.gate_proj         | ✅    | 0.11     | 1032.54
...

📊 System Info
📦 Model: deepseek-v2
📁 Safetensors Path: /models/deepseek-v2
⏱️ Total Execution Time: 6.87 seconds
🧠 Final Memory Usage: 1380.12 MB
💻 CPU Usage: 47.3%
🖥️ GPU Usage: 63%
🔄 Max Layers in Memory: 30
🔄 Cache Type: memory
⚙️ Framework: torch
```

---

## ⚡ Redis Cache Support

You can enable Redis caching like this:

```python
LazyModelBuilder(
    cache_type='redis',
    redis_config={
        'host': 'localhost',
        'port': 6379,
        'db': 0,
        'prefix': 'deeplazy_cache'
    },
    ...
)
```

---

## 🤝 Contributing

Feel free to fork the project and submit pull requests for:

- Adapter improvements
- Support for more frameworks
- Optimizations in caching and memory handling
- UX improvements in the dashboard

---

## 📄 License

MIT License © 2025
