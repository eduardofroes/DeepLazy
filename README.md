# 🧠 DeepLazy — Lazy Loading Framework for Large Language Models

**DeepLazy** is a modular and extensible Python library designed to facilitate **lazy loading of large language models (LLMs)**. By loading model weights layer-by-layer on-demand during inference, DeepLazy significantly reduces memory usage and startup time, making it ideal for environments with limited resources.

## 🌟 Key Features

- **Efficient Memory Usage**: Load only necessary model layers during inference.
- **Support for Heavy Models**: Optimized for Transformer-based models like LLaMA, DeepSeek, and Falcon.
- **Versatile Environment Compatibility**: Suitable for low-memory environments such as edge devices and research clusters.
- **Fine-Grained Profiling**: Offers detailed execution profiling and system monitoring.

---

## 📦 Installation

Install **DeepLazy** from [PyPI](https://pypi.org/project/deeplazy):

```bash
pip install deeplazy
```

> **Requirements**: Python ≥ 3.8 and either `torch` or `tensorflow`, depending on your chosen framework.

## 📚 Documentation

### Example Usage

Below is a more detailed example demonstrating the use of DeepLazy with a GPT-2 model:

```python
from deeplazy.core.lazy_model import LazyModel
from transformers import AutoTokenizer, GPT2Model, GPT2Config
from deeplazy.core.lazy_cache import LocalLRUCache
from deeplazy.core.lazy_tensor_loader import LazyLoader
import torch
import psutil
import os
from deeplazy.ui.dashboard_monitor import DashboardMonitor

def print_memory(stage=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2
    print(f"{stage}: {mem:.2f} MB")

if __name__ == "__main__":
    MODEL_PATH = "/opt/repository/gpt2_safetensors/model.safetensors"

    print_memory("Início")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    cache = LocalLRUCache(capacity=10)

    monitor = DashboardMonitor()
    monitor.enable()

    loader = LazyLoader(weights_path=[MODEL_PATH],
                        device='cpu', cache_backend=cache, enable_monitor=True)

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
```

---

## 📊 Built-in Dashboard (Optional)

Enable a **real-time terminal dashboard** for:

- Monitoring layer-by-layer execution
- Tracking memory consumption
- Observing CPU/GPU usage
- Measuring execution time per layer
- Viewing final model statistics

Activate by setting `enable_dashboard=True` in `.forward()`.

---

## 🔧 Cache Support

Choose your caching strategy:

- **Memory Cache** (default): In-memory caching of layer weights.
- **Redis Cache**: Share cache across multiple processes or machines.

Example configuration for Redis:

```python
cache_type='redis',
redis_config={'host': 'localhost', 'port': 6379, 'db': 0, 'prefix': 'layer_cache'}
```

---

## 📁 File Format

- Utilizes **`.safetensors` format with index.json**.
- Compatible with models exported via 🤗 Transformers or custom serialization.

---

## 🤝 Contributing

We welcome pull requests and feature suggestions.  
Please open an issue to discuss major changes before contributing.

---

## 📜 License

MIT License — Feel free to use, fork, and build upon this project.
