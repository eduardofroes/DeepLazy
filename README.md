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
from deeplazy.core.lazy_cache import PytorchLocalLRUCache
from deeplazy.core.lazy_tensor_loader import LazyLoader
import torch
from deeplazy.enums.framework_enum import FrameworkType



if __name__ == "__main__":
    pt_loader = LazyLoader(
        weights_path=["/opt/repository/gpt2_safetensors/model.safetensors"],
        device="cpu",
        cache_backend=PytorchLocalLRUCache(capacity=10),
        enable_monitor=True,
        model_name="gpt2_pytorch",
        framework=FrameworkType.PYTORCH
    )

    from transformers.models.gpt2.modeling_gpt2 import GPT2Model
    pt_model = LazyModel(config=GPT2Config(), cls=GPT2Model, loader=pt_loader)
    pt_input = torch.randint(0, 1000, (1, 10))
    pt_output = pt_model(input_ids=pt_input)
    print("PyTorch GPT2 output:", pt_output.last_hidden_state.shape)
```

```python
from transformers.models.gpt2.modeling_tf_gpt2 import TFGPT2Model
from deeplazy.core.lazy_model import LazyModel
from deeplazy.core.lazy_tensor_loader import LazyLoader
from deeplazy.core.lazy_cache import TFLRULazyCache
from deeplazy.enums.framework_enum import FrameworkType

from transformers.models.gpt2.configuration_gpt2 import GPT2Config

import tensorflow as tf

tf_loader = LazyLoader(
    weights_path=["/opt/repository/gpt2_safetensors/model.safetensors"],
    device="/CPU:0",
    cache_backend=TFLRULazyCache(capacity=4),
    enable_monitor=True,
    model_name="gpt2_tensorflow",
    framework=FrameworkType.TENSORFLOW
)


tf_model = LazyModel(config=GPT2Config(), cls=TFGPT2Model, loader=tf_loader)
tf_input = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
tf_output = tf_model(input_ids=tf_input)
print("TensorFlow GPT2 output:", tf_output.last_hidden_state.shape)
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
