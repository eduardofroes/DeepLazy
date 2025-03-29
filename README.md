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
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from deeplazy.core.lazy_cache import PytorchLocalLRUCache
from deeplazy.core.lazy_tensor_loader import LazyLoader
from deeplazy.enums.framework_enum import FrameworkType
import torch
import psutil
import os


def print_memory(stage=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2
    print(f"{stage}: {mem:.2f} MB")


if __name__ == "__main__":
    WEIGHTS_DIR = "/opt/repository/gpt2_lm"

    pt_loader = LazyLoader(
        weights_dir=WEIGHTS_DIR,
        device="cpu",
        cache_backend=PytorchLocalLRUCache(capacity=10),
        enable_monitor=True,
        model_name="gpt2_pytorch",
        framework=FrameworkType.PYTORCH
    )

    # Inicializa o modelo lazy com o loader e a classe do modelo
    pt_model = LazyModel(cls=GPT2LMHeadModel, loader=pt_loader)

    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Frase inicial
    prompt = "The future of artificial intelligence is"

    # Tokeniza a entrada
    inputs = tokenizer(prompt, return_tensors="pt")

    # Acessa o modelo real com lazy loading aplicado
    model_for_generation = pt_model.model
    model_for_generation.eval()

    # Geração de texto
    with torch.no_grad():
        output_ids = model_for_generation.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            top_p=1,
            temperature=0.01,
            num_return_sequences=1
        )

    # Decodifica e imprime o texto gerado
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("📝 Texto gerado:")
    print(generated_text)
```

```python
from deeplazy.core.lazy_model import LazyModel
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from deeplazy.core.lazy_cache import TFLRULazyCache
from deeplazy.core.lazy_tensor_loader import LazyLoader
from deeplazy.enums.framework_enum import FrameworkType
import tensorflow as tf
import psutil
import os


def print_memory(stage=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2
    print(f"{stage}: {mem:.2f} MB")


if __name__ == "__main__":
    WEIGHTS_DIR = "/opt/repository/gpt2_lm"

    tf_loader = LazyLoader(
        weights_dir=WEIGHTS_DIR,
        device="cpu",
        cache_backend=TFLRULazyCache(capacity=10),
        enable_monitor=True,
        model_name="gpt2_tensorflow",
        framework=FrameworkType.TENSORFLOW
    )

    # Inicializa o modelo lazy sem necessidade de config
    lazy_model = LazyModel(cls=TFGPT2LMHeadModel, loader=tf_loader)

    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Prompt
    prompt = "The future of artificial intelligence is"

    # Tokeniza a entrada
    inputs = tokenizer(prompt, return_tensors="tf")

    model_for_generation = lazy_model.model
    model_for_generation.trainable = False

    # Geração
    output_ids = model_for_generation.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        top_k=50,
        top_p=1.0,
        temperature=0.01,
        num_return_sequences=1
    )

    # Decodificação
    generated_text = tokenizer.decode(
        output_ids[0].numpy(), skip_special_tokens=True)

    print("📝 Texto gerado:")
    print(generated_text)
```

---

## 📊 Built-in Dashboard (Optional)

Enable a **real-time terminal dashboard** for:

- Monitoring layer-by-layer execution
- Tracking memory consumption
- Observing CPU/GPU usage
- Measuring execution time per layer
- Viewing final model statistics

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

## 📊 Dashboard Example

![Dashboard Example](./docs/dashboard.png)

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
