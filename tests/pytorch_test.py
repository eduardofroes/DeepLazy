import pytest
pytest.skip("Requires ML stack", allow_module_level=True)
import torch
from deeplazy.core.lazy_model import LazyModel
from deeplazy.core.lazy_cache import PytorchLocalLRUCache
from deeplazy.core.lazy_tensor_loader import LazyLoader
from deeplazy.enums.framework_enum import FrameworkType

from transformers import AutoTokenizer, AutoConfig, GenerationConfig, AutoModelForCausalLM
import psutil
import os


def print_memory(stage=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2
    print(f"{stage}: {mem:.2f} MB")


if __name__ == "__main__":
    WEIGHTS_DIR = "/opt/repository/deepseek_qwen"

    # 🔁 Inicializa o loader com lazy loading
    pt_loader = LazyLoader(
        weights_dir=WEIGHTS_DIR,
        device="cpu",  # ou "cuda" se preferir
        cache_backend=PytorchLocalLRUCache(capacity=6),
        enable_monitor=True,
        model_name="deepseek-qwen-1.5b",
        framework=FrameworkType.PYTORCH
    )

    # ⚙️ Carrega config do modelo
    config = AutoConfig.from_pretrained(WEIGHTS_DIR, trust_remote_code=True)

    # 🧠 Cria o modelo com LazyModel
    pt_model = LazyModel(
        cls=AutoModelForCausalLM,
        loader=pt_loader,
    )

    model_for_generation = pt_model.model
    model_for_generation.generation_config = GenerationConfig.from_pretrained(
        WEIGHTS_DIR, trust_remote_code=True)
    model_for_generation.generation_config.pad_token_id = model_for_generation.generation_config.eos_token_id
    model_for_generation.eval()

    # 🧾 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        WEIGHTS_DIR, trust_remote_code=True)

    # 💬 Conversa
    messages = [
        {"role": "user", "content": "Write a piece of quicksort code in C++"}
    ]
    input_tensor = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt")

    # ✨ Geração
    with torch.no_grad():
        outputs = model_for_generation.generate(
            input_tensor.to(pt_loader.device),
            max_new_tokens=10
        )

    result = tokenizer.decode(
        outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    print("📝 Resposta gerada:")
    print(result)
