import pytest
pytest.skip("Requires ML stack", allow_module_level=True)
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
