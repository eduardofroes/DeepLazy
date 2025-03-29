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
