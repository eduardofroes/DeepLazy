import torch
import asyncio
import os
import time
import psutil
from transformers import AutoTokenizer

from storage.pytorch_loader import PyTorchStorageManager
from core.lazy_model_builder import LazyModelBuilder


def print_memory(stage=""):
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    print(f"[MEMORY] {stage}: {mem:.2f} MB")


async def main():
    MODEL_DIR = "/opt/repository/gpt2_safetensors"

    # Inicializa storage
    storage = PyTorchStorageManager(MODEL_DIR)

    # Constrói modelo lazy
    builder = LazyModelBuilder(
        framework='torch',
        storage=storage,
        config_path=os.path.join(MODEL_DIR, "config.json"),
        max_layers_in_memory=10,
        use_cache=True,
        cache_type='memory',
        redis_config=None
    )
    model = builder.build_pytorch_model(storage)

    # Tokenizer do Roberta
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

    # Entrada QA
    context = "The capital of France is Paris."
    question = "What is the capital of France?"

    # Prepara entrada
    inputs = tokenizer(question, context, return_tensors="pt")
    input_ids = inputs["input_ids"]

    print_memory("Before forward")
    start_time = time.time()

    outputs = await model.forward(input_ids, enable_dashboard=False)
    start_logits = outputs.start_logits if hasattr(
        outputs, "start_logits") else outputs[0]
    end_logits = outputs.end_logits if hasattr(
        outputs, "end_logits") else outputs[1]

    # Obtem resposta
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits) + 1
    answer = tokenizer.decode(input_ids[0][start_index:end_index])

    end_time = time.time()
    print_memory("After forward")

    print("\n📝 Question:", question)
    print("📘 Context:", context)
    print("✅ Answer:", answer)
    print(f"⏱️ Inference time: {end_time - start_time:.4f}s")


if __name__ == "__main__":
    asyncio.run(main())
