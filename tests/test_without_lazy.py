import torch
from transformers import AutoTokenizer, AutoConfig, GenerationConfig, AutoModelForCausalLM
import psutil
import os


def print_memory(stage=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2
    print(f"{stage}: {mem:.2f} MB")


if __name__ == "__main__":
    WEIGHTS_DIR = "/opt/repository/deepseek_qwen"

    # 🔁 Imprime memória antes de carregar o modelo
    print_memory("Antes de carregar o modelo")

    # ⚙️ Carrega config
    config = AutoConfig.from_pretrained(WEIGHTS_DIR, trust_remote_code=True)

    # 🧠 Carrega modelo inteiro na memória
    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS_DIR, trust_remote_code=True, torch_dtype=torch.float32)
    model.eval()

    # 🔁 Imprime memória após carregar o modelo
    print_memory("Depois de carregar o modelo")

    # ⚙️ Config de geração
    model.generation_config = GenerationConfig.from_pretrained(
        WEIGHTS_DIR, trust_remote_code=True)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

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
        outputs = model.generate(
            input_tensor.to("cpu"),
            max_new_tokens=100
        )

    result = tokenizer.decode(
        outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    print("📝 Resposta gerada:")
    print(result)

    # 🔁 Imprime memória após geração
    print_memory("Após geração")
