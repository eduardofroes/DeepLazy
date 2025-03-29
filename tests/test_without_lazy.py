from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
import psutil
import os

# Função para medir o uso de memória do processo atual


def print_memory(stage):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024**2
    print(f"{stage} | Memória usada: {mem_mb:.2f} MB")


# Caminho local onde o modelo foi baixado (com .safetensors, config.json, etc)
MODEL_DIR = "/opt/repository/gpt2_lm"

# Carrega o tokenizer e o modelo a partir do diretório local
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel(GPT2Config.from_pretrained(MODEL_DIR))
model.eval()

# Prompt de entrada
prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

# 📏 Memória antes da geração
print_memory("Antes da geração")

# Gera texto
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        top_k=50,
        top_p=1,
        temperature=0.01,
        num_return_sequences=1
    )

# 📏 Memória após a geração
print_memory("Após a geração")

# Decodifica e exibe
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("📝 Texto gerado:")
print(generated_text)
