from deeplazy.core.lazy_model import LazyModel
from transformers import AutoTokenizer, GPT2Model, GPT2Config
from deeplazy.core.lazy_cache import PytorchLocalLRUCache
from deeplazy.core.lazy_tensor_loader import LazyLoader
import torch
import psutil
import os
from deeplazy.enums.framework_enum import FrameworkType


def print_memory(stage=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2
    print(f"{stage}: {mem:.2f} MB")


if __name__ == "__main__":
    # PyTorch - Exemplo com Redis
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
