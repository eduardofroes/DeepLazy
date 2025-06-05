import pytest
pytest.skip("Requires ML stack", allow_module_level=True)
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
import torch.nn as nn
import torch
from functools import wraps
import psutil
import os
import gc
from safetensors import safe_open

PTH_PATH = "/opt/repository/gpt2_safetensors/model.safetensors"

_seen_classes = set()
_applied_modules = set()


def print_memory_usage(description):
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024**2
    print(f"\n{description}:")
    print(f"CPU: {cpu_mem:.2f} MB")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu_mem = torch.cuda.memory_allocated() / 1024**2
        print(f"GPU: {gpu_mem:.2f} MB")


def is_on_meta(module):
    try:
        return next(module.parameters()).device == torch.device('meta')
    except StopIteration:
        return False


def load_submodule_weights(module_name):
    with safe_open(PTH_PATH, framework="pt") as f:
        return {
            k: f.get_tensor(k)
            for k in f.keys()
            if k.startswith(module_name + ".")
        }


def patch_all_module_classes():
    def recursive_patch(cls):
        if cls in _seen_classes:
            return
        _seen_classes.add(cls)

        if not issubclass(cls, nn.Module):
            return

        orig_call = cls.__call__

        if getattr(orig_call, "_is_patched", False):
            return

        @wraps(orig_call)
        def wrapper(self, *args, **kwargs):
            module_name = getattr(self, '_lazy_module_name', None)

            if module_name and id(self) not in _applied_modules:
                try:
                    sub_state = load_submodule_weights(module_name)

                    if is_on_meta(self) and sub_state:
                        target_device = next(iter(sub_state.values())).device

                        print(f"\n--- Carregando {module_name} ---")
                        print_memory_usage("Antes")

                        self.to_empty(device=target_device)
                        if hasattr(self, 'reset_parameters'):
                            self.reset_parameters()

                        self.load_state_dict(sub_state, strict=False)
                        _applied_modules.add(id(self))
                        print(f"✅ {module_name}")
                        print_memory_usage("Após carga")

                except Exception as e:
                    print(f"⚠️ Erro em {module_name}: {str(e)}")

            result = orig_call(self, *args, **kwargs)

            if module_name and id(self) in _applied_modules:
                # Limpeza profunda
                for name in list(self._parameters.keys()):
                    param = self._parameters.pop(name)
                    del param

                for name in list(self._buffers.keys()):
                    buffer = self._buffers.pop(name)
                    del buffer

                # Remove referências nos módulos pais
                parent = next((m for m in self.modules()
                              if self in m.children()), None)
                if parent:
                    for name, child in list(parent.named_children()):
                        if child is self:
                            del parent._modules[name]

                self.to(device='meta')
                _applied_modules.remove(id(self))

                # Liberação agressiva de memória
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print(f"♻️ {module_name}")
                print_memory_usage("Após descarga")

            return result

        wrapper._is_patched = True
        cls.__call__ = wrapper

        for subclass in cls.__subclasses__():
            recursive_patch(subclass)

    recursive_patch(nn.Module)


def annotate_module_names(model):
    for name, module in model.named_modules():
        module._lazy_module_name = name


print_memory_usage("Início")

patch_all_module_classes()

print_memory_usage("Antes de criar modelo")
config = GPT2Config.from_pretrained("gpt2")
model = GPT2Model(config).to(device='meta')
annotate_module_names(model)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print_memory_usage("Após tokenizer")

inputs = tokenizer("Teste de memória eficiente", return_tensors="pt")
print_memory_usage("Antes do forward")
outputs = model(**inputs)
print_memory_usage("Após forward")

print("Saída:", outputs)
