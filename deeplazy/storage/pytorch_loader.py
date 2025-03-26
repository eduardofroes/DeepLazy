import os
import torch
from typing import Dict


class PyTorchStorageManager:
    """
    Gerenciador de arquivos .pt/.pth/.bin do PyTorch.
    Lê o state_dict e permite acesso aos tensores sob demanda.
    Se for passado um diretório, tenta automaticamente encontrar o arquivo de pesos.
    """

    def __init__(self, path: str):
        self.file_path = self._resolve_pytorch_file(path)
        self.state_dict = self._load_state_dict()
        self.tensor_index = self._build_tensor_index()

    def _resolve_pytorch_file(self, path: str) -> str:
        if os.path.isfile(path):
            return path

        if os.path.isdir(path):
            candidates = sorted([
                f for f in os.listdir(path)
                if f.endswith((".pt", ".pth", ".bin"))
            ])
            if not candidates:
                raise FileNotFoundError(
                    "Nenhum arquivo .pt, .pth ou .bin encontrado no diretório.")
            return os.path.join(path, candidates[0])

        raise FileNotFoundError(f"Caminho inválido: {path}")

    def _load_state_dict(self) -> Dict[str, torch.Tensor]:
        state_dict = torch.load(self.file_path, map_location="cpu")

        if hasattr(state_dict, "state_dict"):
            state_dict = state_dict.state_dict()

        if isinstance(state_dict, dict) and "model" in state_dict:
            return state_dict["model"]

        return state_dict

    def _build_tensor_index(self) -> Dict[str, torch.Size]:
        return {key: tensor.shape for key, tensor in self.state_dict.items()}

    def load_tensor(self, tensor_key: str) -> torch.Tensor:
        if tensor_key not in self.state_dict:
            raise KeyError(
                f"Tensor '{tensor_key}' não encontrado no arquivo {self.file_path}")
        return self.state_dict[tensor_key]

    def get_index(self) -> Dict[str, torch.Size]:
        return self.tensor_index
