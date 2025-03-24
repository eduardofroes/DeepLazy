from safetensors import safe_open
import os
import json


class SafeTensorStorageManager:
    """
    Gerenciador de shards safetensors com suporte à ordem de mapeamento definida no index_path,
    mas leitura física dos arquivos safetensors sempre ocorre, garantindo disponibilidade real.
    """

    def __init__(self, shards_dir: str, index_path: str = None):
        self.shards_dir = shards_dir
        self.index_path = index_path

        # Etapa 1: Carregar index (ordem dos tensores)
        self.tensor_index = self._build_tensor_index()

        # Etapa 2: Carregar arquivos fisicamente com safe_open
        self.files = self._load_all_safetensor_files()

    def _build_tensor_index(self) -> dict:
        index = {}

        if self.index_path and os.path.exists(self.index_path):
            with open(self.index_path, "r") as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            for tensor_name, file_name in weight_map.items():
                full_path = os.path.join(self.shards_dir, file_name)
                index[tensor_name] = full_path
        else:
            safetensor_files = sorted(
                f for f in os.listdir(self.shards_dir) if f.endswith(".safetensors")
            )
            for fname in safetensor_files:
                path = os.path.join(self.shards_dir, fname)
                with safe_open(path, framework="pt") as f:
                    for key in f.keys():
                        index[key] = path

        return index

    def _load_all_safetensor_files(self):
        """Carrega fisicamente todos os arquivos .safetensors encontrados no diretório."""
        files = {}
        safetensor_files = sorted(
            f for f in os.listdir(self.shards_dir) if f.endswith(".safetensors")
        )
        for fname in safetensor_files:
            path = os.path.join(self.shards_dir, fname)
            if path not in files:
                files[path] = safe_open(path, framework="pt")
        return files

    def load_tensor(self, tensor_key: str):
        path = self.tensor_index[tensor_key]
        return self.files[path].get_tensor(tensor_key)

    def get_index(self):
        return self.tensor_index
