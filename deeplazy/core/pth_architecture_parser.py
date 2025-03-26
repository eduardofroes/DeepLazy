import re
import torch
from enum import Enum
from typing import Dict
from collections import defaultdict
from enums.layer_type_enum import LayerType
from collections import OrderedDict
from transformers import AutoConfig


class PyTorchGenericLayerParser:
    def __init__(self, state_dict: Dict[str, torch.Tensor], config_path: str):
        self.state_dict = state_dict
        self.tree = self._build_tree()

        self.config = AutoConfig.from_pretrained(
            config_path, local_files_only=True, trust_remote_code=True)

    def _build_tree(self) -> dict:
        tree = OrderedDict()
        for key in self.state_dict:
            parts = key.split(".")
            node = tree
            for part in parts:
                node = node.setdefault(part, OrderedDict())
            node["_leaf"] = key  # marca folha
        return tree

    def _gather_params(self, node: dict) -> Dict[str, torch.Tensor]:
        result = {}
        if "_leaf" in node:
            result[node["_leaf"]] = self.state_dict[node["_leaf"]]
        for k, v in node.items():
            if isinstance(v, dict):
                result.update(self._gather_params(v))
        return result

    def _walk_and_parse(self, node: dict, path: str = "") -> Dict[str, Dict]:
        architecture = {}

        # Recolhe todos os parâmetros do path atual e seus filhos
        params = self._gather_params(node)
        layer_type = self._infer_layer_type(path, params)

        if layer_type != LayerType.UNKNOWN:
            # Se for um tipo conhecido, registra o path atual com todos os parâmetros dos filhos também
            pytorch_class = self._get_pytorch_class_name(layer_type, params)
            architecture[path] = {
                "type": layer_type.value,
                "class": pytorch_class,
                "params": params
            }
            return architecture  # Não precisa descer mais, já agrupou

        for k, v in node.items():
            if k == "_leaf":
                continue
            sub_path = f"{path}.{k}" if path else k
            architecture.update(self._walk_and_parse(v, sub_path))

        return architecture

    def parse(self) -> Dict[str, Dict]:
        return self._walk_and_parse(self.tree)

    def _infer_layer_type(self, prefix: str, params: Dict[str, torch.Size]) -> LayerType:
        lower_prefix = prefix.lower()

        if "in_proj" in lower_prefix:
            return LayerType.MULTIHEAD_ATTENTION

        if "attn" in lower_prefix:
            # Refinar ainda mais abaixo se necessário
            if "c_attn" in lower_prefix:
                any_qkv = any(
                    "c_attn.weight" in k and v[1] % 3 == 0 for k, v in params.items())
                if any_qkv:
                    return LayerType.ATTENTION_QKV
            if "c_proj" in lower_prefix:
                return LayerType.ATTENTION_PROJ
            return LayerType.ATTENTION

        if "ln" in lower_prefix or "norm" in lower_prefix:
            if all(len(v) == 1 for v in params.values()):
                return LayerType.LAYERNORM

        if "embedding" in lower_prefix or "wte" in lower_prefix or "emb" in lower_prefix:
            return LayerType.EMBEDDING

        if "position" in lower_prefix or "wpe" in lower_prefix or "pos" in lower_prefix:
            return LayerType.POSITIONAL_EMBEDDING

        if "conv" in lower_prefix:
            for k, shape in params.items():
                if len(shape) == 3:
                    return LayerType.CONV1D
                elif len(shape) == 4:
                    return LayerType.CONV2D
                elif len(shape) == 5:
                    return LayerType.CONV3D

        if "proj" in lower_prefix:
            return LayerType.ATTENTION_PROJ

        if "mlp" in lower_prefix:
            if any(".weight" in k and len(v) == 2 for k, v in params.items()):
                return LayerType.LINEAR

        return LayerType.UNKNOWN

    def _get_pytorch_class_name(self, layer_type: LayerType, params: Dict[str, torch.Size]) -> str:
        mapping = {
            LayerType.LINEAR: "torch.nn.Linear",
            LayerType.EMBEDDING: "torch.nn.Embedding",
            LayerType.POSITIONAL_EMBEDDING: "torch.nn.Embedding",
            LayerType.LAYERNORM: "torch.nn.LayerNorm",
            LayerType.RMSNORM: "torch.nn.Identity",
            LayerType.BATCHNORM: "torch.nn.BatchNorm1d",
            LayerType.GROUPNORM: "torch.nn.GroupNorm",
            LayerType.INSTANCENORM: "torch.nn.InstanceNorm1d",
            LayerType.CONV1D: "torch.nn.Conv1d",
            LayerType.CONV2D: "torch.nn.Conv2d",
            LayerType.CONV3D: "torch.nn.Conv3d",
            LayerType.ATTENTION: "torch.nn.MultiheadAttention",
            LayerType.MULTIHEAD_ATTENTION: "torch.nn.MultiheadAttention",
            LayerType.ATTENTION_PROJ: "torch.nn.Linear",
            LayerType.ATTENTION_QKV: "torch.nn.Linear",
            LayerType.ATTENTION_MASK: "torch.nn.Identity",
            LayerType.BIAS: "torch.nn.Identity",
            LayerType.SCALING: "torch.nn.Identity",
            LayerType.POOLING: "torch.nn.AdaptiveAvgPool1d",
            LayerType.CLASSIFIER: "torch.nn.Linear",
            LayerType.CLASSIFIER_FINAL: "torch.nn.Linear",
            LayerType.DEPTHWISE_CONV: "torch.nn.Conv2d",
        }
        return mapping.get(layer_type, "Unknown")

    def print_summary(self):
        arch = self.parse()
        for name, info in arch.items():
            print(f"{name:50} {info['type']:25} {info['class']}")
