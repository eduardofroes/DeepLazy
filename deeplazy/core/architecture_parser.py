import os
import json
from transformers import AutoConfig
from enums.layer_type_enum import LayerType
import re
from collections import defaultdict


class ModelArchitectureParser:
    def __init__(self, config_path, index_path, tensor_index):
        if os.path.isfile(config_path):
            config_path = os.path.dirname(config_path)

        self.config = AutoConfig.from_pretrained(
            config_path, local_files_only=True, trust_remote_code=True)

        self.index_path = index_path
        self.tensor_index = tensor_index

        if self.index_path:
            if os.path.isfile(self.index_path):
                self.index = json.load(open(self.index_path))
                self.tensor_keys = list(self.index["weight_map"].keys())
        else:
            self.tensor_keys = list(self.tensor_index.keys())

        self.model_config = self.config.to_dict()

    def _infer_layer_type(self, key: str) -> LayerType:
        key = key.lower()

        if "multihead_attn" in key or "multihead_attention" in key:
            return LayerType.MULTIHEAD_ATTENTION
        if "self_attn" in key or "self_attention" in key:
            return LayerType.SELF_ATTENTION
        if "cross_attn" in key or "cross_attention" in key:
            return LayerType.CROSS_ATTENTION
        if any(x in key for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            return LayerType.ATTENTION_PROJ
        if "attn" in key or "attention" in key:
            return LayerType.ATTENTION

        if any(x in key for x in ["gate_proj", "up_proj", "down_proj", "dense", "fc", "linear", "proj", "mlp"]):
            return LayerType.LINEAR

        if any(x in key for x in ["wpe", "position_emb", "position_embedding", "pos_emb", "pos_embedding"]):
            return LayerType.POSITIONAL_EMBEDDING

        if any(x in key for x in ["wte", "embedding", "embed_tokens", "token_emb", "segment_emb", "rotary_emb", "embed", "emb"]):
            return LayerType.EMBEDDING

        if "ln_f" in key:
            return LayerType.LAYERNORM_FINAL
        if "rmsnorm" in key:
            return LayerType.RMSNORM
        if ("layernorm" in key or "ln_" in key or "ln" in key) and "ln_f" not in key:
            return LayerType.LAYERNORM
        if "batchnorm" in key or "bn" in key:
            return LayerType.BATCHNORM
        if "groupnorm" in key or "gn" in key:
            return LayerType.GROUPNORM
        if "instancenorm" in key:
            return LayerType.INSTANCENORM

        if "depthwise" in key:
            return LayerType.DEPTHWISE_CONV
        if "conv3d" in key:
            return LayerType.CONV3D
        if "conv2d" in key:
            return LayerType.CONV2D
        if "conv1d" in key:
            return LayerType.CONV1D
        if "conv" in key:
            return LayerType.CONV

        if any(x in key for x in ["classifier", "output_head", "predictions", "output_proj"]):
            return LayerType.CLASSIFIER

        if "lm_head" in key:
            return LayerType.CLASSIFIER_FINAL

        if "pool" in key or "pooler" in key:
            return LayerType.POOLING

        if "bias" in key and "norm" not in key:
            return LayerType.BIAS

        if "scale" in key or "scaling" in key:
            return LayerType.SCALING

        if key.endswith(".weight") or key.endswith(".kernel"):
            return LayerType.LINEAR

        return LayerType.UNKNOWN

    def get_architecture_schema(self):
        raw_schema = {}
        embedding_keys = []

        activation_fn = (
            self.model_config.get("hidden_act") or
            self.model_config.get("activation_function") or
            self.model_config.get("activation") or
            self.model_config.get("act_fn") or
            "gelu"
        )

        defaults = {
            "hidden_size": self.model_config.get("hidden_size", 768),
            "intermediate_size": self.model_config.get("intermediate_size", 3072),
            "vocab_size": self.model_config.get("vocab_size", 30522),
            "num_attention_heads": self.model_config.get("num_attention_heads", 12),
            "num_hidden_layers": self.model_config.get("num_hidden_layers", 12),
            "layer_norm_eps": self.model_config.get("layer_norm_eps", 1e-5),
            "is_encoder_decoder": self.model_config.get("is_encoder_decoder", False),
            "is_decoder": self.model_config.get("is_decoder", False),
            "activation_function": activation_fn
        }

        for name in self.tensor_keys:
            layer_type = self._infer_layer_type(name)
            raw_schema[name] = {"type": layer_type.value}

            if layer_type == LayerType.ATTENTION_QKV:
                raw_schema[name].update({
                    "in_features": defaults["hidden_size"],
                    "out_features": 3 * defaults["hidden_size"]
                })
            elif layer_type in {LayerType.ATTENTION_PROJ, LayerType.LINEAR}:
                raw_schema[name].update({
                    "in_features": defaults["hidden_size"],
                    "out_features": defaults["hidden_size"],
                    "activation_function": defaults["activation_function"]
                })
            elif layer_type in [LayerType.ATTENTION, LayerType.MULTIHEAD_ATTENTION, LayerType.SELF_ATTENTION, LayerType.CROSS_ATTENTION]:
                raw_schema[name].update({
                    "in_features": defaults["hidden_size"],
                    "num_heads": defaults["num_attention_heads"],
                })
            elif layer_type == LayerType.BIAS:
                raw_schema[name].update({
                    "in_features": defaults["hidden_size"],
                    "out_features": defaults["hidden_size"]
                })
            elif layer_type in {LayerType.EMBEDDING, LayerType.POSITIONAL_EMBEDDING}:
                raw_schema[name].update({
                    "vocab_size": defaults["vocab_size"],
                    "hidden_size": defaults["hidden_size"]
                })
                embedding_keys.append(name)
            elif layer_type in {
                LayerType.LAYERNORM, LayerType.LAYERNORM_FINAL,
                LayerType.RMSNORM, LayerType.BATCHNORM,
                LayerType.GROUPNORM, LayerType.INSTANCENORM
            }:
                raw_schema[name].update({
                    "hidden_size": defaults["hidden_size"],
                    "eps": defaults["layer_norm_eps"]
                })
            elif layer_type in {
                LayerType.CONV1D, LayerType.CONV2D,
                LayerType.CONV3D, LayerType.DEPTHWISE_CONV, LayerType.CONV
            }:
                raw_schema[name].update({
                    "in_channels": defaults["hidden_size"],
                    "out_channels": defaults["hidden_size"],
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1
                })
            elif layer_type == LayerType.CLASSIFIER:
                raw_schema[name].update({
                    "in_features": defaults["hidden_size"],
                    "out_features": defaults["hidden_size"]
                })
            elif layer_type == LayerType.CLASSIFIER_FINAL:
                raw_schema[name].update({
                    "in_features": defaults["hidden_size"],
                    "out_features": defaults["vocab_size"]
                })
            elif layer_type == LayerType.SCALING:
                raw_schema[name].update({
                    "hidden_size": defaults["hidden_size"]
                })

        if not self.index_path:

            lay_norm_final_keys = []
            classifier_final_keys = []

            for key in raw_schema:
                layer_type = raw_schema[key]["type"]
                if layer_type == LayerType.LAYERNORM_FINAL.value:
                    lay_norm_final_keys.append(key)
                elif layer_type == LayerType.CLASSIFIER_FINAL.value:
                    classifier_final_keys.append(key)

            has_classifier_final = any(
                raw_schema[k]["type"] == LayerType.CLASSIFIER_FINAL.value for k in raw_schema
            )
            if not has_classifier_final and "wte.weight" in raw_schema:
                raw_schema["lm_head.weight"] = {
                    "type": LayerType.CLASSIFIER_FINAL.value,
                    "in_features": defaults["hidden_size"],
                    "out_features": defaults["vocab_size"],
                    "tied_with": "wte.weight"
                }
                classifier_final_keys.append("lm_head.weight")

            intermediate_keys = [
                k for k in self.tensor_keys
                if k not in embedding_keys + lay_norm_final_keys + classifier_final_keys
                and k in raw_schema
            ]

            intermediate_keys = self.sort_by_any_layer_index(intermediate_keys)

            ordered_keys = embedding_keys + intermediate_keys + \
                lay_norm_final_keys + classifier_final_keys
            schema = {key: raw_schema[key] for key in ordered_keys}

            schema['metadata'] = defaults
            return schema

        raw_schema['metadata'] = defaults
        return raw_schema

    def sort_by_any_layer_index(self, keys):
        pattern = re.compile(r"(.*?)\.(\d+)\.")

        grouped_layers = defaultdict(list)
        others = []

        for idx, key in enumerate(keys):
            match = pattern.search(key)
            if match:
                prefix = match.group(1)
                index = int(match.group(2))
                block_key = (prefix, index)
                grouped_layers[block_key].append((idx, key))
            else:
                others.append((idx, key))  # mantém ordem original

        sorted_keys = []
        for block_key in sorted(grouped_layers, key=lambda x: x[1]):
            block = grouped_layers[block_key]
            sorted_keys.extend([k for _, k in sorted(block)])

        sorted_keys.extend([k for _, k in sorted(others)])

        return sorted_keys
