# core/architecture_parser.py

import os
from transformers import AutoConfig
from enums.layer_type_enum import LayerType


class ModelArchitectureParser:
    def __init__(self, config_path, tensor_index):
        if os.path.isfile(config_path):
            config_path = os.path.dirname(config_path)

        self.config = AutoConfig.from_pretrained(
            config_path, local_files_only=True, trust_remote_code=True
        )
        self.tensor_index = tensor_index
        self.tensor_keys = list(tensor_index.keys())
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
            return LayerType.ATTENTION_PROJECTION
        if "attn" in key or "attention" in key:
            return LayerType.ATTENTION
        if any(x in key for x in ["gate_proj", "up_proj", "down_proj", "dense", "fc", "linear", "proj", "mlp"]):
            return LayerType.LINEAR
        if any(x in key for x in ["embedding", "embed_tokens", "token_emb", "position_emb", "segment_emb", "rotary_emb"]):
            return LayerType.EMBEDDING
        if "rmsnorm" in key:
            return LayerType.RMSNORM
        if "layernorm" in key or "ln_" in key or "ln" in key:
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
        if any(x in key for x in ["classifier", "lm_head", "output_head", "predictions"]):
            return LayerType.CLASSIFIER
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
        schema = {}

        defaults = {
            "hidden_size": self.model_config.get("hidden_size", 768),
            "intermediate_size": self.model_config.get("intermediate_size", 3072),
            "vocab_size": self.model_config.get("vocab_size", 30522),
            "num_attention_heads": self.model_config.get("num_attention_heads", 12),
            "num_hidden_layers": self.model_config.get("num_hidden_layers", 12),
            "layer_norm_eps": self.model_config.get("layer_norm_eps", 1e-5),
            "is_encoder_decoder": self.model_config.get("is_encoder_decoder", False),
            "is_decoder": self.model_config.get("is_decoder", False),
        }

        for key in self.tensor_keys:
            layer_key = ".".join(key.split(".")[:-1])
            if layer_key not in schema:
                layer_type = self._infer_layer_type(key)
                schema[layer_key] = {"type": layer_type.value}

                if layer_type in {
                    LayerType.SELF_ATTENTION,
                    LayerType.CROSS_ATTENTION,
                    LayerType.MULTIHEAD_ATTENTION,
                    LayerType.ATTENTION,
                }:
                    schema[layer_key].update({
                        "hidden_size": defaults["hidden_size"],
                        "num_heads": defaults["num_attention_heads"],
                        "head_size": defaults["hidden_size"] // defaults["num_attention_heads"]
                    })

                elif layer_type in {LayerType.ATTENTION_PROJECTION, LayerType.LINEAR}:
                    schema[layer_key].update({
                        "in_features": defaults["hidden_size"],
                        "out_features": defaults["hidden_size"]
                    })

                elif layer_type == LayerType.EMBEDDING:
                    schema[layer_key].update({
                        "vocab_size": defaults["vocab_size"],
                        "hidden_size": defaults["hidden_size"]
                    })

                elif layer_type in {
                    LayerType.LAYERNORM,
                    LayerType.RMSNORM,
                    LayerType.BATCHNORM,
                    LayerType.GROUPNORM,
                    LayerType.INSTANCENORM
                }:
                    schema[layer_key].update({
                        "hidden_size": defaults["hidden_size"],
                        "eps": defaults["layer_norm_eps"]
                    })

                elif layer_type in {
                    LayerType.CONV1D, LayerType.CONV2D,
                    LayerType.CONV3D, LayerType.DEPTHWISE_CONV, LayerType.CONV
                }:
                    schema[layer_key].update({
                        "in_channels": defaults["hidden_size"],
                        "out_channels": defaults["hidden_size"],
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1
                    })

                elif layer_type == LayerType.CLASSIFIER:
                    schema[layer_key].update({
                        "in_features": defaults["hidden_size"],
                        "out_features": defaults["hidden_size"]
                    })

                elif layer_type == LayerType.BIAS:
                    schema[layer_key].update({
                        "out_features": defaults["hidden_size"]
                    })

                elif layer_type == LayerType.SCALING:
                    schema[layer_key].update({
                        "hidden_size": defaults["hidden_size"]
                    })

        schema["_metadata"] = defaults
        return schema
