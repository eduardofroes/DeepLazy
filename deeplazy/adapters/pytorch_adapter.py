import torch
import torch.nn as nn
from adapters.base_adapter import LayerAdapter


class PyTorchAdapter(LayerAdapter):
    def build_empty_layer(self, layer_type, config):
        if layer_type == "Linear":
            return nn.Linear(config["in_features"], config["out_features"])

        elif layer_type == "Embedding":
            return nn.Embedding(config["vocab_size"], config["hidden_size"])

        elif layer_type == "LayerNorm":
            return nn.LayerNorm(config["hidden_size"], eps=config.get("eps", 1e-5))

        elif layer_type == "RMSNorm":
            return nn.LayerNorm(config["hidden_size"], eps=config.get("eps", 1e-6), elementwise_affine=False)

        elif layer_type == "BatchNorm":
            return nn.BatchNorm1d(config["hidden_size"], eps=config.get("eps", 1e-5))

        elif layer_type in ["Attention", "MultiheadAttention", "SelfAttention", "CrossAttention"]:
            return nn.MultiheadAttention(
                embed_dim=config["hidden_size"],
                num_heads=config["num_heads"],
                batch_first=True
            )

        elif layer_type == "Conv1D":
            return nn.Conv1d(
                in_channels=config["in_channels"],
                out_channels=config["out_channels"],
                kernel_size=config["kernel_size"]
            )

        elif layer_type == "Conv2D":
            return nn.Conv2d(
                in_channels=config["in_channels"],
                out_channels=config["out_channels"],
                kernel_size=config["kernel_size"]
            )

        elif layer_type == "Activation":
            act = config.get("activation_function", "gelu").lower()
            acts = {
                "gelu": nn.GELU(),
                "relu": nn.ReLU(),
                "silu": nn.SiLU(),
                "tanh": nn.Tanh(),
                "softmax": nn.Softmax(dim=-1)
            }
            return acts.get(act, nn.GELU())

        elif layer_type == "Dropout":
            return nn.Dropout(config.get("dropout_prob", 0.1))

        elif layer_type == "Classifier":
            return nn.Linear(config["in_features"], config["out_features"])

        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

    def load_weights(self, layer, weights):
        tensor_weights = {
            k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v
            for k, v in weights.items()
        }
        layer.load_state_dict(tensor_weights, strict=False)
