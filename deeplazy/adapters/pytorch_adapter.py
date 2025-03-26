import torch
import torch.nn as nn
from adapters.base_adapter import LayerAdapter


class PositionalEmbeddingWrapper(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(vocab_size, hidden_size))

    def forward(self, x):
        return self.weight[:x.size(1)].unsqueeze(0)


class MLPWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        params = config.get("params", {})

        # Inferir c_fc
        c_fc_weight_shape = next(
            (v for k, v in params.items() if "c_fc.weight" in k), None)
        if c_fc_weight_shape is None:
            raise ValueError("Missing c_fc.weight in MLP config")

        c_fc_in, c_fc_out = c_fc_weight_shape[1], c_fc_weight_shape[0]
        self.c_fc = nn.Linear(c_fc_in, c_fc_out)

        # Inferir c_proj
        c_proj_weight_shape = next(
            (v for k, v in params.items() if "c_proj.weight" in k), None)
        if c_proj_weight_shape is None:
            raise ValueError("Missing c_proj.weight in MLP config")

        c_proj_in, c_proj_out = c_proj_weight_shape[1], c_proj_weight_shape[0]
        self.c_proj = nn.Linear(c_proj_in, c_proj_out)

    def forward(self, x):
        return self.c_proj(torch.nn.functional.gelu(self.c_fc(x)))


class PyTorchAdapter(LayerAdapter):
    def build_empty_layer(self, layer_type, config):
        params = config.get("params", {})

        def get_shape_by_suffix(suffix):
            for k, v in params.items():
                if k.endswith(suffix):
                    return v
            return None

        if (
            "c_fc.weight" in params.keys() and "c_proj.weight" in params.keys()
            and layer_type == "Linear"
        ):
            return MLPWrapper(config)

        if layer_type in ["Linear", "Classifier", "ClassifierFinal"]:
            if any(k.endswith("c_fc.weight") for k in params.keys()) and any(k.endswith("c_proj.weight") for k in params.keys()):
                return MLPWrapper(config)

            shape = get_shape_by_suffix("weight")
            if shape[0] > shape[1]:
                in_features, out_features = shape[1], shape[0]
            else:
                in_features, out_features = shape[0], shape[1]
            return nn.Linear(in_features, out_features)

        elif layer_type == "Embedding":
            shape = get_shape_by_suffix("weight")
            vocab_size, hidden_size = shape[0], shape[1]
            return nn.Embedding(vocab_size, hidden_size)

        elif layer_type == "Bias":
            shape = get_shape_by_suffix("bias")
            return nn.Parameter(torch.zeros(shape[0]))

        elif layer_type == "PositionalEmbedding":
            shape = get_shape_by_suffix("weight")
            vocab_size, hidden_size = shape[0], shape[1]
            return PositionalEmbeddingWrapper(vocab_size, hidden_size)

        elif layer_type in ["LayerNorm", "LayerNormFinal", "RMSNorm", "BatchNorm"]:
            shape = get_shape_by_suffix(
                "weight") or get_shape_by_suffix("bias")
            if shape is None:
                raise ValueError(
                    "Could not infer shape for normalization layer.")
            normalized_shape = shape[0] if len(shape) == 1 else shape[-1]
            if layer_type == "RMSNorm":
                return nn.LayerNorm(normalized_shape, eps=config.get("eps", 1e-6), elementwise_affine=False)
            elif layer_type == "BatchNorm":
                return nn.BatchNorm1d(normalized_shape, eps=config.get("eps", 1e-5))
            else:
                return nn.LayerNorm(normalized_shape, eps=config.get("eps", 1e-5))

        elif layer_type in ["Attention", "MultiheadAttention", "SelfAttention", "CrossAttention"]:
            shape = get_shape_by_suffix("c_attn.weight")
            if shape is None:
                raise ValueError(
                    "Missing c_attn.weight for MultiheadAttention")
            qkv_dim = shape[1]
            embed_dim = qkv_dim // 3
            num_heads = config.get("num_heads", 12)
            return nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        elif layer_type == "Conv1D":
            shape = get_shape_by_suffix("weight")
            out_channels, in_channels, kernel_size = shape
            return nn.Conv1d(in_channels, out_channels, kernel_size)

        elif layer_type == "Conv2D":
            shape = get_shape_by_suffix("weight")
            out_channels, in_channels, kh, kw = shape
            return nn.Conv2d(in_channels, out_channels, (kh, kw))

        elif layer_type == "Dropout":
            return nn.Dropout(config.get("dropout_prob", 0.1))

        elif layer_type == "Pooling":
            kernel_size = config.get("kernel_size", 2)
            return nn.MaxPool2d(kernel_size=kernel_size)

        elif layer_type == "Scaling":
            shape = get_shape_by_suffix("weight")
            return nn.Parameter(torch.ones(shape[0]))

        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

    def load_weights(self, layer, weights, config=None):
        tensor_weights = {
            k.split('.')[-1] if '.' in k else k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v
            for k, v in weights.items()
        }

        if isinstance(layer, PositionalEmbeddingWrapper):
            weight_tensor = tensor_weights.get(
                "weight", list(tensor_weights.values())[0])
            layer.weight = nn.Parameter(weight_tensor)

        elif isinstance(layer, nn.Parameter):
            layer.data = list(tensor_weights.values())[0]

        elif isinstance(layer, MLPWrapper):
            # Carrega pesos manualmente para subcamadas
            for name in ["c_fc", "c_proj"]:
                submodule = getattr(layer, name)
                prefix = name + "."
                sub_weights = {
                    k.replace(prefix, ""): torch.tensor(v) if not isinstance(v, torch.Tensor) else v
                    for k, v in weights.items()
                    if prefix in k
                }
                try:
                    submodule.load_state_dict(sub_weights, strict=False)
                except Exception as e:
                    print(
                        f"[Warning] Failed to load weights for {name} in MLPWrapper: {e}")
                    print(
                        f"  Expected keys: {list(submodule.state_dict().keys())}")
                    print(f"  Provided keys: {list(sub_weights.keys())}")

        elif isinstance(layer, nn.Module):
            try:
                layer.load_state_dict(tensor_weights, strict=False)
            except Exception as e:
                print(
                    f"[Warning] Failed to load weights for {layer.__class__.__name__}: {e} - {config}")
                print(f"Expected keys: {list(layer.state_dict().keys())}")
                print(f"Provided keys: {list(tensor_weights.keys())}")
