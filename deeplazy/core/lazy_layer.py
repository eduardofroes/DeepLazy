import torch
import tensorflow as tf
import torch.nn as nn
from enums.layer_type_enum import LayerType
from enums.framework_enum import FrameworkType


class LazyLayer:
    def __init__(self, layer_type: LayerType, adapter, tensor_loader, keys, config=None, framework: FrameworkType = FrameworkType.TORCH, metadata: dict = None, activation_function: str = None):
        """
        :param layer_type: Type of the layer (e.g., 'Linear', 'Attention', etc.)
        :param adapter: Framework-specific adapter (PyTorch or TensorFlow).
        :param tensor_loader: Responsible for loading tensors.
        :param keys: Keys for the layer weights.
        :param config: Configuration dictionary for the layer.
        :param framework: 'torch' or 'tensorflow'
        """
        self.layer_type = layer_type
        self.adapter = adapter
        self.tensor_loader = tensor_loader
        self.keys = keys
        self.config = config or {}
        self.framework = framework
        self.layer = None
        self.is_built = False
        self.metadata = metadata
        self.activation_function = activation_function

    async def async_build_layer_from_weights(self, weights):
        self.layer = self.adapter.build_empty_layer(
            self.layer_type.value, self.config)

        self.adapter.load_weights(self.layer, weights,  self.config)

        self.is_built = True

    def unload(self):
        if self.layer is not None:
            del self.layer
            self.layer = None
        self.is_built = False

    def forward(self, x):
        if not self.is_built:
            raise RuntimeError("Layer was not built before forward pass.")

        if self.layer is None:
            raise RuntimeError("Layer is not initialized.")

        if self.framework == FrameworkType.TORCH:
            with torch.no_grad():
                if self.layer_type in [
                    LayerType.ATTENTION, LayerType.MULTIHEAD_ATTENTION,
                    LayerType.SELF_ATTENTION, LayerType.CROSS_ATTENTION
                ]:
                    x = x.float()
                    return self.layer(x, x, x)[0]
                elif self.layer_type == LayerType.EMBEDDING:
                    x = x.long()
                    return self.layer(x)
                elif self.layer_type == LayerType.POSITIONAL_EMBEDDING:
                    x = x.float()
                    return self.layer(x)
                else:
                    return self.layer(x)

        elif self.framework == FrameworkType.TENSORFLOW:
            if self.layer_type in [
                LayerType.ATTENTION, LayerType.MULTIHEAD_ATTENTION,
                LayerType.SELF_ATTENTION, LayerType.CROSS_ATTENTION
            ]:
                return self.layer(query=x, key=x, value=x)
            else:
                return self.layer(x)

        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

    def __call__(self, x):
        x = self.forward(x)
        if self.activation_function:
            activation_fn = self.get_activation_function(
                self.activation_function)
            if activation_fn is None:
                raise ValueError(
                    f"Invalid activation function: {self.activation_function}")
            x = activation_fn(x)
        return x

    def get_activation_function(self, activation_function: str):
        if self.framework == FrameworkType.TORCH:
            if activation_function == "relu":
                return nn.ReLU()
            elif activation_function == "gelu":
                return nn.GELU()
            elif activation_function == "tanh":
                return nn.Tanh()
            elif activation_function == "sigmoid":
                return nn.Sigmoid()
            elif activation_function == "silu":
                return nn.SiLU()
            elif activation_function == "swish":
                return nn.SiLU()
            elif activation_function == "gelu_new":
                return nn.GELU()
            elif activation_function == "softmax":
                return nn.Softmax(dim=-1)
            elif activation_function == "softplus":
                return nn.Softplus()
            elif activation_function == "softsign":
                return nn.Softsign()

        elif self.framework == FrameworkType.TENSORFLOW:
            if activation_function == "relu":
                return tf.nn.relu
            elif activation_function == "gelu":
                return tf.nn.gelu
            elif activation_function == "tanh":
                return tf.nn.tanh
            elif activation_function == "sigmoid":
                return tf.nn.sigmoid
            elif activation_function == "silu":
                return tf.nn.silu
            elif activation_function == "swish":
                return tf.nn.swish
            elif activation_function == "gelu_new":
                return tf.nn.gelu
            elif activation_function == "softmax":
                return tf.nn.softmax
            elif activation_function == "softplus":
                return tf.nn.softplus
            elif activation_function == "softsign":
                return tf.nn.softsign
        return None
