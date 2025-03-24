from enums.layer_type_enum import LayerType
from enums.framework_enum import FrameworkType


class LazyLayer:
    def __init__(self, layer_type: LayerType, adapter, tensor_loader, keys, config=None, framework: FrameworkType = FrameworkType.TORCH):
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

    def load_weights(self):
        if self.layer is None:
            self.layer = self.adapter.build_empty_layer(
                self.layer_type.value, self.config)
            weights = {key: self.tensor_loader.load(key) for key in self.keys}
            self.adapter.load_weights(self.layer, weights)
            self.is_built = True
            del weights

    async def async_build_layer_from_weights(self, weights):
        self.layer = self.adapter.build_empty_layer(
            self.layer_type.value, self.config)
        self.adapter.load_weights(self.layer, weights)
        self.is_built = True

    def unload(self):
        if self.layer is not None:
            del self.layer
            self.layer = None
        self.is_built = False

    def forward(self, x):
        if not self.is_built:
            raise RuntimeError("Layer was not built before forward pass.")

        if self.framework == FrameworkType.TORCH:
            import torch
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
        return self.forward(x)
