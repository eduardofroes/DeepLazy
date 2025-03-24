from abc import ABC, abstractmethod


class LayerAdapter(ABC):
    @abstractmethod
    def build_empty_layer(self, layer_name: str):
        pass

    @abstractmethod
    def load_weights(self, layer, weights: dict):
        pass
