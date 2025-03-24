import tensorflow as tf
from adapters.base_adapter import LayerAdapter


class TensorFlowAdapter(LayerAdapter):
    def build_empty_layer(self, layer_type, config):
        """
        Build an empty Keras layer based on the given type and configuration.
        """
        if layer_type == "Linear":
            return tf.keras.layers.Dense(config["out_features"])

        elif layer_type == "Embedding":
            return tf.keras.layers.Embedding(config["vocab_size"], config["hidden_size"])

        elif layer_type == "LayerNorm":
            return tf.keras.layers.LayerNormalization(epsilon=config.get("eps", 1e-5))

        elif layer_type == "RMSNorm":
            # RMSNorm is not natively available in Keras, approximate using LayerNormalization
            return tf.keras.layers.LayerNormalization(epsilon=config.get("eps", 1e-6))

        elif layer_type == "BatchNorm":
            return tf.keras.layers.BatchNormalization(epsilon=config.get("eps", 1e-5))

        elif layer_type == "GroupNorm":
            raise NotImplementedError(
                "GroupNorm is not natively available in Keras.")

        elif layer_type == "InstanceNorm":
            raise NotImplementedError(
                "InstanceNorm requires tf_addons.layers.InstanceNormalization.")

        elif layer_type in ["Attention", "MultiheadAttention", "SelfAttention", "CrossAttention"]:
            return tf.keras.layers.MultiHeadAttention(
                num_heads=config["num_heads"],
                key_dim=config.get(
                    "head_size", config["hidden_size"] // config["num_heads"]),
                output_shape=config["hidden_size"]
            )

        elif layer_type == "AttentionProjection":
            return tf.keras.layers.Dense(config["out_features"])

        elif layer_type == "Conv1D":
            return tf.keras.layers.Conv1D(
                filters=config["out_channels"],
                kernel_size=config["kernel_size"],
                padding="same"
            )

        elif layer_type == "Conv2D":
            return tf.keras.layers.Conv2D(
                filters=config["out_channels"],
                kernel_size=config["kernel_size"],
                padding="same"
            )

        elif layer_type == "Conv3D":
            return tf.keras.layers.Conv3D(
                filters=config["out_channels"],
                kernel_size=config["kernel_size"],
                padding="same"
            )

        elif layer_type == "Classifier":
            return tf.keras.layers.Dense(config["out_features"])

        elif layer_type == "Bias":
            return tf.keras.layers.Dense(config["out_features"], use_bias=True, activation=None)

        elif layer_type == "Scaling":
            # Scaling can be implemented using a Lambda layer with a learnable or static scale
            scale_value = config.get("scale", 1.0)
            return tf.keras.layers.Lambda(lambda x: x * scale_value)

        elif layer_type == "Activation":
            act = config.get("activation_function", "gelu").lower()
            activations = {
                "gelu": tf.keras.activations.gelu,
                "relu": tf.keras.activations.relu,
                "silu": tf.keras.activations.swish,
                "tanh": tf.keras.activations.tanh,
                "softmax": tf.keras.activations.softmax
            }
            return tf.keras.layers.Activation(activations.get(act, tf.keras.activations.gelu))

        elif layer_type == "Dropout":
            return tf.keras.layers.Dropout(config.get("dropout_prob", 0.1))

        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

    def load_weights(self, layer, weights):
        """
        Load weights into a given Keras layer.
        If weights is a dictionary, it will convert it into a list.
        """
        try:
            if isinstance(weights, dict):
                weights_list = list(weights.values())
            else:
                weights_list = weights
            layer.set_weights(weights_list)
        except Exception as e:
            raise RuntimeError(f"Error loading weights into the layer: {e}")
