from transformers.models.gpt2.modeling_tf_gpt2 import TFGPT2Model
from deeplazy.core.lazy_model import LazyModel
from deeplazy.core.lazy_tensor_loader import LazyLoader
from deeplazy.core.lazy_cache import TFLRULazyCache
from deeplazy.enums.framework_enum import FrameworkType

from transformers.models.gpt2.configuration_gpt2 import GPT2Config

import tensorflow as tf

tf_loader = LazyLoader(
    weights_path=["/opt/repository/gpt2_safetensors/model.safetensors"],
    device="/CPU:0",
    cache_backend=TFLRULazyCache(capacity=4),
    enable_monitor=True,
    model_name="gpt2_tensorflow",
    framework=FrameworkType.TENSORFLOW
)


tf_model = LazyModel(config=GPT2Config(), cls=TFGPT2Model, loader=tf_loader)
tf_input = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
tf_output = tf_model(input_ids=tf_input)
print("TensorFlow GPT2 output:", tf_output.last_hidden_state.shape)
