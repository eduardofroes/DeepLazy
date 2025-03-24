from adapters.pytorch_adapter import PyTorchAdapter
from adapters.tensorflow_adapter import TensorFlowAdapter
from core.lazy_layer import LazyLayer
from core.layer_cache import LayerCache
from core.lazy_tensor_loader import LazyTensorLoader
from core.architecture_parser import ModelArchitectureParser
from core.lazy_model import LazyModel
from core.redis_layer_cache import RedisLayerCache
from ui.dashboard_monitor import DashboardMonitor
from enums.framework_enum import FrameworkType
from enums.layer_type_enum import LayerType


class LazyModelBuilder:
    def __init__(self, framework='torch', use_cache=True, cache_type='memory', cache_size=100,
                 redis_config=None, storage=None, config_path=None, index_path=None, max_layers_in_memory=10):
        """
        :param framework: 'torch' or 'tensorflow'
        :param use_cache: Enable cache usage
        :param cache_type: 'memory' (default LayerCache) or 'redis'
        :param cache_size: Used only for memory cache
        :param redis_config: Dict with Redis params (host, port, db, prefix)
        :param max_layers_in_memory: Number of layers to load in memory at once (default 10)
        """
        self.framework = framework
        self.use_cache = use_cache
        self.max_layers_in_memory = max_layers_in_memory
        self.redis_config = redis_config

        self.storage = storage
        self.config_path = config_path
        self.index_path = index_path
        self.cache_type = cache_type
        if use_cache:
            if cache_type == 'redis':
                redis_config = redis_config or {}
                self.cache = RedisLayerCache(
                    redis_host=redis_config.get('host', 'localhost'),
                    redis_port=redis_config.get('port', 6379),
                    db=redis_config.get('db', 0),
                    prefix=redis_config.get('prefix', 'layer_cache')
                )
            else:
                self.cache = LayerCache(cache_size)
        else:
            self.cache = None

        self.adapter = self._get_adapter(framework)

    def _get_adapter(self, framework):
        adapters = {
            FrameworkType.TORCH.value: PyTorchAdapter,
            FrameworkType.TENSORFLOW.value: TensorFlowAdapter
        }
        if framework not in adapters:
            raise ValueError(f"Unsupported framework: {framework}")
        return adapters[framework]()

    def build_model(self):
        parser = ModelArchitectureParser(
            self.config_path, self.storage.get_index())
        schema = parser.get_architecture_schema()
        loader = LazyTensorLoader(
            self.storage, framework=self.framework, cache=self.cache)

        metadata = schema.pop("_metadata", {})
        layers = {}

        for layer_name, layer_config in schema.items():
            keys = [k for k in parser.tensor_index if k.startswith(
                layer_name) and not k.endswith("_scale_inv")]
            if not keys:
                continue

            try:
                layer_type_enum = LayerType(layer_config["type"])
            except ValueError:
                layer_type_enum = LayerType.UNKNOWN

            lazy_layer = LazyLayer(
                layer_type=layer_type_enum,
                adapter=self.adapter,
                tensor_loader=loader,
                keys=keys,
                config=layer_config,
                framework=FrameworkType(self.framework)
            )
            layers[layer_name] = lazy_layer

        model_name = getattr(parser.config, "_name_or_path", "LazyLLM")

        self.dashboard = DashboardMonitor(
            model_name=model_name,
            safetensors_path=self.storage.shards_dir,
            max_visible_layers=20,
            max_layers_in_memory=self.max_layers_in_memory,
            cache_type=self.cache_type,
            framework=self.framework
        )

        return LazyModel(
            layers,
            metadata=metadata,
            max_layers_in_memory=self.max_layers_in_memory,
            dashboard=self.dashboard
        )
