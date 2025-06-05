import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import types
import pytest

from deeplazy.enums.framework_enum import FrameworkType


class DummyCache:
    def __init__(self):
        self.data = {}
    def get(self, key):
        return self.data.get(key)
    def put(self, key, value):
        self.data[key] = value
    def pop(self, key):
        self.data.pop(key, None)
    def keys(self):
        return list(self.data.keys())


@pytest.fixture(autouse=True)
def fake_safetensors(monkeypatch):
    class DummyHandler:
        def __init__(self, mapping):
            self.mapping = mapping
        def keys(self):
            return self.mapping.keys()
        def get_tensor(self, key):
            return self.mapping[key]
    def dummy_safe_open(path, framework=None, device=None):
        return DummyHandler({'layer.weight': 1})
    module = types.ModuleType('safetensors')
    module.safe_open = dummy_safe_open
    monkeypatch.setitem(sys.modules, 'safetensors', module)
    yield
    sys.modules.pop('safetensors', None)


@pytest.fixture
def fake_torch(monkeypatch):
    torch = types.SimpleNamespace()
    torch.device = lambda d: d
    torch.load = lambda path, map_location=None: {'layer.weight': 1}
    class Cuda:
        @staticmethod
        def is_available():
            return False
        empty_cache = staticmethod(lambda: None)
        ipc_collect = staticmethod(lambda: None)
    torch.cuda = Cuda()
    monkeypatch.setitem(sys.modules, 'torch', torch)
    yield
    sys.modules.pop('torch', None)


@pytest.fixture
def fake_tf(monkeypatch):
    tf = types.SimpleNamespace()
    tf.convert_to_tensor = lambda x: x
    class Reader:
        def get_tensor(self, key):
            return 1
    tf.train = types.SimpleNamespace(
        load_checkpoint=lambda path: Reader(),
        list_variables=lambda path: [('layer.weight', None)]
    )
    class Backend:
        clear_session = staticmethod(lambda: None)
    tf.keras = types.SimpleNamespace(backend=Backend())
    tf.config = types.SimpleNamespace(
        experimental=types.SimpleNamespace(clear_memory=lambda: None)
    )
    monkeypatch.setitem(sys.modules, 'tensorflow', tf)
    yield
    sys.modules.pop('tensorflow', None)


@pytest.fixture
def fake_h5py(monkeypatch):
    class DummyDataset:
        def __init__(self, value):
            self.value = value
        def __getitem__(self, item):
            if item == ():
                return self.value
            return self.value

    class DummyFile(dict):
        def visititems(self, func):
            for k, v in self.items():
                func(k, v)

    def File(path, mode='r'):
        return DummyFile({'layer.weight': DummyDataset(1)})

    module = types.ModuleType('h5py')
    module.File = File
    module.Dataset = DummyDataset
    monkeypatch.setitem(sys.modules, 'h5py', module)
    yield
    sys.modules.pop('h5py', None)


def test_safetensors_loading(tmp_path, fake_torch):
    weights_dir = tmp_path
    (weights_dir / 'model.safetensors').write_text('x')

    from deeplazy.core.lazy_tensor_loader import LazyLoader

    loader = LazyLoader(
        weights_dir=str(weights_dir),
        device='cpu',
        cache_backend=DummyCache(),
        framework=FrameworkType.PYTORCH
    )
    loader.load_module('layer')
    assert loader.weights_format == 'safetensors'
    assert loader.cache.get('layer')['.weight'] == 1


def test_pth_loading(tmp_path, fake_torch):
    weights_dir = tmp_path
    (weights_dir / 'model.pth').write_text('x')

    from deeplazy.core.lazy_tensor_loader import LazyLoader

    loader = LazyLoader(
        weights_dir=str(weights_dir),
        device='cpu',
        cache_backend=DummyCache(),
        framework=FrameworkType.PYTORCH
    )
    loader.load_module('layer')
    assert loader.weights_format == 'pth'
    assert loader.cache.get('layer')['.weight'] == 1


def test_ckpt_loading(tmp_path, fake_tf):
    weights_dir = tmp_path
    (weights_dir / 'model.ckpt').write_text('x')

    from deeplazy.core.lazy_tensor_loader import LazyLoader

    loader = LazyLoader(
        weights_dir=str(weights_dir),
        device='cpu',
        cache_backend=DummyCache(),
        framework=FrameworkType.TENSORFLOW
    )
    loader.load_module('layer')
    assert loader.weights_format == 'ckpt'
    assert loader.cache.get('layer')['.weight'] == 1


def test_h5_loading(tmp_path, fake_tf, fake_h5py):
    weights_dir = tmp_path
    (weights_dir / 'model.h5').write_text('x')

    from deeplazy.core.lazy_tensor_loader import LazyLoader

    loader = LazyLoader(
        weights_dir=str(weights_dir),
        device='cpu',
        cache_backend=DummyCache(),
        framework=FrameworkType.TENSORFLOW
    )
    loader.load_module('layer')
    assert loader.weights_format == 'h5'
    assert loader.cache.get('layer')['.weight'] == 1
