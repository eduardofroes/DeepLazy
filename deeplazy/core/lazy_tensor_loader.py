import gc
import json
import mmap as _mmap
import os
import struct
import threading
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional

import numpy as np

from deeplazy.enums.framework_enum import FrameworkType


_INDEX_UNSET = object()


class MmapSafetensorsFile:
    """Zero-copy, mmap-backed safetensors reader with MADV_DONTNEED support.

    Tensors are numpy views directly into the file's mmap — no per-tensor
    heap allocation on ``get_tensor``.  Pages are brought into process RSS
    only when the CPU actually touches them (e.g. during a forward pass).

    After evicting a module from the LRU cache, call ``advise_dontneed``
    on each of its tensor keys.  This immediately releases the pages from
    *process* RSS while keeping them in the *kernel page cache* — so the
    next re-access is a cheap page-table update (no disk I/O under normal
    conditions) rather than a full disk read.
    """

    _NP_DTYPE: dict = {
        'F16': np.float16, 'F32': np.float32, 'F64': np.float64,
        'I8': np.int8,  'I16': np.int16,  'I32': np.int32,  'I64': np.int64,
        'U8': np.uint8, 'U16': np.uint16, 'U32': np.uint32, 'U64': np.uint64,
        'BOOL': np.bool_,
    }
    _TORCH_DTYPE: dict = {}  # filled after torch import in get_tensor

    def __init__(self, path: str) -> None:
        self.path = path
        self._file = open(path, 'rb')
        hdr_len = struct.unpack('<Q', self._file.read(8))[0]
        self._header: dict = json.loads(self._file.read(hdr_len))
        self._data_start: int = 8 + hdr_len
        self._mm = _mmap.mmap(self._file.fileno(), 0, access=_mmap.ACCESS_READ)
        self._page: int = _mmap.PAGESIZE
        self._keys: list = [k for k in self._header if k != '__metadata__']
        self._madv_ok: bool = (
            hasattr(_mmap, 'MADV_DONTNEED') and hasattr(self._mm, 'madvise'))

    def keys(self) -> list:
        return self._keys

    def get_tensor(self, key: str):
        import torch
        if not MmapSafetensorsFile._TORCH_DTYPE:
            MmapSafetensorsFile._TORCH_DTYPE = {
                'F16': torch.float16, 'BF16': torch.bfloat16,
                'F32': torch.float32, 'F64': torch.float64,
                'I8': torch.int8,  'I16': torch.int16,
                'I32': torch.int32, 'I64': torch.int64,
                'U8': torch.uint8,  'BOOL': torch.bool,
            }

        meta = self._header[key]
        dtype_str: str = meta['dtype']
        shape: tuple = tuple(meta['shape'])
        start, end = meta['data_offsets']
        abs_start = self._data_start + start
        nbytes = end - start

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            if dtype_str == 'BF16':
                np_arr = np.frombuffer(self._mm, dtype=np.uint16,
                                       offset=abs_start, count=nbytes // 2)
                t = torch.from_numpy(np_arr).view(torch.bfloat16)
            else:
                np_dtype = self._NP_DTYPE[dtype_str]
                item = np.dtype(np_dtype).itemsize
                np_arr = np.frombuffer(self._mm, dtype=np_dtype,
                                       offset=abs_start, count=nbytes // item)
                t = torch.from_numpy(np_arr)

        return t.reshape(shape) if shape else t

    def _aligned_region(self, key: str):
        meta = self._header[key]
        start, end = meta['data_offsets']
        abs_start = self._data_start + start
        abs_end = self._data_start + end
        pg = self._page
        al_start = (abs_start // pg) * pg
        al_len = ((abs_end + pg - 1) // pg) * pg - al_start
        return al_start, al_len

    def advise_dontneed(self, key: str) -> None:
        """Release process RSS for this tensor's pages; kernel cache keeps them."""
        if not self._madv_ok or self._mm.closed:
            return
        al_start, al_len = self._aligned_region(key)
        if al_len > 0:
            try:
                self._mm.madvise(_mmap.MADV_DONTNEED, al_start, al_len)
            except OSError:
                pass

    def advise_willneed(self, key: str) -> None:
        """Ask the kernel to pre-load this tensor's pages into page cache.

        This is the prefetch primitive: zero memory cost to the process
        (pages land in the *kernel* page cache, not process RSS), and when
        ``get_tensor`` is later called the data is already resident — no
        disk I/O, just a page-table update (~microseconds).
        """
        if self._mm.closed or not hasattr(_mmap, 'MADV_WILLNEED'):
            return
        if not hasattr(self._mm, 'madvise'):
            return
        al_start, al_len = self._aligned_region(key)
        if al_len > 0:
            try:
                self._mm.madvise(_mmap.MADV_WILLNEED, al_start, al_len)
            except OSError:
                pass

    def close(self) -> None:
        if not self._mm.closed:
            try:
                self._mm.close()
            except Exception:
                pass
        if not self._file.closed:
            self._file.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class LazyLoader:
    """Lazy weight loader.

    Performance characteristics:
      * File handlers are opened once per loader lifetime — never reopened
        per forward.
      * A precomputed index maps every module name (and every dotted
        prefix of every weight key) to its weight entries, so
        ``load_module`` is O(weights_in_module) instead of O(total_keys).
      * Optional background prefetching overlaps weight I/O with the
        forward of the previous module.
    """

    def __init__(self, weights_dir: str, device='cpu', cache_backend=None,
                 enable_monitor=False, model_name=None,
                 framework=FrameworkType.PYTORCH,
                 preserve_dtype: bool = True,
                 enable_prefetch: bool = True,
                 use_mmap: bool = True,
                 quantize: str | None = None):
        """
        ``use_mmap`` (default True): use memory-mapped I/O for safetensors
        files.  Tensors are zero-copy views into the file's mmap; when a
        module is evicted from the LRU cache the backing pages are
        immediately released from process RSS via ``madvise(MADV_DONTNEED)``,
        while remaining in the kernel page cache for fast re-access.
        Set to False to fall back to the legacy ``safetensors.safe_open``
        path (copies tensors onto the heap).

        ``quantize`` (default None): on-the-fly quantization applied at
        load time, before the tensor is stored in the LRU cache.
        Accepted values:

          * ``None``    — no quantization (default, exact dtype from file).
          * ``"int8"``  — dynamic INT8: scale = max(|w|)/127, tensor stored
                          as torch.int8 alongside its per-tensor scale.
                          Memory: ~50% of BF16/FP16. Accuracy: near-lossless
                          for most transformer weight matrices.
          * ``"fp16"``  — cast float32 weights to float16 (no-op for BF16).
                          Memory: 50% of FP32 models, same for BF16/FP16.

        When ``quantize="int8"`` the cache stores
        ``{"<param>": tensor_int8, "<param>.__scale__": scale_f32}`` and
        the patcher dequantizes transparently before each forward pass.
        """
        self.framework = framework
        self.cache = cache_backend
        self.monitor = None
        self.weights_dir = weights_dir
        self.preserve_dtype = preserve_dtype
        self.enable_prefetch = enable_prefetch
        self.use_mmap = use_mmap

        if quantize not in (None, 'int8', 'fp16'):
            raise ValueError(f"quantize must be None, 'int8', or 'fp16', got {quantize!r}")
        self.quantize = quantize

        self.weights_paths = [
            os.path.join(weights_dir, f)
            for f in os.listdir(weights_dir)
            if f.endswith('.safetensors')
        ]
        self.weights_format = 'safetensors'

        if not self.weights_paths:
            if self.framework == FrameworkType.PYTORCH:
                self.weights_paths = [
                    os.path.join(weights_dir, f)
                    for f in os.listdir(weights_dir)
                    if f.endswith('.pth')
                ]
                self.weights_format = 'pth'
            elif self.framework == FrameworkType.TENSORFLOW:
                self.weights_paths = [
                    os.path.join(weights_dir, f)
                    for f in os.listdir(weights_dir)
                    if f.endswith('.ckpt') or f.endswith('.h5')
                ]
                if self.weights_paths:
                    if self.weights_paths[0].endswith('.h5'):
                        self.weights_format = 'h5'
                    else:
                        self.weights_format = 'ckpt'

        if not self.weights_paths:
            raise FileNotFoundError(
                f"No supported weight files found in {weights_dir}")

        self.is_safetensors = self.weights_format == 'safetensors'
        self.file_handlers = []
        self.key_to_handler = {}
        self._handlers_initialized = False

        # module_name -> list of (short_key, original_key, handler)
        # Built lazily on the first load_module call for a given prefix.
        # ``_INDEX_UNSET`` is a sentinel separate from ``None`` because
        # ``None`` is a legitimate value of ``base_model_prefix``.
        self._module_index = {}
        self._index_built_for_prefix = _INDEX_UNSET

        # Concurrency primitives for prefetching.
        self._index_lock = threading.Lock()
        self._inflight_lock = threading.Lock()
        self._inflight: dict[str, Future] = {}
        self._executor: Optional[ThreadPoolExecutor] = None

        if enable_monitor:
            from deeplazy.ui.dashboard_monitor import DashboardMonitor
            capacity = getattr(cache_backend, 'capacity', 0)
            cache_type = cache_backend.__class__.__name__ if cache_backend else None
            self.monitor = DashboardMonitor(
                model_name=model_name,
                safetensors_path=self.weights_paths,
                framework=framework.value,
                cache_type=cache_type,
                max_layers_in_memory=capacity
            )
            self.monitor.enable()

        if self.framework == FrameworkType.PYTORCH:
            import torch
            self.device = torch.device(device)
        elif self.framework == FrameworkType.TENSORFLOW:
            self.device = device
        else:
            self.device = device

        # Wire up the LRU eviction callback so that madvise(MADV_DONTNEED)
        # is called automatically whenever a module leaves the cache.
        if (self.use_mmap and self.is_safetensors
                and cache_backend is not None
                and hasattr(cache_backend, 'on_evict')):
            cache_backend.on_evict = self._on_module_evicted

    # ------------------------------------------------------------------
    # File handlers
    # ------------------------------------------------------------------
    def _init_file_handlers(self):
        if self._handlers_initialized:
            return

        with self._index_lock:
            if self._handlers_initialized:
                return

            # Imported lazily so that test suites which monkeypatch
            # ``sys.modules['safetensors']`` see their replacement, and
            # so importing deeplazy without safetensors installed for a
            # non-safetensors workflow does not fail.
            if self.weights_format == 'safetensors' and not self.use_mmap:
                from safetensors import safe_open
            for path in self.weights_paths:
                if self.weights_format == 'safetensors':
                    if self.use_mmap:
                        handler = MmapSafetensorsFile(path)
                    else:
                        from safetensors import safe_open
                        handler = safe_open(
                            path, framework=self.framework.value, device='cpu')
                    self.file_handlers.append(handler)
                    for key in handler.keys():
                        self.key_to_handler[key] = handler
                elif self.weights_format == 'pth':
                    import torch
                    state_dict = torch.load(path, map_location='cpu')
                    self.file_handlers.append(state_dict)
                    for key in state_dict.keys():
                        self.key_to_handler[key] = state_dict
                elif self.weights_format == 'ckpt':
                    import tensorflow as tf
                    reader = tf.train.load_checkpoint(path)
                    self.file_handlers.append(reader)
                    for key, _ in tf.train.list_variables(path):
                        self.key_to_handler[key] = reader
                elif self.weights_format == 'h5':
                    import h5py
                    f = h5py.File(path, 'r')
                    self.file_handlers.append(f)
                    collected = []

                    def _collect(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            collected.append(name)
                    f.visititems(_collect)
                    for key in collected:
                        self.key_to_handler[key] = f

            self._handlers_initialized = True

    # ------------------------------------------------------------------
    # Module index
    # ------------------------------------------------------------------
    def _ensure_index(self, base_model_prefix):
        """Build module_name -> [(short_key, original_key, handler)] once."""
        if self._index_built_for_prefix == base_model_prefix:
            return

        with self._index_lock:
            if self._index_built_for_prefix == base_model_prefix:
                return

            prefix = (base_model_prefix + ".") if base_model_prefix else ""
            index = defaultdict(list)
            for original_key, handler in self.key_to_handler.items():
                # Match legacy semantics of `key.replace(prefix, "")`.
                stripped = original_key.replace(
                    prefix, "") if prefix else original_key
                if not stripped:
                    continue
                parts = stripped.split('.')
                # Every dotted prefix of `stripped` is a candidate
                # module name. We bucket the short_key under each one.
                for i in range(1, len(parts)):
                    module_name = '.'.join(parts[:i])
                    short_key = original_key[len(module_name):]
                    index[module_name].append(
                        (short_key, original_key, handler))

            self._module_index = dict(index)
            self._index_built_for_prefix = base_model_prefix

    # ------------------------------------------------------------------
    # mmap eviction callback
    # ------------------------------------------------------------------
    def _on_module_evicted(self, module_name: str) -> None:
        """Called by the LRU cache when a module is evicted.

        Issues ``madvise(MADV_DONTNEED)`` for every tensor that belonged
        to the evicted module, releasing their pages from process RSS.
        Pages stay in the kernel page cache so re-access is fast.
        """
        entries = self._module_index.get(module_name, ())
        for _short_key, original_key, handler in entries:
            if isinstance(handler, MmapSafetensorsFile):
                handler.advise_dontneed(original_key)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _materialize(self, original_key, handler):
        if self.weights_format in ('safetensors', 'ckpt'):
            tensor = handler.get_tensor(original_key)
        elif self.weights_format == 'pth':
            tensor = handler[original_key]
        elif self.weights_format == 'h5':
            tensor = handler[original_key][()]
        else:
            return None

        if self.framework == FrameworkType.TENSORFLOW:
            import tensorflow as tf
            tensor = tf.convert_to_tensor(tensor)
            return tensor

        # ── on-the-fly quantization (PyTorch only) ───────────────────
        # After quantization the result is a new heap tensor; the mmap
        # view that ``get_tensor`` returned is no longer referenced so
        # we issue DONTNEED immediately to release those pages.
        if self.quantize == 'fp16' and hasattr(tensor, 'to'):
            import torch
            if tensor.dtype == torch.float32:
                tensor = tensor.to(torch.float16)
                if isinstance(handler, MmapSafetensorsFile):
                    handler.advise_dontneed(original_key)

        elif self.quantize == 'int8' and hasattr(tensor, 'to'):
            import torch
            if tensor.dtype in (torch.float16, torch.bfloat16, torch.float32):
                # Memory-efficient path: compute scale from the original dtype
                # (no full float32 copy) — abs() and max() stay in native dtype,
                # only the scalar result is promoted to float32.
                scale = tensor.abs().float().max().clamp(min=1e-8) / 127.0
                # Quantize in native dtype to avoid a large intermediate copy.
                quantized = (tensor / scale.to(tensor.dtype)).round() \
                                .clamp(-128, 127).to(torch.int8)
                if isinstance(handler, MmapSafetensorsFile):
                    handler.advise_dontneed(original_key)
                return (quantized, scale)

        return tensor

    def _do_load(self, module_name, base_model_prefix):
        self._init_file_handlers()
        self._ensure_index(base_model_prefix)

        if self.cache and self.cache.get(module_name) is not None:
            return

        start_time = time.time()
        entries = self._module_index.get(module_name, ())
        module_weights = {}
        for short_key, original_key, handler in entries:
            if short_key in module_weights:
                continue
            result = self._materialize(original_key, handler)
            if result is None:
                continue
            # INT8 quantization returns (int8_tensor, scale) — unpack and
            # store scale under a sentinel key so the patcher can find it.
            if isinstance(result, tuple) and len(result) == 2:
                q_tensor, scale = result
                module_weights[short_key] = q_tensor
                module_weights[short_key + '.__scale__'] = scale
            else:
                module_weights[short_key] = result

        if module_weights and self.cache:
            self.cache.put(module_name, module_weights)

        if self.monitor:
            exec_time = time.time() - start_time
            self.monitor.record_layer(module_name, exec_time)

    def load_module(self, module_name, base_model_prefix=None):
        # If a prefetch for this module is in flight, block on it instead
        # of starting a duplicate load.
        future = None
        with self._inflight_lock:
            future = self._inflight.get(module_name)

        if future is not None:
            future.result()
            return

        self._do_load(module_name, base_model_prefix)

    def prefetch_module(self, module_name, base_model_prefix=None) -> Optional[Future]:
        """Prefetch a module's weights before it is needed.

        mmap mode (``use_mmap=True``):
            Issues ``madvise(MADV_WILLNEED)`` for every tensor page of the
            module.  The kernel starts loading them into its page cache in
            the background — zero process-RSS cost.  When ``load_module``
            is later called the mmap ``get_tensor`` hits the warm page
            cache instead of disk (~µs vs ~ms).

        legacy mode (``use_mmap=False``):
            Submits a thread-pool task that calls ``_do_load``, placing
            the tensor into the LRU cache (original behaviour).
        """
        if not self.enable_prefetch:
            return None
        if self.cache and self.cache.get(module_name) is not None:
            return None

        # ── mmap path: WILLNEED is a cheap syscall, no LRU slot used ──
        if self.use_mmap and self.is_safetensors:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix='deeplazy-prefetch')

            def _willneed_task():
                self._ensure_index(base_model_prefix)
                for _, original_key, handler in self._module_index.get(module_name, ()):
                    if isinstance(handler, MmapSafetensorsFile):
                        handler.advise_willneed(original_key)

            return self._executor.submit(_willneed_task)

        # ── legacy path: load into LRU cache ──────────────────────────
        with self._inflight_lock:
            if module_name in self._inflight:
                return self._inflight[module_name]

            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix='deeplazy-prefetch')

            future = self._executor.submit(
                self._do_load, module_name, base_model_prefix)
            self._inflight[module_name] = future

        # ``add_done_callback`` is registered OUTSIDE the lock: if the
        # task is already finished it fires inline on this thread, and
        # we'd otherwise re-enter the same non-reentrant lock.
        def _cleanup(_f, name=module_name):
            with self._inflight_lock:
                self._inflight.pop(name, None)

        future.add_done_callback(_cleanup)
        return future

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------
    def unload_module(self, module_name):
        """Evict a single module's weights from the cache.

        Note: this no longer destroys the underlying file handlers — they
        live for the lifetime of the loader. The PyTorch patcher no
        longer calls this after every forward; eviction is now the LRU
        cache's responsibility.
        """
        if self.cache:
            self.cache.pop(module_name)

    def close(self):
        """Release file handlers and the prefetch executor."""
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

        # Clear the LRU cache first so tensor references are dropped before
        # we close the mmap files they point into.
        if self.cache is not None and hasattr(self.cache, 'cache'):
            self.cache.cache.clear()
        gc.collect()

        # Close mmap handlers (safe now that the LRU no longer holds tensors).
        for handler in self.file_handlers:
            if isinstance(handler, MmapSafetensorsFile):
                try:
                    handler.close()
                except Exception:
                    pass

        self.file_handlers = []
        self.key_to_handler = {}
        self._module_index = {}
        self._handlers_initialized = False
        self._index_built_for_prefix = _INDEX_UNSET
        gc.collect()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
