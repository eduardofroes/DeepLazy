import torch
import torch.nn as nn
from functools import wraps
import threading


class PytorchLazyModelPatcher:
    """Patches a meta-built PyTorch model so each module loads its
    weights on-demand.

    Key performance properties:
      * The LRU cache keeps hot layers warm — modules are NOT
        force-unloaded after every forward.
      * Tensor dtype is preserved (no implicit upcast to float32).
      * Forward is wrapped in ``torch.inference_mode`` to skip autograd.
      * Trace-driven prefetch: after the first forward pass, each module
        triggers a background load of the next module before its own
        forward starts, hiding I/O behind compute.
    """

    def __init__(self, loader, is_tied=False):
        self.loader = loader
        self.modules_by_name = {}
        self.base_model_prefix = None
        self.is_tied = is_tied
        # Trace of module execution order across forwards.
        # Maps a module's normalized name to the next normalized name
        # observed after it. Used for one-step lookahead prefetch.
        self._next_module = {}
        self._last_seen_module = None
        self._trace_lock = threading.Lock()

    def patch(self, model):
        self.base_model_prefix = getattr(model, 'base_model_prefix', None)
        self._annotate_module_names(model)
        self._patch_module_instances()
        return model

    def _annotate_module_names(self, model):
        for name, module in model.named_modules():
            module._lazy_full_name = name
            normalized = name
            if self.base_model_prefix and name.startswith(f"{self.base_model_prefix}."):
                normalized = name[len(self.base_model_prefix) + 1:]
            module._lazy_normalized_name = normalized
            self.modules_by_name[name] = module

    def _patch_module_instances(self):
        for name, module in self.modules_by_name.items():
            if hasattr(module, '_lazy_wrapped'):
                continue
            if isinstance(module, nn.Module) and not isinstance(module, nn.Sequential):
                self._wrap_instance_forward(module)

    def _record_trace(self, normalized_name):
        with self._trace_lock:
            prev = self._last_seen_module
            if prev is not None and prev != normalized_name:
                self._next_module[prev] = normalized_name
            self._last_seen_module = normalized_name
            return self._next_module.get(normalized_name)

    def _ensure_loaded(self, normalized_name, loader_ref):
        if normalized_name == "lm_head" and loader_ref.cache.get("lm_head") is None:
            if self.is_tied:
                if loader_ref.cache.get("wte") is None:
                    loader_ref.load_module("wte", self.base_model_prefix)
                wte = loader_ref.cache.get("wte")
                if wte is not None:
                    # GPT-2 style: lm_head shares the embedding weights.
                    loader_ref.cache.put("lm_head", wte)
                    return
                # Fall through: model claims tied weights but "wte" doesn't
                # exist under that name (e.g. LLaMA uses "embed_tokens").
            loader_ref.load_module("lm_head", self.base_model_prefix)

        if loader_ref.cache.get(normalized_name) is None:
            loader_ref.load_module(normalized_name, self.base_model_prefix)

    def _wrap_instance_forward(self, module):
        orig_forward = module.forward
        loader_ref = self.loader
        module._lazy_wrapped = True
        patcher = self

        @wraps(orig_forward)
        def wrapped_forward(*args, **kwargs):
            full_name = getattr(module, '_lazy_full_name', '')
            normalized_name = getattr(
                module, '_lazy_normalized_name', full_name)

            # Record execution trace and pull the next-module hint.
            next_name = patcher._record_trace(normalized_name)

            patcher._ensure_loaded(normalized_name, loader_ref)

            # Prefetch the next module concurrently with this forward.
            if next_name is not None:
                loader_ref.prefetch_module(next_name, patcher.base_model_prefix)

            module_weights = loader_ref.cache.get(normalized_name)
            original_attrs = []

            # ``to_empty`` moves meta-device parameters to real (CPU/CUDA)
            # storage, but it allocates heap memory for EVERY parameter and
            # then we immediately overwrite the ones we have weights for via
            # ``register_buffer``.  Those ephemeral allocations accumulate in
            # the allocator pool and inflate RSS.
            #
            # Optimised path: only call ``to_empty`` for meta-device params
            # that are NOT covered by ``module_weights`` (rare).  For params
            # that we WILL replace, go straight to ``register_buffer`` — the
            # meta tensor is removed from ``_parameters`` there, no interim
            # heap allocation needed.
            if module_weights is not None:
                covered = {pn.split('.')[-1] for pn in module_weights}
                meta_uncovered = any(
                    p is not None and p.device.type == 'meta' and n not in covered
                    for n, p in module._parameters.items()
                )
                if meta_uncovered:
                    module.to_empty(device=loader_ref.device)
            else:
                module.to_empty(device=loader_ref.device)

            if module_weights is not None:
                for param_name, tensor in module_weights.items():
                    # Skip INT8 scale sentinels — consumed below alongside
                    # their companion int8 tensor.
                    if param_name.endswith('.__scale__'):
                        continue

                    name_parts = param_name.split('.')
                    target = module
                    name = name_parts[-1]

                    # INT8 dequantization: tensor is int8, scale is stored
                    # under the sentinel key "<param>.__scale__".
                    scale_key = param_name + '.__scale__'
                    if (tensor.dtype == torch.int8
                            and scale_key in module_weights):
                        scale = module_weights[scale_key]
                        tensor = tensor.to(torch.float32).mul(scale)
                        # Cast to the expected compute dtype (bfloat16 or fp16)
                        if not loader_ref.preserve_dtype:
                            tensor = tensor.to(torch.float32)
                        else:
                            tensor = tensor.to(torch.bfloat16)

                    if hasattr(target, name):
                        original_attrs.append(name)
                        for attr_dict in [target._parameters, target._buffers, target.__dict__]:
                            if name in attr_dict:
                                attr_dict.pop(name, None)
                        try:
                            delattr(target, name)
                        except Exception:
                            pass

                    # Move only — preserve original dtype unless caller
                    # explicitly opted out (preserve_dtype=False forces
                    # legacy upcast to float32).
                    if loader_ref.preserve_dtype:
                        tensor = tensor.to(device=loader_ref.device)
                    else:
                        tensor = tensor.to(
                            dtype=torch.float32, device=loader_ref.device)
                    target.register_buffer(name, tensor)

                if hasattr(module, 'masked_bias') and hasattr(module, 'bias'):
                    module.masked_bias = module.bias

            with torch.inference_mode():
                result = orig_forward(*args, **kwargs)

            for name in original_attrs:
                try:
                    delattr(module, name)
                except Exception:
                    pass
                setattr(module, name, None)

            # Eviction is the LRU cache's responsibility.
            # No torch.cuda.empty_cache() per forward — that's a
            # driver-level sync that wrecks throughput.
            return result

        if hasattr(module, 'weight') or hasattr(module, 'bias') or hasattr(module, 'masked_bias'):
            module.forward = wrapped_forward
