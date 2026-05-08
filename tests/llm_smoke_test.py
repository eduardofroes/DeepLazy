"""Smoke test: GPT-2 inference under DeepLazy vs eager loading.

Run manually with:

    python3 tests/llm_smoke_test.py

The test downloads GPT-2 once into ``.cache/gpt2`` (gitignored) and
runs the same prompt twice — once with the lazy loader, once with the
standard ``AutoModelForCausalLM`` — printing peak resident memory and
wall-clock time for each path so the impact of lazy loading is
visible.
"""

from __future__ import annotations

import gc
import os
import sys
import time

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

from deeplazy.core.lazy_cache import PytorchLocalLRUCache  # noqa: E402
from deeplazy.core.lazy_model import LazyModel  # noqa: E402
from deeplazy.core.lazy_tensor_loader import LazyLoader  # noqa: E402
from deeplazy.enums.framework_enum import FrameworkType  # noqa: E402


WEIGHTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".cache", "gpt2"))
PROMPT = "The future of artificial intelligence is"
MAX_NEW_TOKENS = 20


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def _print_stage(stage: str, t_start: float, mem_start: float):
    elapsed = time.perf_counter() - t_start
    delta_mem = _rss_mb() - mem_start
    print(f"  [{stage:<14}] +{elapsed:6.2f}s  rss={_rss_mb():7.1f}MB "
          f"(Δ={delta_mem:+7.1f}MB)")


# ---------------------------------------------------------------------------
# Lazy path
# ---------------------------------------------------------------------------

def run_lazy(tokenizer, prompt: str):
    print("\n=== LAZY (DeepLazy) ===")
    gc.collect()
    mem0 = _rss_mb()
    t0 = time.perf_counter()

    loader = LazyLoader(
        weights_dir=WEIGHTS_DIR,
        device="cpu",
        cache_backend=PytorchLocalLRUCache(capacity=4),
        framework=FrameworkType.PYTORCH,
        enable_prefetch=True,
    )
    _print_stage("loader built", t0, mem0)

    lazy = LazyModel(cls=AutoModelForCausalLM, loader=loader)
    model = lazy.model
    model.eval()
    _print_stage("model wrapped", t0, mem0)

    inputs = tokenizer(prompt, return_tensors="pt")
    peak = _rss_mb()
    with torch.inference_mode():
        for n in range(MAX_NEW_TOKENS):
            outputs = model(**inputs)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            inputs = {
                "input_ids": torch.cat([inputs["input_ids"], next_token], dim=1)
            }
            peak = max(peak, _rss_mb())
    _print_stage("generation done", t0, mem0)

    text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    print(f"  peak rss during generation: {peak:.1f} MB")
    print(f"  output: {text!r}")

    loader.close()
    del lazy, model, loader
    gc.collect()
    return text, peak


# ---------------------------------------------------------------------------
# Eager path (baseline)
# ---------------------------------------------------------------------------

def run_eager(tokenizer, prompt: str):
    print("\n=== EAGER (standard from_pretrained) ===")
    gc.collect()
    mem0 = _rss_mb()
    t0 = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(WEIGHTS_DIR)
    model.eval()
    _print_stage("model loaded", t0, mem0)

    inputs = tokenizer(prompt, return_tensors="pt")
    peak = _rss_mb()
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        peak = max(peak, _rss_mb())
    _print_stage("generation done", t0, mem0)

    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    print(f"  peak rss during generation: {peak:.1f} MB")
    print(f"  output: {text!r}")

    del model
    gc.collect()
    return text, peak


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.isdir(WEIGHTS_DIR) or not os.path.isfile(
            os.path.join(WEIGHTS_DIR, "model.safetensors")):
        sys.exit(
            f"weights not found in {WEIGHTS_DIR} — run snapshot_download first")

    print(f"weights dir: {WEIGHTS_DIR}")
    print(f"prompt:      {PROMPT!r}")
    print(f"max tokens:  {MAX_NEW_TOKENS}")
    print(f"baseline rss before any model load: {_rss_mb():.1f} MB")

    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)

    eager_text, eager_peak = run_eager(tokenizer, PROMPT)
    lazy_text, lazy_peak = run_lazy(tokenizer, PROMPT)

    print("\n=== summary ===")
    print(f"eager peak rss : {eager_peak:7.1f} MB")
    print(f"lazy  peak rss : {lazy_peak:7.1f} MB")
    print(f"savings        : {eager_peak - lazy_peak:+7.1f} MB "
          f"({(1 - lazy_peak / eager_peak) * 100:+.1f}%)")


if __name__ == "__main__":
    main()
