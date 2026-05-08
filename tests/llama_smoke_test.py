"""Smoke test: TinyLlama-1.1B inference under DeepLazy vs eager loading.

Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (~2.2 GB, fp16)
Architecture: LLaMA — a real "large" model compared to GPT-2 (117 M).

Run manually with:

    python3 tests/llama_smoke_test.py

Pass --download to fetch the model weights automatically (requires
``huggingface_hub`` installed):

    python3 tests/llama_smoke_test.py --download

The script runs the same prompt twice — once with the lazy loader,
once with the standard ``AutoModelForCausalLM`` — and prints peak
resident memory and wall-clock time for each path.
"""

from __future__ import annotations

import argparse
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
from deeplazy.core.lazy_model import LazyModel              # noqa: E402
from deeplazy.core.lazy_tensor_loader import LazyLoader     # noqa: E402
from deeplazy.enums.framework_enum import FrameworkType     # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
WEIGHTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".cache", "tinyllama-1.1b"))

PROMPT = (
    "<|system|>\nYou are a helpful assistant.</s>\n"
    "<|user|>\nWhat is the future of artificial intelligence?</s>\n"
    "<|assistant|>\n"
)
MAX_NEW_TOKENS = 30

# TinyLlama has 22 transformer layers; keep ~6 in cache at once.
# PyTorch's allocator retains freed tensors in its pool (not released to OS),
# so peak RSS during generation converges toward eager loading. The real
# benefit of DeepLazy is at *initialisation time*: the model wraps in
# milliseconds with near-zero extra RSS, vs. full weight loading up front.
LRU_CAPACITY = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def _print_stage(label: str, t_start: float, mem_start: float) -> None:
    elapsed = time.perf_counter() - t_start
    delta = _rss_mb() - mem_start
    print(f"  [{label:<18}] +{elapsed:6.2f}s  rss={_rss_mb():7.1f} MB "
          f"(Δ={delta:+7.1f} MB)")


def _has_weights(weights_dir: str) -> bool:
    if not os.path.isdir(weights_dir):
        return False
    files = os.listdir(weights_dir)
    return any(f.endswith(".safetensors") for f in files)


def _download_model(weights_dir: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub not installed — run: pip install huggingface_hub")

    print(f"Downloading {MODEL_ID} → {weights_dir}")
    snapshot_download(repo_id=MODEL_ID, local_dir=weights_dir,
                      ignore_patterns=["*.bin", "*.pt", "flax_*", "tf_*"])
    print("Download complete.\n")


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
        cache_backend=PytorchLocalLRUCache(capacity=LRU_CAPACITY),
        framework=FrameworkType.PYTORCH,
        enable_prefetch=True,
    )
    _print_stage("loader built", t0, mem0)

    lazy = LazyModel(cls=AutoModelForCausalLM, loader=loader)
    model = lazy.model
    model.eval()
    mem_after_wrap = _rss_mb()
    t_wrap = time.perf_counter()
    _print_stage("model wrapped", t0, mem0)

    inputs = tokenizer(prompt, return_tensors="pt")
    peak = _rss_mb()
    ttft = None  # time-to-first-token

    with torch.inference_mode():
        for step in range(MAX_NEW_TOKENS):
            outputs = model(**inputs)
            if ttft is None:
                ttft = time.perf_counter() - t_wrap
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            inputs = {
                "input_ids": torch.cat(
                    [inputs["input_ids"], next_token], dim=1),
                "attention_mask": torch.ones(
                    1, inputs["input_ids"].shape[1] + 1, dtype=torch.long),
            }
            peak = max(peak, _rss_mb())

    _print_stage("generation done", t0, mem0)

    text = tokenizer.decode(
        inputs["input_ids"][0], skip_special_tokens=True)
    print(f"  rss after wrap (before any fwd) : {mem_after_wrap:.1f} MB")
    print(f"  peak rss during generation      : {peak:.1f} MB")
    print(f"  time-to-first-token             : {ttft:.2f}s")
    print(f"  output: {text!r}")

    loader.close()
    del lazy, model, loader
    gc.collect()
    return text, peak, mem_after_wrap, ttft


# ---------------------------------------------------------------------------
# Eager path (baseline)
# ---------------------------------------------------------------------------

def run_eager(tokenizer, prompt: str):
    print("\n=== EAGER (standard from_pretrained) ===")
    gc.collect()
    mem0 = _rss_mb()
    t0 = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS_DIR, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.eval()
    mem_after_load = _rss_mb()
    t_loaded = time.perf_counter()
    _print_stage("model loaded", t0, mem0)

    inputs = tokenizer(prompt, return_tensors="pt")
    peak = _rss_mb()

    with torch.inference_mode():
        # TTFT: one raw forward pass (no KV cache tricks).
        _ = model(**inputs)
        ttft = time.perf_counter() - t_loaded
        peak = max(peak, _rss_mb())

        # Full generation.
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        peak = max(peak, _rss_mb())

    _print_stage("generation done", t0, mem0)

    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    print(f"  rss after load (before any fwd) : {mem_after_load:.1f} MB")
    print(f"  peak rss during generation      : {peak:.1f} MB")
    print(f"  time-to-first-token             : {ttft:.2f}s")
    print(f"  output: {text!r}")

    del model
    gc.collect()
    return text, peak, mem_after_load, ttft


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepLazy smoke test — TinyLlama 1.1B")
    parser.add_argument(
        "--download", action="store_true",
        help="Download model weights from HuggingFace Hub before running")
    args = parser.parse_args()

    if args.download:
        _download_model(WEIGHTS_DIR)

    if not _has_weights(WEIGHTS_DIR):
        sys.exit(
            f"No .safetensors weights found in {WEIGHTS_DIR}\n"
            "Run with --download to fetch them automatically, or manually:\n"
            f"  python3 -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('{MODEL_ID}', local_dir='{WEIGHTS_DIR}', "
            f"ignore_patterns=['*.bin','*.pt','flax_*','tf_*'])\"")

    print(f"model      : {MODEL_ID}")
    print(f"weights dir: {WEIGHTS_DIR}")
    print(f"prompt     : {PROMPT!r}")
    print(f"max tokens : {MAX_NEW_TOKENS}")
    print(f"lru cap    : {LRU_CAPACITY} layers")
    print(f"baseline rss before any load: {_rss_mb():.1f} MB")

    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)

    eager_text, eager_peak, eager_mem_ready, eager_ttft = run_eager(tokenizer, PROMPT)
    lazy_text,  lazy_peak,  lazy_mem_ready,  lazy_ttft  = run_lazy(tokenizer, PROMPT)

    print("\n=== summary ===")
    print(f"{'metric':<38} {'eager':>10} {'lazy':>10}  {'delta':>10}")
    print("-" * 72)
    print(f"{'rss when model is ready (MB)':<38} "
          f"{eager_mem_ready:>10.1f} {lazy_mem_ready:>10.1f}  "
          f"{lazy_mem_ready - eager_mem_ready:>+10.1f}")
    print(f"{'peak rss during generation (MB)':<38} "
          f"{eager_peak:>10.1f} {lazy_peak:>10.1f}  "
          f"{lazy_peak - eager_peak:>+10.1f}")
    print(f"{'time-to-first-token (s)':<38} "
          f"{eager_ttft:>10.2f} {lazy_ttft:>10.2f}  "
          f"{lazy_ttft - eager_ttft:>+10.2f}")
    print()

    init_saving_mb = eager_mem_ready - lazy_mem_ready
    init_saving_pct = (init_saving_mb / eager_mem_ready * 100) if eager_mem_ready else 0
    print(f"Initialisation memory saved : {init_saving_mb:+.1f} MB "
          f"({init_saving_pct:+.1f}%)")
    print()
    print("Memory strategy: mmap-backed weights + madvise(MADV_DONTNEED) on LRU")
    print("eviction releases pages from process RSS while keeping them in the")
    print("kernel page cache — re-access is fast (no disk I/O under normal load).")


if __name__ == "__main__":
    main()
