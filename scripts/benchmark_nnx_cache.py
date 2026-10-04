"""Compare NNX graph/parameter caches using the official FP32 VGGT weights.

uv run python scripts/benchmark_nnx_cache.py
Prepare .cache/parity/inputs.npz with scripts/validate_parity.py first.
"""

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.60")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from vggt_jax import VGGT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path(".cache/parity/inputs.npz"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path(".cache/model.safetensors")
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--report", type=Path, default=Path("reports/nnx_cache_comparison.json")
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be nonnegative and repeats must be positive")
    if jax.default_backend() != "gpu":
        raise RuntimeError("This benchmark requires a GPU")
    model = VGGT.from_pretrained(str(args.checkpoint))
    with np.load(args.inputs) as inputs:
        host_images = inputs["images"]
    images = jnp.asarray(host_images)
    jax.block_until_ready((nnx.state(model), images))

    forward = nnx.jit(lambda m, x, q=None: m(x, q), graph=True, graph_updates=True)
    tree = model.jit()
    start = time.perf_counter()
    tree_compiled = tree.lower(images, None).compile()
    tree_compile_seconds = time.perf_counter() - start
    print("jit_partial compilation", tree_compile_seconds, flush=True)
    start = time.perf_counter()
    graph_compiled = forward.lower(model, images, None).compile()
    graph_compile_seconds = time.perf_counter() - start
    print("nnx.jit compilation", graph_compile_seconds, flush=True)

    cached_compiled = nnx.cached_partial(
        graph_compiled, model, graph=True, graph_updates=True
    )
    cached = nnx.cached_partial(forward, model, graph=True, graph_updates=True)
    methods = {
        "jit_partial_compiled": lambda: tree_compiled(images, None),
        "cached_partial_compiled": lambda: cached_compiled(images, None),
        "nnx_jit_compiled": lambda: graph_compiled(model, images, None),
        "jit_partial": lambda: tree(images, None),
        "cached_partial": lambda: cached(images, None),
        "nnx_jit": lambda: forward(model, images, None),
    }
    outputs, results = {}, {}
    for name, predict in methods.items():
        start = time.perf_counter()
        outputs[name] = jax.block_until_ready(predict())
        first = time.perf_counter() - start
        for _ in range(args.warmup):
            outputs[name] = jax.block_until_ready(predict())
        results[name] = {
            "first_call_seconds": first,
            "samples_ms": [],
            "dispatch_ms": [],
        }
        print(name, "ready", first, flush=True)

    # Alternate methods to reduce the effect of changing clocks and temperature.
    for _ in range(args.repeats):
        for name, predict in methods.items():
            start = time.perf_counter()
            output = predict()
            dispatch = time.perf_counter()
            jax.block_until_ready(output)
            results[name]["samples_ms"].append((time.perf_counter() - start) * 1000)
            results[name]["dispatch_ms"].append((dispatch - start) * 1000)

    reference = jax.tree.leaves(outputs["jit_partial_compiled"])
    for name, stats in results.items():
        stats["median_ms"] = float(np.median(stats["samples_ms"]))
        stats["p90_ms"] = float(np.percentile(stats["samples_ms"], 90))
        stats["median_dispatch_ms"] = float(np.median(stats["dispatch_ms"]))
        leaves = jax.tree.leaves(outputs[name])
        stats["identical_outputs"] = all(
            np.array_equal(np.asarray(actual), np.asarray(expected))
            for actual, expected in zip(leaves, reference, strict=True)
        )
        print(
            name,
            {key: value for key, value in stats.items() if not isinstance(value, list)},
            flush=True,
        )

    report = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": jax.devices()[0].device_kind,
        "jax": jax.__version__,
        "flax": flax.__version__,
        "optimization_level": jax.config.jax_optimization_level,
        "checkpoint": str(args.checkpoint),
        "input_sha256": hashlib.sha256(host_images.tobytes()).hexdigest(),
        "input_shape": list(images.shape),
        "dtype": str(images.dtype),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timing": "synchronized wall time, resident model and inputs, alternating methods; excludes loading, compilation, transfers; normal Python GC",
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "compile_seconds": {
            "jit_partial": tree_compile_seconds,
            "nnx_jit": graph_compile_seconds,
        },
        "methods": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if not all(stats["identical_outputs"] for stats in results.values()):
        raise AssertionError("NNX cache variants produced different predictions")


if __name__ == "__main__":
    main()
