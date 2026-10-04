"""Measure VGGT's default Tokamax attention against JAX XLA attention on a GPU."""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import numpy as np
import tokamax

from vggt_jax.layers import attention, attention_precision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--report", type=Path, default=Path("reports/attention_benchmark.json")
    )
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError("This benchmark requires a GPU")
    results = {}
    for dtype in (jnp.float32, jnp.bfloat16):
        shape = (1, args.tokens, 16, 64)
        inputs = tuple(
            jax.random.normal(jax.random.key(i), shape, dtype=dtype) for i in range(3)
        )

        def reference(q, k, v):
            with jax.default_matmul_precision("highest"):
                return jax.nn.dot_product_attention(q, k, v, implementation="xla")

        kernels = {
            "tokamax": jax.jit(attention).lower(*inputs).compile(),
            "jax_xla": jax.jit(reference).lower(*inputs).compile(),
        }
        timings, outputs = {}, {}
        for name, kernel in kernels.items():
            for _ in range(3):
                outputs[name] = kernel(*inputs).block_until_ready()
            samples = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                kernel(*inputs).block_until_ready()
                samples.append((time.perf_counter() - start) * 1000)
            timings[name] = float(np.median(samples))
        a, b = (
            np.asarray(outputs[name], np.float32) for name in ("jax_xla", "tokamax")
        )
        nrmse = float(np.sqrt(np.mean((b - a) ** 2)) / np.sqrt(np.mean(a**2)))
        hlo = kernels["tokamax"].as_text().lower()
        results[jnp.dtype(dtype).name] = {
            "shape": list(shape),
            "precision": str(attention_precision(dtype)),
            "median_ms": timings,
            "speedup": timings["jax_xla"] / timings["tokamax"],
            "nrmse_vs_xla": nrmse,
            "custom_kernel_in_hlo": "triton" in hlo or "pallas" in hlo,
        }
        print(jnp.dtype(dtype).name, results[jnp.dtype(dtype).name], flush=True)
    # Verify that the default GPU implementation also provides finite gradients.
    x = jax.random.normal(jax.random.key(7), (1, 128, 2, 64))
    grad = jax.jit(jax.grad(lambda q: jnp.mean(attention(q, x, x) ** 2)))(x)
    assert np.isfinite(grad).all()
    report = {
        "jax": jax.__version__,
        "tokamax": tokamax.__version__,
        "optimization_level": jax.config.jax_optimization_level,
        "device": str(jax.devices()),
        "gpu_gradient_finite": True,
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
