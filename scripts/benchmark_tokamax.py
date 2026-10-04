"""Compare applicable Tokamax kernels on VGGT inference shapes.

uv run --extra validation python scripts/benchmark_tokamax.py
GPU timings use CUPTI; wall times also include Python dispatch and synchronization.
"""

import argparse
import dataclasses
import functools
import importlib.metadata
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.60")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import numpy as np
import tokamax

from vggt_jax.layers import attention_precision


def manual_norm(x, scale, offset, *, epsilon):
    dtype = x.dtype
    x = x.astype(jnp.float32)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    x = (x - mean) * jax.lax.rsqrt(var + epsilon)
    if scale is not None:
        x = x * scale + offset
    return x.astype(dtype)


def linear(x, weight, bias):
    return jnp.matmul(x, weight.T, precision=jax.lax.Precision.HIGHEST) + bias


def ragged_linear(x, weight, bias, *, implementation, precision):
    groups = jnp.asarray([x.shape[0]], jnp.int32)
    return (
        tokamax.ragged_dot(
            x,
            weight.T[None],
            groups,
            precision=precision,
            implementation=implementation,
            bypass_device_check=False,
        )
        + bias
    )


def measure(fn, inputs, reference, repeats):
    try:
        start = time.perf_counter()
        compiled = jax.jit(fn).lower(*inputs).compile()
        compile_seconds = time.perf_counter() - start
        for _ in range(3):
            jax.block_until_ready(compiled(*inputs))
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            output = compiled(*inputs)
            jax.block_until_ready(output)
            samples.append((time.perf_counter() - start) * 1000)
        actual = np.asarray(output, np.float32)
        error = actual.astype(np.float64) - reference
        sf, xs = tokamax.standardize_function(fn, *inputs)
        device = tokamax.benchmark(sf, xs, iterations=repeats, method="cupti")
        return {
            "status": "ok",
            "compile_seconds": compile_seconds,
            "wall_median_ms": float(np.median(samples)),
            "wall_samples_ms": samples,
            "gpu_median_ms": device.median_evaluation_time_ms,
            "gpu_samples_ms": list(device.evaluation_times_ms),
            "finite": bool(np.isfinite(actual).all()),
            "max_abs_error": float(np.max(np.abs(error))),
            "nrmse": float(
                np.sqrt(np.mean(error**2)) / max(np.sqrt(np.mean(reference**2)), 1e-8)
            ),
        }
    except Exception as exc:
        return {
            "status": "unsupported_or_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--op", choices=("attention", "layer_norm", "linear"), action="append"
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/tokamax_kernels.json")
    )
    parser.add_argument("--filter", help="Only run cases whose name contains this text")
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument(
        "--autotune-cache", type=Path, default=Path(".cache/tokamax_autotune.json")
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if jax.default_backend() != "gpu":
        raise RuntimeError("This benchmark requires a GPU")
    rng = np.random.default_rng(0)
    report = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("jax", "tokamax", "jax-triton", "triton")
        },
        "device": jax.devices()[0].device_kind,
        "compute_capability": str(jax.devices()[0].compute_capability),
        "optimization_level": jax.config.jax_optimization_level,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "warmup": 3,
        "repeats": args.repeats,
        "timing": "CUPTI GPU execution and synchronized wall time; excludes compilation",
        "shape_source": "2 images at 350x518, 5 track queries, 64 virtual tracks",
        "skipped_ops": {
            "gated_linear_unit": "VGGT has biased GELU MLPs, not gated MLPs; replacing them changes the model",
            "linear_softmax_cross_entropy_loss": "Not used by VGGT inference",
            "triangle_multiplication": "Not used by VGGT",
        },
        "cases": {},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    tuning_cache = None

    def array(shape, dtype, scale=1.0):
        return jnp.asarray(rng.standard_normal(shape).astype(np.float32) * scale, dtype)

    def run(name, metadata, inputs, reference_fn, candidates):
        nonlocal tuning_cache
        if args.filter and args.filter not in name:
            return
        reference = np.asarray(jax.jit(reference_fn)(*inputs), np.float64)
        result = dict(metadata, implementations={})
        for impl, fn in candidates.items():
            value = measure(fn, inputs, reference, args.repeats)
            result["implementations"][impl] = value
            print(
                name,
                impl,
                {
                    k: v
                    for k, v in value.items()
                    if k not in ("wall_samples_ms", "gpu_samples_ms")
                },
                flush=True,
            )
        if args.autotune:
            for impl, fn in candidates.items():
                if "triton" not in impl:
                    continue
                print(name, impl, "autotuning", flush=True)
                tuned = tokamax.autotune(
                    fn,
                    *inputs,
                    ignore_cache=True,
                    progress_bar=False,
                    timeout=180,
                    max_workers=2,
                )
                configs = []
                for ba, data in tuned.data:
                    try:
                        best = data.fastest_config
                    except (ValueError, ExceptionGroup) as exc:
                        configs.append({"configs_tried": len(data), "error": str(exc)})
                        continue
                    configs.append(
                        {
                            "configs_tried": len(data),
                            "fastest": dataclasses.asdict(best),
                        }
                    )
                with tuned:
                    value = measure(fn, inputs, reference, args.repeats)
                result["implementations"][f"{impl}_autotuned"] = dict(
                    value, tuning=configs
                )
                print(
                    name,
                    impl,
                    "autotuned",
                    value.get("gpu_median_ms"),
                    configs,
                    flush=True,
                )
                pruned = tokamax.AutotuningResult(
                    tuned.device_kind,
                    tuple(
                        (ba, data.prune())
                        for ba, data in tuned.data
                        if data.prune_errors()
                    ),
                    tuned.tokamax_version,
                )
                tuning_cache = pruned if tuning_cache is None else tuning_cache | pruned
                args.autotune_cache.parent.mkdir(parents=True, exist_ok=True)
                args.autotune_cache.write_text(
                    tuning_cache.dumps(prune_errors=True) + "\n"
                )
        report["cases"][name] = result
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        jax.clear_caches()

    ops = args.op or ("attention", "layer_norm", "linear")
    if "attention" in ops:
        shapes = (
            ("frame", (2, 930, 16, 64), 930, False),
            ("global", (1, 1860, 16, 64), 1860, False),
            ("camera", (1, 2, 16, 128), 2, False),
            # Tracker reshapes [batch, points + virtual, frames, dim] into
            # [batch * groups, sequence, heads, head_dim].
            ("track_time", (69, 2, 8, 48), 2, True),
            ("track_space", (2, 64, 8, 48), 64, True),
            ("track_point_to_virtual", (2, 5, 8, 48), 64, True),
            ("track_virtual_to_point", (2, 64, 8, 48), 5, True),
        )
        for label, shape, key_tokens, tracking in shapes:
            dtypes = (
                (jnp.float32, jnp.bfloat16)
                if label in ("frame", "global")
                else (jnp.float32,)
            )
            for dtype in dtypes:
                key_shape = (shape[0], key_tokens, shape[2], shape[3])
                inputs = (
                    array(shape, dtype),
                    array(key_shape, dtype),
                    array(key_shape, dtype),
                )
                precision = (
                    jax.lax.Precision.HIGHEST
                    if tracking
                    else attention_precision(dtype)
                )

                def reference(q, k, v):
                    with jax.default_matmul_precision("highest"):
                        return jax.nn.dot_product_attention(
                            q, k, v, implementation="xla"
                        )

                candidates = {
                    impl: functools.partial(
                        tokamax.dot_product_attention,
                        precision=precision,
                        implementation=impl,
                    )
                    for impl in ("triton", "xla", "xla_chunked", "cudnn", "mosaic")
                }
                if tracking:
                    # nn.MultiheadAttention pre-scales Q before the dot product.
                    inputs = (
                        jax.lax.optimization_barrier(inputs[0] * shape[-1] ** -0.5),
                        *inputs[1:],
                    )

                    def reference(q, k, v):
                        with jax.default_matmul_precision("highest"):
                            return jax.nn.dot_product_attention(
                                q, k, v, scale=1.0, implementation="xla"
                            )

                    candidates = {
                        impl: functools.partial(fn, scale=1.0)
                        for impl, fn in candidates.items()
                    }
                run(
                    f"attention/{label}/{jnp.dtype(dtype).name}",
                    {
                        "q_shape": shape,
                        "kv_shape": key_shape,
                        "precision": str(precision),
                        "scale": 1.0 if tracking else shape[-1] ** -0.5,
                    },
                    inputs,
                    reference,
                    candidates,
                )
    if "layer_norm" in ops:
        shapes = (
            ("backbone", (1860, 1024), True, 1e-6),
            ("qk", (2, 930, 16, 64), True, 1e-6),
            ("dense", (2, 925, 2048), True, 1e-5),
            ("camera", (1, 2, 2048), False, 1e-6),
            ("track", (69, 2, 384), True, 1e-5),
            ("track_input", (69, 2, 388), True, 1e-5),
            ("track_fmap", (2, 175, 259, 128), True, 1e-5),
            ("track_feature", (1, 5, 2, 128), True, 1e-5),
        )
        for label, shape, affine, eps in shapes:
            dtypes = (
                (jnp.float32, jnp.bfloat16)
                if label in ("backbone", "qk")
                else (jnp.float32,)
            )
            for dtype in dtypes:
                inputs = (
                    array(shape, dtype),
                    array((shape[-1],), jnp.float32, 0.1) + 1 if affine else None,
                    array((shape[-1],), jnp.float32, 0.1) if affine else None,
                )
                reference = functools.partial(manual_norm, epsilon=eps)
                candidates = {"manual_xla": reference} | {
                    impl: functools.partial(
                        tokamax.layer_norm, epsilon=eps, implementation=impl
                    )
                    for impl in ("xla", "triton")
                }
                run(
                    f"layer_norm/{label}/{jnp.dtype(dtype).name}",
                    {"shape": shape, "affine": affine, "epsilon": eps},
                    inputs,
                    reference,
                    candidates,
                )
    if "linear" in ops:
        for label, m, k, n in (
            ("qkv", 1860, 1024, 3072),
            ("mlp_up", 1860, 1024, 4096),
            ("mlp_down", 1860, 4096, 1024),
            ("camera", 2, 2048, 6144),
            ("track", 138, 384, 1536),
        ):
            dtypes = (
                (jnp.float32, jnp.bfloat16)
                if label in ("qkv", "mlp_up", "mlp_down")
                else (jnp.float32,)
            )
            for dtype in dtypes:
                inputs = (
                    array((m, k), dtype),
                    array((n, k), dtype, k**-0.5),
                    array((n,), dtype, 0.1),
                )
                candidates = {"matmul": linear}
                for impl in ("xla", "triton", "mosaic"):
                    candidates[f"ragged_{impl}"] = functools.partial(
                        ragged_linear,
                        implementation=impl,
                        precision=jax.lax.Precision.HIGHEST,
                    )
                if dtype == jnp.float32:
                    candidates["ragged_triton_tf32x3"] = functools.partial(
                        ragged_linear,
                        implementation="triton",
                        precision=jax.lax.DotAlgorithmPreset.TF32_TF32_F32_X3,
                    )
                run(
                    f"linear/{label}/{jnp.dtype(dtype).name}",
                    {"x_shape": [m, k], "weight_shape": [n, k], "bias": True},
                    inputs,
                    linear,
                    candidates,
                )
    print(f"Report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
