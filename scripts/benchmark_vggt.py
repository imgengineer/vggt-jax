"""Benchmark official PyTorch and JAX/Tokamax VGGT in separate GPU processes.

uv run --extra validation python scripts/benchmark_vggt.py
Uses the identical images/queries saved by scripts/validate_parity.py.
"""

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from validate_parity import REFERENCE_COMMIT, compare, flatten_predictions

CASES = {
    "float32": ("float32", False),
    "float32-tracking": ("float32", True),
    "bfloat16": ("bfloat16", False),
}


def measure(predict, synchronize, warmup, repeats):
    start = time.perf_counter()
    output = predict()
    synchronize(output)
    first_seconds = time.perf_counter() - start
    for _ in range(warmup):
        output = predict()
        synchronize(output)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = predict()
        synchronize(output)
        samples.append((time.perf_counter() - start) * 1000)
    return output, {
        "first_forward_seconds": first_seconds,
        "median_ms": float(np.median(samples)),
        "p10_ms": float(np.percentile(samples, 10)),
        "p90_ms": float(np.percentile(samples, 90)),
        "samples_ms": samples,
    }


def worker(args):
    dtype, tracking = CASES[args.case[0]]
    with np.load(args.inputs) as inputs:
        images = inputs["images"]
        queries = inputs["queries"] if tracking else None
    start = time.perf_counter()
    if args.backend == "torch":
        sys.path.insert(0, str(args.reference_dir.resolve()))
        import torch
        from safetensors.torch import load_file
        from vggt.models.vggt import VGGT

        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = VGGT()
        state = load_file(str(args.checkpoint))
        model.load_state_dict(state, strict=True, assign=True)
        del state
        model = model.eval().cuda()
        images = torch.from_numpy(images).cuda()
        queries = None if queries is None else torch.from_numpy(queries).cuda()
        torch.cuda.synchronize()
        setup_seconds = time.perf_counter() - start
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if dtype == "bfloat16"
            else contextlib.nullcontext()
        )
        # The official forward keeps camera/depth/point heads outside autocast.
        with torch.no_grad(), autocast:
            output, timing = measure(
                lambda: model(images, queries),
                lambda _: torch.cuda.synchronize(),
                args.warmup,
                args.repeats,
            )
        arrays = {
            key: value.float().cpu().numpy()
            for key, value in flatten_predictions(output).items()
        }
        meta = {
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(),
            "execution": "official model.forward, eager, torch.no_grad",
            "attention": "official torch.nn.functional.scaled_dot_product_attention",
            "tf32": False,
            "compile_seconds": None,
        }
    else:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.60")
        os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
        import flax
        import jax
        import jax.numpy as jnp
        import tokamax
        from flax import nnx

        from vggt_jax import VGGT

        if jax.default_backend() != "gpu":
            raise RuntimeError("This benchmark requires a JAX GPU backend")
        model = VGGT.from_pretrained(str(args.checkpoint), dtype=getattr(jnp, dtype))
        images = jnp.asarray(images)
        queries = None if queries is None else jnp.asarray(queries)
        jax.block_until_ready((nnx.state(model), images, queries))
        setup_seconds = time.perf_counter() - start
        predict = model.jit()
        start = time.perf_counter()
        compiled = predict.lower(images, queries).compile()
        compile_seconds = time.perf_counter() - start
        print(f"jax compilation: {compile_seconds:.2f}s", flush=True)
        output, timing = measure(
            lambda: compiled(images, queries),
            jax.block_until_ready,
            args.warmup,
            args.repeats,
        )
        arrays = {
            key: np.asarray(value, np.float32)
            for key, value in flatten_predictions(output).items()
        }
        meta = {
            "jax": jax.__version__,
            "flax": flax.__version__,
            "tokamax": tokamax.__version__,
            "device": jax.devices()[0].device_kind,
            "execution": "model.jit(): nnx.jit_partial(graph=False) of complete model.forward",
            "attention": "Tokamax, implementation=triton on GPU (xla on CPU); FP32 SM80+: TF32_TF32_F32_X3, otherwise HIGHEST; tracker HIGHEST",
            "optimization_level": jax.config.jax_optimization_level,
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "compile_seconds": compile_seconds,
        }
    meta.update(timing, setup_seconds=setup_seconds)
    np.savez(args.work_dir / f"{args.backend}.npz", **arrays)
    (args.work_dir / f"{args.backend}.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    print(
        f"{args.case[0]} {args.backend}: {timing['median_ms']:.2f} ms median "
        f"(p10/p90 {timing['p10_ms']:.2f}/{timing['p90_ms']:.2f})",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path(".cache/parity/inputs.npz"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path(".cache/model.safetensors")
    )
    parser.add_argument("--reference-dir", type=Path, default=Path(".reference/vggt"))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--case", choices=tuple(CASES), action="append")
    parser.add_argument("--work-dir", type=Path, default=Path(".cache/speed"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/speed_benchmark.json")
    )
    parser.add_argument("--backend", choices=("torch", "jax"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be nonnegative and repeats must be positive")
    if args.backend:
        worker(args)
        return
    commit = subprocess.check_output(
        ["git", "-C", str(args.reference_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != REFERENCE_COMMIT:
        raise RuntimeError(f"Reference must be at {REFERENCE_COMMIT}, got {commit}")
    with np.load(args.inputs) as inputs:
        shape = list(inputs["images"].shape)
        input_hash = hashlib.sha256(inputs["images"].tobytes()).hexdigest()
        query_count = inputs["queries"].shape[-2]
    report = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "reference_commit": commit,
        "checkpoint": str(args.checkpoint),
        "input_shape": shape,
        "input_sha256": input_hash,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timing": "synchronized wall time, model inference only; excludes loading, compilation, preprocessing, host/device transfers and output serialization",
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "parity_limits": {"nrmse": 1e-3, "track_rmse_pixels": 1.0, "score_mae": 0.05},
        "cases": {},
    }
    for case in args.case or CASES:
        work_dir = args.work_dir / case
        work_dir.mkdir(parents=True, exist_ok=True)
        for backend in ("torch", "jax"):
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--backend",
                    backend,
                    "--case",
                    case,
                    "--inputs",
                    str(args.inputs.resolve()),
                    "--checkpoint",
                    str(args.checkpoint.resolve()),
                    "--reference-dir",
                    str(args.reference_dir.resolve()),
                    "--work-dir",
                    str(work_dir.resolve()),
                    "--warmup",
                    str(args.warmup),
                    "--repeats",
                    str(args.repeats),
                ],
                check=True,
            )
        torch_result, jax_result = (
            json.loads((work_dir / f"{backend}.json").read_text())
            for backend in ("torch", "jax")
        )
        with (
            np.load(work_dir / "torch.npz") as reference,
            np.load(work_dir / "jax.npz") as actual,
        ):
            metrics = compare(reference, actual, nrmse_limit=1e-3, track_rmse_limit=1.0)
        dtype, tracking = CASES[case]
        result = {
            "dtype": dtype,
            "query_count": query_count if tracking else 0,
            "torch": torch_result,
            "jax": jax_result,
            "speedup_torch_over_jax": torch_result["median_ms"]
            / jax_result["median_ms"],
            "parity_passed": all(value["passed"] for value in metrics.values()),
            "metrics": metrics,
        }
        report["cases"][case] = result
        # Keep completed cases on disk if a later case is interrupted.
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"{case}: speedup {result['speedup_torch_over_jax']:.3f}x; "
            f"parity passed: {result['parity_passed']}",
            flush=True,
        )
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
