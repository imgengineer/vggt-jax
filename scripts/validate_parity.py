"""Compare independent PyTorch/JAX processes with the official VGGT checkpoint.

uv run --extra validation python scripts/validate_parity.py --checkpoint .cache/model.safetensors
"""

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REFERENCE_COMMIT = "a288dd0f14786c93483e45524328726ab7b1b4ce"


def flatten_predictions(predictions):
    values = {}
    for key, value in predictions.items():
        if isinstance(value, (list, tuple)):
            values.update({f"{key}/{i}": item for i, item in enumerate(value)})
        elif value is not None:
            values[key] = value
    return values


def worker(args):
    inputs = np.load(args.work_dir / "inputs.npz")
    images, queries = inputs["images"], inputs["queries"]
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
        if args.checkpoint.suffix == ".safetensors":
            state = load_file(str(args.checkpoint))
        else:
            state = torch.load(
                args.checkpoint, map_location="cpu", weights_only=True, mmap=True
            )
        model.load_state_dict(state, strict=True, assign=True)
        del state
        model = model.eval().to(args.device)
        images = torch.from_numpy(images).to(args.device)
        queries = torch.from_numpy(queries).to(args.device)
        with torch.no_grad():
            autocast = (
                torch.autocast(args.device, dtype=torch.bfloat16)
                if args.dtype == "bfloat16"
                else contextlib.nullcontext()
            )
            with autocast:
                tokens, index = model.aggregator(images)
            predictions = {"images": images, "pose_enc_list": model.camera_head(tokens)}
            predictions["pose_enc"] = predictions["pose_enc_list"][-1]
            predictions["depth"], predictions["depth_conf"] = model.depth_head(
                tokens, images, index
            )
            predictions["world_points"], predictions["world_points_conf"] = (
                model.point_head(tokens, images, index)
            )
            tracks, predictions["vis"], predictions["conf"] = model.track_head(
                tokens, images, index, queries
            )
            predictions["track"] = tracks[-1]
        outputs = {
            key: value.float().cpu().numpy()
            for key, value in flatten_predictions(predictions).items()
        }
        outputs.update(
            {
                f"aggregator/{i}": tokens[i].float().cpu().numpy()
                for i in (4, 11, 17, 23)
            }
        )
        versions = {"torch": torch.__version__, "device": args.device}
    else:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.60")
        os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
        if args.device == "cpu":
            os.environ["JAX_PLATFORMS"] = "cpu"
        import flax
        import jax
        import jax.numpy as jnp
        import tokamax
        from flax import nnx

        from vggt_jax import VGGT

        model = VGGT.from_pretrained(
            str(args.checkpoint), dtype=getattr(jnp, args.dtype)
        )
        if args.device == "cuda" and jax.default_backend() != "gpu":
            raise RuntimeError(
                f"CUDA validation requested but JAX backend is {jax.default_backend()}"
            )
        images, queries = jnp.asarray(images), jnp.asarray(queries)
        # Separate compiled stages allow inspecting intermediate features and avoid
        # compiling repeated heads and all 24 attention blocks into one huge program.
        tokens, index = nnx.jit(lambda m, x: m(x))(model.aggregator, images)
        jax.block_until_ready(tokens)
        predictions = {"images": images}
        predictions["pose_enc_list"] = nnx.jit(lambda m, x: m(x))(
            model.camera_head, tokens
        )
        predictions["pose_enc"] = predictions["pose_enc_list"][-1]
        dense_head = nnx.jit(lambda m, x, im: m(x, im, 5))
        predictions["depth"], predictions["depth_conf"] = dense_head(
            model.depth_head, tokens, images
        )
        jax.block_until_ready(predictions["depth"])
        predictions["world_points"], predictions["world_points_conf"] = dense_head(
            model.point_head, tokens, images
        )
        jax.block_until_ready(predictions["world_points"])
        tracks, predictions["vis"], predictions["conf"] = nnx.jit(
            lambda m, x, im, q: m(x, im, 5, q)
        )(model.track_head, tokens, images, queries)
        predictions["track"] = tracks[-1]
        outputs = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in flatten_predictions(predictions).items()
        }
        outputs.update(
            {
                f"aggregator/{i}": np.asarray(tokens[i], dtype=np.float32)
                for i in (4, 11, 17, 23)
            }
        )
        versions = {
            "jax": jax.__version__,
            "flax": flax.__version__,
            "tokamax": tokamax.__version__,
            "attention": "Tokamax, implementation=triton on GPU (xla on CPU); FP32 SM80+: TF32_TF32_F32_X3, otherwise HIGHEST; tracker HIGHEST",
            "optimization_level": jax.config.jax_optimization_level,
            "device": str(jax.devices()),
        }
    elapsed = time.perf_counter() - start
    np.savez(args.work_dir / f"{args.backend}.npz", **outputs)
    (args.work_dir / f"{args.backend}_meta.json").write_text(
        json.dumps(
            {**versions, "elapsed_seconds_with_loading_and_compilation": elapsed},
            indent=2,
        )
    )
    print(f"{args.backend}: completed in {elapsed:.1f}s", flush=True)


def compare(
    reference,
    actual,
    *,
    nrmse_limit,
    track_rmse_limit,
    score_mae_limit=0.05,
    strict_tracking=False,
):
    if set(reference.files) != set(actual.files):
        raise ValueError("Prediction keys differ")
    report = {}
    for key in reference.files:
        a, b = reference[key].astype(np.float64), actual[key].astype(np.float64)
        if a.shape != b.shape:
            raise ValueError(f"{key}: shape mismatch {a.shape} vs {b.shape}")
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        error = b - a
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(error**2)))
        nrmse = rmse / max(float(np.sqrt(np.mean(a**2))), 1e-8)
        passed = finite and nrmse <= nrmse_limit
        if key == "track":
            passed = finite and rmse <= track_rmse_limit
        elif key in ("vis", "conf"):
            passed = finite and mae <= score_mae_limit
        if strict_tracking and key in ("track", "vis", "conf"):
            passed = passed and nrmse <= nrmse_limit
        report[key] = {
            "shape": list(a.shape),
            "mae": mae,
            "max_abs": float(np.max(np.abs(error))),
            "rmse": rmse,
            "nrmse": nrmse,
            "finite": finite,
            "passed": passed,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path(".cache/model.safetensors")
    )
    parser.add_argument("--reference-dir", type=Path, default=Path(".reference/vggt"))
    parser.add_argument("--images", nargs="+", type=Path)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mode", choices=("crop", "pad"), default="crop")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--nrmse-limit", type=float, default=1e-3)
    parser.add_argument("--track-rmse-limit", type=float, default=1.0)
    parser.add_argument("--score-mae-limit", type=float, default=0.05)
    parser.add_argument(
        "--strict-tracking",
        action="store_true",
        help="Also require tracking outputs to pass NRMSE",
    )
    parser.add_argument("--work-dir", type=Path, default=Path(".cache/parity"))
    parser.add_argument("--report", type=Path, default=Path("reports/parity.json"))
    parser.add_argument("--backend", choices=("torch", "jax"), help=argparse.SUPPRESS)
    parser.add_argument(
        "--reuse-torch",
        action="store_true",
        help="Reuse saved reference outputs for debugging",
    )
    args = parser.parse_args()
    if args.backend:
        worker(args)
        return
    if not args.checkpoint.is_file():
        from huggingface_hub import hf_hub_download

        args.checkpoint = Path(hf_hub_download("facebook/VGGT-1B", "model.safetensors"))
    if not args.reference_dir.exists():
        args.reference_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/facebookresearch/vggt.git",
                str(args.reference_dir),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(args.reference_dir), "checkout", REFERENCE_COMMIT],
            check=True,
        )
    commit = subprocess.check_output(
        ["git", "-C", str(args.reference_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != REFERENCE_COMMIT:
        raise RuntimeError(f"Reference must be at {REFERENCE_COMMIT}, got {commit}")
    paths = (
        args.images
        or sorted((args.reference_dir / "examples/kitchen/images").glob("*.png"))[
            : args.frames
        ]
    )
    from vggt_jax.data import load_and_preprocess_images

    images = load_and_preprocess_images(paths, args.mode)
    images = np.broadcast_to(images[None], (args.batch_size, *images.shape)).copy()
    h, w = images.shape[-2:]
    queries = np.array(
        [
            [w * 0.5, h * 0.5],
            [w * 0.25 + 0.3, h * 0.3 + 0.7],
            [w * 0.7, h * 0.6],
            [0.5, 0.5],
            [w - 1.5, h - 1.5],
        ],
        np.float32,
    )
    queries = np.broadcast_to(queries[None], (args.batch_size, *queries.shape)).copy()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = args.work_dir / "inputs.npz"
    if args.reuse_torch:
        saved = np.load(inputs_path)
        np.testing.assert_array_equal(saved["images"], images)
        np.testing.assert_array_equal(saved["queries"], queries)
    else:
        np.savez(inputs_path, images=images, queries=queries)
    for backend in ("torch", "jax"):
        if backend == "torch" and args.reuse_torch:
            continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--backend",
            backend,
            "--checkpoint",
            str(args.checkpoint.resolve()),
            "--work-dir",
            str(args.work_dir.resolve()),
            "--reference-dir",
            str(args.reference_dir.resolve()),
            "--dtype",
            args.dtype,
            "--device",
            args.device,
        ]
        subprocess.run(command, check=True)
    with (
        np.load(args.work_dir / "torch.npz") as reference,
        np.load(args.work_dir / "jax.npz") as actual,
    ):
        metrics = compare(
            reference,
            actual,
            nrmse_limit=args.nrmse_limit,
            track_rmse_limit=args.track_rmse_limit,
            score_mae_limit=args.score_mae_limit,
            strict_tracking=args.strict_tracking,
        )
    report = {
        "passed": all(item["passed"] for item in metrics.values()),
        "reference_commit": commit,
        "checkpoint": str(args.checkpoint),
        "dtype": args.dtype,
        "images": list(map(str, paths)),
        "input_shape": list(images.shape),
        "input_sha256": hashlib.sha256(images.tobytes()).hexdigest(),
        "nrmse_limit": args.nrmse_limit,
        "track_rmse_limit_pixels": args.track_rmse_limit,
        "score_mae_limit": args.score_mae_limit,
        "strict_tracking": args.strict_tracking,
        "torch": json.loads((args.work_dir / "torch_meta.json").read_text()),
        "jax": json.loads((args.work_dir / "jax_meta.json").read_text()),
        "metrics": metrics,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    for key, item in metrics.items():
        print(
            f"{key:24s} NRMSE={item['nrmse']:.3g} max_abs={item['max_abs']:.3g} {'PASS' if item['passed'] else 'FAIL'}"
        )
    print(f"Report: {args.report}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
