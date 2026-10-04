"""Locate tracking divergence with identical inputs and controlled perturbations.

uv run --extra validation python scripts/debug_tracking.py --inputs .cache/ten_views/inputs.npz
The two GPU workers run sequentially. Large feature maps and replay inputs stay in
the ignored work directory; the JSON report contains only measurements.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from validate_parity import REFERENCE_COMMIT


def error(reference, actual):
    reference, actual = (
        np.asarray(reference, np.float64),
        np.asarray(actual, np.float64),
    )
    delta = actual - reference
    rmse = float(np.sqrt(np.mean(delta**2)))
    return {
        "rmse": rmse,
        "nrmse": rmse / max(float(np.sqrt(np.mean(reference**2))), 1e-8),
        "mae": float(np.mean(np.abs(delta))),
        "max_abs": float(np.max(np.abs(delta))),
    }


def replay_names():
    names = ["updateformer", "corr_mlp", "ffeat_norm", "ffeat_updater.0"]
    for group in (
        "time_blocks",
        "space_virtual2point_blocks",
        "space_virtual_blocks",
        "space_point2virtual_blocks",
    ):
        for index in range(6):
            name = f"updateformer.{group}.{index}"
            names.extend([name, name + (".cross_attn" if "2" in group else ".attn")])
    return names


def torch_worker(args):
    sys.path.insert(0, str(args.reference_dir.resolve()))
    import torch
    from safetensors import safe_open
    from vggt.heads.track_modules.base_track_predictor import BaseTrackerPredictor
    from vggt.heads.track_modules.utils import get_2d_embedding

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with np.load(args.inputs) as inputs:
        images, queries = inputs["images"], torch.from_numpy(inputs["queries"]).cuda()
    if args.features:
        with np.load(args.features) as source:
            fmaps = torch.from_numpy(source["fmaps"]).cuda()
        tracker = BaseTrackerPredictor(stride=2, corr_levels=7).eval()
        prefix = "track_head.tracker."
        with safe_open(str(args.checkpoint), framework="pt") as checkpoint:
            tracker.load_state_dict(
                {
                    key[len(prefix) :]: checkpoint.get_tensor(key)
                    for key in checkpoint.keys()
                    if key.startswith(prefix)
                },
                strict=True,
                assign=True,
            )
        tracker = tracker.cuda()
    else:
        from vggt.models.vggt import VGGT

        model = VGGT(enable_camera=False, enable_depth=False, enable_point=False).eval()
        keys = set(model.state_dict())
        with safe_open(str(args.checkpoint), framework="pt") as checkpoint:
            model.load_state_dict(
                {key: checkpoint.get_tensor(key) for key in keys},
                strict=True,
                assign=True,
            )
        model = model.cuda()
        with torch.no_grad():
            image_tensor = torch.from_numpy(images).cuda()
            tokens, start = model.aggregator(image_tensor)
            fmaps = model.track_head.feature_extractor(tokens, image_tensor, start)
        tracker = model.track_head.tracker
        del tokens, image_tensor, model
        torch.cuda.empty_cache()
    np.savez(args.work_dir / "features.npz", fmaps=fmaps.cpu().numpy())
    captures, counters = {}, {}

    def hook(name):
        def collect(module, inputs, output):
            call = counters.get(name, 0)
            counters[name] = call + 1
            output = output[0] if isinstance(output, tuple) else output
            base = f"{name}/call{call}"
            captures[f"{base}/in"] = inputs[0].detach().cpu().numpy()
            captures[f"{base}/out"] = output.detach().cpu().numpy()
            if len(inputs) > 1:
                captures[f"{base}/context"] = inputs[1].detach().cpu().numpy()

        return collect

    names = set(replay_names())
    handles = [
        module.register_forward_hook(hook(name))
        for name, module in tracker.named_modules()
        if name in names
    ]
    with torch.no_grad():
        tracks, vis, conf = tracker(queries, fmaps, iters=4)
    for handle in handles:
        handle.remove()
    track_arrays = np.stack([track.cpu().numpy() for track in tracks])
    captures.update(tracks=track_arrays, vis=vis.cpu().numpy(), conf=conf.cpu().numpy())
    with torch.no_grad():
        for iteration in range(4):
            coords = (
                queries[:, None] / 2 if iteration == 0 else tracks[iteration - 1] / 2
            )
            flows = (
                (coords - coords[:, :1])
                .permute(0, 2, 1, 3)
                .reshape(-1, coords.shape[1], 2)
            )
            # At iteration zero the repeated coordinates give zero displacement.
            if iteration == 0:
                flows = flows.expand(-1, fmaps.shape[1], -1)
            captures[f"flows/{iteration}"] = flows.cpu().numpy()
            captures[f"flow_embedding/{iteration}"] = (
                get_2d_embedding(flows, 64, cat_coords=False).cpu().numpy()
            )
    np.savez(args.work_dir / "torch.npz", **captures)

    sensitivity = []
    for direction in (1, -1):
        calls = [0]

        def perturb(module, inputs, output):
            if calls[0] == 0:
                delta, auxiliary = output
                delta = delta.clone()
                delta[..., :2] = torch.nextafter(
                    delta[..., :2],
                    torch.full_like(delta[..., :2], direction * float("inf")),
                )
                output = delta, auxiliary
            calls[0] += 1
            return output

        handle = tracker.updateformer.register_forward_hook(perturb)
        with torch.no_grad():
            perturbed, _, _ = tracker(queries, fmaps, iters=4)
        handle.remove()
        sensitivity.append(
            {
                "first_delta_coordinate_ulp": direction,
                "iterations": [
                    error(reference, actual.cpu().numpy())
                    for reference, actual in zip(track_arrays, perturbed)
                ],
            }
        )
    (args.work_dir / "torch.json").write_text(
        json.dumps(
            {"torch": torch.__version__, "ulp_sensitivity": sensitivity}, indent=2
        )
        + "\n"
    )


def jax_worker(args):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.60")
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
    import flax
    import jax
    import jax.numpy as jnp
    import tokamax
    from flax import nnx
    from safetensors import safe_open

    import vggt_jax.tracking as tracking
    from vggt_jax.config import VGGTConfig
    from vggt_jax.layers import attention_implementation
    from vggt_jax.weights import load_state_dict

    tracker = nnx.eval_shape(
        lambda: tracking.BaseTrackerPredictor(VGGTConfig(), rngs=nnx.Rngs(0))
    )
    prefix = "track_head.tracker."
    with safe_open(str(args.checkpoint), framework="numpy") as checkpoint:
        load_state_dict(
            tracker,
            {
                key[len(prefix) :]: checkpoint.get_tensor(key)
                for key in checkpoint.keys()
                if key.startswith(prefix)
            },
        )
    with (
        np.load(args.inputs) as inputs,
        np.load(args.work_dir / "features.npz") as features,
    ):
        queries, fmaps = jnp.asarray(inputs["queries"]), jnp.asarray(features["fmaps"])
    with np.load(args.work_dir / "torch.npz") as reference:
        predict = nnx.jit(lambda model, q, f: model(q, f, iters=4))
        tracks, vis, conf = predict(tracker, queries, fmaps)
        shared = [error(a, b) for a, b in zip(reference["tracks"], tracks)]
        np.savez(
            args.work_dir / "jax.npz",
            tracks=np.stack([np.asarray(track) for track in tracks]),
            vis=np.asarray(vis),
            conf=np.asarray(conf),
        )
        encoding = jax.jit(lambda flows: tracking.flow_embedding(flows, 64))
        same_flow = [
            error(
                reference[f"flow_embedding/{i}"],
                encoding(jnp.asarray(reference[f"flows/{i}"])),
            )
            for i in range(4)
        ]

        # A causal diagnostic: feed official embeddings while all correlations,
        # feature updates and coordinates continue to come from the JAX tracker.
        original_embedding = tracking.flow_embedding
        flow_values = iter(
            jnp.asarray(reference[f"flow_embedding/{i}"]) for i in range(4)
        )
        tracking.flow_embedding = lambda flows, dim: next(flow_values)
        try:
            fixed_predict = nnx.jit(lambda model, q, f: model(q, f, iters=4))
            fixed_tracks, _, _ = fixed_predict(tracker, queries, fmaps)
            fixed_flow = [
                error(a, b) for a, b in zip(reference["tracks"], fixed_tracks)
            ]
        finally:
            tracking.flow_embedding = original_embedding

        call = nnx.jit(lambda module, x: module(x))
        call_attention = nnx.jit(lambda module, x, context: module(x, context))
        replay = {}
        for name in replay_names():
            module = tracker
            for component in name.split("."):
                if isinstance(module, nnx.List):
                    module = module[int(component)]
                elif isinstance(module, nnx.Dict):
                    module = module[component]
                else:
                    module = getattr(module, component)
            replay[name] = []
            for iteration in range(4):
                base = f"{name}/call{iteration}"
                x = jnp.asarray(reference[f"{base}/in"])
                if name.startswith("updateformer."):
                    context = (
                        jnp.asarray(reference[f"{base}/context"])
                        if f"{base}/context" in reference
                        else None
                    )
                    actual = call_attention(module, x, context)
                else:
                    actual = call(module, x)
                replay[name].append(error(reference[f"{base}/out"], actual))
        results = {
            "jax": jax.__version__,
            "flax": flax.__version__,
            "tokamax": tokamax.__version__,
            "device": jax.devices()[0].device_kind,
            "optimization_level": jax.config.jax_optimization_level,
            "tracker_attention": attention_implementation(tracker=True),
            "shared_official_feature_maps_track_pixels": shared,
            "same_input_flow_embedding": same_flow,
            "official_flow_embedding_intervention_track_pixels": fixed_flow,
            "same_input_module_replay": replay,
            "shared_feature_maps_score_errors": {
                "vis": error(reference["vis"], vis),
                "conf": error(reference["conf"], conf),
            },
        }
    (args.work_dir / "jax.json").write_text(json.dumps(results, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", type=Path, default=Path(".cache/ten_views/inputs.npz")
    )
    parser.add_argument(
        "--features",
        type=Path,
        help="Reuse official feature maps saved under the fmaps key",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path(".cache/model.safetensors")
    )
    parser.add_argument("--reference-dir", type=Path, default=Path(".reference/vggt"))
    parser.add_argument("--work-dir", type=Path, default=Path(".cache/tracking_debug"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/tracking_debug.json")
    )
    parser.add_argument("--backend", choices=("torch", "jax"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.backend:
        (torch_worker if args.backend == "torch" else jax_worker)(args)
        print(f"{args.backend} tracking debug completed", flush=True)
        return
    commit = subprocess.check_output(
        ["git", "-C", str(args.reference_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != REFERENCE_COMMIT:
        raise RuntimeError(f"Reference must be at {REFERENCE_COMMIT}, got {commit}")
    for backend in ("torch", "jax"):
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--backend",
                backend,
            ],
            check=True,
        )
    with (
        np.load(args.inputs) as inputs,
        np.load(args.work_dir / "features.npz") as features,
    ):
        report = {
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "reference_commit": commit,
            "checkpoint": str(args.checkpoint),
            "input_shape": list(inputs["images"].shape),
            "input_sha256": hashlib.sha256(inputs["images"].tobytes()).hexdigest(),
            "feature_shape": list(features["fmaps"].shape),
            "feature_sha256": hashlib.sha256(features["fmaps"].tobytes()).hexdigest(),
            "queries": inputs["queries"].tolist(),
            "flow_encoding": {
                "stride_image_pixels": 2,
                "max_frequency_radians_per_feature_pixel": 968.75,
                "min_wavelength_image_pixels": 4 * np.pi / 968.75,
            },
            "torch": json.loads((args.work_dir / "torch.json").read_text()),
            "jax": json.loads((args.work_dir / "jax.json").read_text()),
            "notes": [
                "Both trackers consume identical official DPT feature maps and official weights.",
                "Module replays consume identical PyTorch inputs, preventing feedback between stages.",
                "The fixed-flow intervention is a diagnostic and does not change production inference.",
                "Diagnostic compilation boundaries differ from the complete-model speed benchmark.",
            ],
        }
    jax_result = report["jax"]
    report["diagnosis"] = {
        "amplification_stage": "coordinate displacement -> high-frequency flow_embedding -> iterative coordinate refinement",
        "same_input_module_replay_max_nrmse": max(
            item["nrmse"]
            for rows in jax_result["same_input_module_replay"].values()
            for item in rows
        ),
        "same_input_flow_embedding_bit_identical": all(
            item["max_abs"] == 0 for item in jax_result["same_input_flow_embedding"]
        ),
        "shared_features_final_track_rmse_pixels": jax_result[
            "shared_official_feature_maps_track_pixels"
        ][-1]["rmse"],
        "fixed_official_flow_embedding_final_track_rmse_pixels": jax_result[
            "official_flow_embedding_intervention_track_pixels"
        ][-1]["rmse"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
