"""Load official weights, predict geometry and save NumPy arrays."""

import argparse
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--checkpoint", default="facebook/VGGT-1B")
    parser.add_argument("--output", type=Path, default=Path("predictions.npz"))
    parser.add_argument("--mode", choices=("crop", "pad"), default="crop")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument(
        "--query-points",
        type=Path,
        help=".npy with [N, 2] pixel coordinates after preprocessing",
    )
    args = parser.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.60")
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

    import jax
    import jax.numpy as jnp

    from .data import load_and_preprocess_images
    from .geometry import pose_encoding_to_extri_intri
    from .models import VGGT

    images = jnp.asarray(load_and_preprocess_images(args.images, args.mode))
    queries = (
        None if args.query_points is None else jnp.asarray(np.load(args.query_points))
    )
    print(f"Device: {jax.devices()}; images: {images.shape}", flush=True)
    model = VGGT.from_pretrained(
        args.checkpoint,
        dtype=getattr(jnp, args.dtype),
        enable_track=queries is not None,
    )
    predictions = model.jit()(images, queries)
    predictions["extrinsic"], predictions["intrinsic"] = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    arrays = {
        key: np.asarray(value)
        for key, value in predictions.items()
        if not isinstance(value, list)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
