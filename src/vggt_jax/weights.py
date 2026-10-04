"""Strict, name-preserving loading of official PyTorch and safetensors weights."""

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx
from huggingface_hub import hf_hub_download
from safetensors import safe_open


@dataclass(frozen=True)
class LoadReport:
    loaded: int
    parameters: int
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    skipped: tuple[str, ...]


def _load(model, keys, shape, get, *, strict):
    flat = nnx.to_flat_state(nnx.state(model, nnx.Param))
    expected = {".".join(map(str, path)): (path, variable) for path, variable in flat}
    keys = set(keys)
    skipped = tuple(
        sorted(
            key
            for key in keys
            if any(
                key.startswith(name + ".") and getattr(model, name, True) is None
                for name in ("camera_head", "depth_head", "point_head", "track_head")
            )
        )
    )
    missing = tuple(sorted(expected.keys() - keys))
    unexpected = tuple(sorted(keys - expected.keys() - set(skipped)))
    mismatches = [
        f"{name}: checkpoint {tuple(shape(name))}, model {tuple(variable.shape)}"
        for name, (_, variable) in expected.items()
        if name in keys and tuple(shape(name)) != tuple(variable.shape)
    ]
    if mismatches or (strict and (missing or unexpected)):
        raise ValueError(
            f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}, shapes={mismatches}"
        )
    updates, parameters = {}, 0
    for name, (path, variable) in expected.items():
        if name not in keys:
            continue
        value = get(name)
        if hasattr(value, "detach"):
            value = value.detach().cpu()
            # NumPy has no native torch.bfloat16 representation.
            if str(value.dtype) == "torch.bfloat16":
                value = value.float()
            value = value.numpy()
        array = jnp.asarray(np.asarray(value), dtype=variable.dtype)
        updates[path] = variable.replace(array)
        parameters += array.size
    nnx.update(model, nnx.from_flat_state(updates))
    return LoadReport(len(updates), parameters, missing, unexpected, skipped)


def load_state_dict(model, state_dict, *, strict=True):
    """Load a mapping of NumPy arrays or torch tensors. No layout conversion is needed."""
    return _load(
        model,
        state_dict.keys(),
        lambda key: state_dict[key].shape,
        state_dict.__getitem__,
        strict=strict,
    )


def load_checkpoint(model, path, *, strict=True):
    path = Path(path)
    if path.suffix == ".safetensors":
        with safe_open(str(path), framework="numpy") as checkpoint:
            return _load(
                model,
                checkpoint.keys(),
                lambda key: checkpoint.get_slice(key).get_shape(),
                checkpoint.get_tensor,
                strict=strict,
            )
    if path.suffix in (".pt", ".pth", ".bin"):
        try:
            import torch
        except ImportError as error:
            raise ImportError(
                "Loading .pt requires `uv sync --extra validation`; safetensors needs no PyTorch"
            ) from error
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if "state_dict" in state:
            state = state["state_dict"]
        return load_state_dict(model, state, strict=strict)
    raise ValueError("Checkpoint must be a .safetensors, .pt, .pth or .bin file")


def load_pretrained(model, name_or_path, *, revision=None):
    path = Path(name_or_path)
    if path.is_dir():
        path = next(
            (
                path / name
                for name in ("model.safetensors", "model.pt", "pytorch_model.bin")
                if (path / name).is_file()
            ),
            path / "model.safetensors",
        )
    elif not path.is_file():
        path = Path(
            hf_hub_download(str(name_or_path), "model.safetensors", revision=revision)
        )
    return load_checkpoint(model, path)
