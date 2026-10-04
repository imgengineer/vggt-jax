# Copyright (c) Meta Platforms, Inc. and affiliates.
# JAX adaptation; see LICENSE and NOTICE for upstream attribution.
"""NNX layers with the parameter names and layouts of the official checkpoints."""

import contextlib
import functools
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tokamax
from flax import nnx


@functools.cache
def _attention_autotuning_cache():
    """Load an optional device-specific Tokamax cache for known shapes."""
    if jax.default_backend() != "gpu":
        return None
    cache_path = os.environ.get("VGGT_TOKAMAX_AUTOTUNE")
    if cache_path is None:
        cache_path = Path(__file__).with_name("tokamax_cache") / "rtx5090.json"
    try:
        path = Path(cache_path)
        if not path.is_file():
            return None
        cache = tokamax.AutotuningResult.loads(path.read_text())
        if (
            cache.device_kind != jax.devices()[0].device_kind
            or cache.tokamax_version != tokamax.__version__
        ):
            return None
        return cache
    except (OSError, ValueError, TypeError):
        return None


def parameter(shape, rngs, scale=0.02):
    return nnx.Param(jax.random.normal(rngs.params(), shape) * scale)


class Linear(nnx.Module):
    def __init__(self, dim_in, dim_out, *, rngs, bias=True, dtype=jnp.float32):
        self.weight = parameter((dim_out, dim_in), rngs)
        self.bias = nnx.Param(jnp.zeros(dim_out)) if bias else None
        self.dtype = dtype

    def __call__(self, x):
        x = jnp.matmul(
            x.astype(self.dtype),
            self.weight[...].astype(self.dtype).T,
            precision=jax.lax.Precision.HIGHEST,
        )
        return x + self.bias[...].astype(self.dtype) if self.bias is not None else x


class LayerNorm(nnx.Module):
    def __init__(self, dim, *, eps=1e-5, affine=True):
        self.weight = nnx.Param(jnp.ones(dim)) if affine else None
        self.bias = nnx.Param(jnp.zeros(dim)) if affine else None
        self.eps = eps

    def __call__(self, x):
        dtype = x.dtype
        x = x.astype(jnp.float32)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        x = (x - mean) * jax.lax.rsqrt(var + self.eps)
        if self.weight is not None:
            x = x * self.weight[...] + self.bias[...]
        return x.astype(dtype)


class Conv2d(nnx.Module):
    """NHWC convolution; weights retain PyTorch's OIHW (or IOHW) layout."""

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        *,
        rngs,
        stride=1,
        padding=0,
        bias=True,
        transpose=False,
        dtype=jnp.float32,
    ):
        channels = (dim_in, dim_out) if transpose else (dim_out, dim_in)
        self.weight = parameter((*channels, kernel, kernel), rngs)
        self.bias = nnx.Param(jnp.zeros(dim_out)) if bias else None
        self.stride, self.padding, self.transpose = stride, padding, transpose
        self.dtype = dtype

    def __call__(self, x):
        weight = self.weight[...].astype(self.dtype)
        x = x.astype(self.dtype)
        if self.transpose:
            kernel = weight.shape[-1]
            weight = weight.transpose(1, 0, 2, 3)[:, :, ::-1, ::-1]
            padding = kernel - 1 - self.padding
            x = jax.lax.conv_general_dilated(
                x,
                weight,
                (1, 1),
                ((padding, padding),) * 2,
                lhs_dilation=(self.stride,) * 2,
                dimension_numbers=("NHWC", "OIHW", "NHWC"),
                precision=jax.lax.Precision.HIGHEST,
            )
        else:
            x = jax.lax.conv_general_dilated(
                x,
                weight,
                (self.stride,) * 2,
                ((self.padding, self.padding),) * 2,
                dimension_numbers=("NHWC", "OIHW", "NHWC"),
                precision=jax.lax.Precision.HIGHEST,
            )
        return x + self.bias[...].astype(self.dtype) if self.bias is not None else x


class Mlp(nnx.Module):
    def __init__(
        self, dim, hidden=None, out=None, *, rngs, dtype=jnp.float32, approximate=False
    ):
        self.fc1 = Linear(dim, hidden or dim, rngs=rngs, dtype=dtype)
        self.fc2 = Linear(hidden or dim, out or dim, rngs=rngs, dtype=dtype)
        self.approximate = approximate

    def __call__(self, x):
        return self.fc2(jax.nn.gelu(self.fc1(x), approximate=self.approximate))


def rope_2d(x, positions, frequency=100.0):
    """x is [B, tokens, heads, head_dim]; positions contain (y, x)."""
    dim = x.shape[-1] // 2
    inv = 1.0 / frequency ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
    halves = []
    for axis, part in enumerate(jnp.split(x, 2, axis=-1)):
        angle = (positions[..., axis, None] * inv).astype(x.dtype)
        angle = jnp.concatenate([angle, angle], axis=-1)[:, :, None, :]
        a, b = jnp.split(part, 2, axis=-1)
        halves.append(
            part * jnp.cos(angle) + jnp.concatenate([-b, a], -1) * jnp.sin(angle)
        )
    return jnp.concatenate(halves, axis=-1)


def attention_precision(dtype):
    """Use three Tensor Core products to retain FP32 accuracy on NVIDIA GPUs."""
    if dtype == jnp.float32 and jax.default_backend() == "gpu":
        capability = getattr(jax.devices()[0], "compute_capability", 0) or 0
        if float(capability) >= 8:
            return jax.lax.DotAlgorithmPreset.TF32_TF32_F32_X3
    return jax.lax.Precision.HIGHEST


def attention_implementation(*, tracker=False):
    """Select the tracker override or Tokamax's normal implementation."""
    if tracker:
        selected = os.environ.get("VGGT_TRACK_ATTENTION_IMPLEMENTATION")
        if selected in {"triton", "xla", "xla_chunked", "jax_xla"}:
            return selected
    return "triton" if jax.default_backend() == "gpu" else "xla"


def attention(q, k, v, *, scale=None, precision=None):
    implementation = attention_implementation(tracker=scale == 1.0)
    cache = _attention_autotuning_cache()
    cache_context = (
        cache
        if implementation == "triton" and cache is not None
        else contextlib.nullcontext()
    )
    with cache_context:
        if implementation == "jax_xla":
            return jax.nn.dot_product_attention(
                q, k, v, scale=scale, implementation="xla"
            )
        return tokamax.dot_product_attention(
            q,
            k,
            v,
            scale=scale,
            precision=attention_precision(q.dtype) if precision is None else precision,
            implementation=implementation,
        )


class Attention(nnx.Module):
    def __init__(self, dim, heads, *, rngs, qk_norm=False, eps=1e-5, dtype=jnp.float32):
        self.qkv = Linear(dim, 3 * dim, rngs=rngs, dtype=dtype)
        self.proj = Linear(dim, dim, rngs=rngs, dtype=dtype)
        self.q_norm = LayerNorm(dim // heads, eps=eps) if qk_norm else None
        self.k_norm = LayerNorm(dim // heads, eps=eps) if qk_norm else None
        self.heads = heads

    def __call__(self, x, positions=None):
        b, n, c = x.shape
        q, k, v = jnp.moveaxis(
            self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads), 2, 0
        )
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if positions is not None:
            q, k = rope_2d(q, positions), rope_2d(k, positions)
        return self.proj(attention(q, k, v).reshape(b, n, c))


class LayerScale(nnx.Module):
    def __init__(self, dim, value):
        self.gamma = nnx.Param(jnp.full((dim,), value))

    def __call__(self, x):
        return x * self.gamma[...].astype(x.dtype)


class Block(nnx.Module):
    def __init__(
        self,
        dim,
        heads,
        *,
        rngs,
        eps=1e-5,
        qk_norm=False,
        init_values=0.01,
        dtype=jnp.float32,
    ):
        self.norm1 = LayerNorm(dim, eps=eps)
        self.attn = Attention(
            dim, heads, rngs=rngs, qk_norm=qk_norm, eps=eps, dtype=dtype
        )
        self.ls1 = LayerScale(dim, init_values)
        self.norm2 = LayerNorm(dim, eps=eps)
        self.mlp = Mlp(dim, dim * 4, rngs=rngs, dtype=dtype)
        self.ls2 = LayerScale(dim, init_values)

    def __call__(self, x, positions=None):
        x = x + self.ls1(self.attn(self.norm1(x), positions))
        return x + self.ls2(self.mlp(self.norm2(x)))


def resize_bilinear(x, size, *, align_corners=True):
    """PyTorch interpolate semantics, including align_corners=True used by DPT."""
    for axis, length in zip((1, 2), size):
        old = x.shape[axis]
        if old == length:
            continue
        if align_corners:
            coords = jnp.arange(length, dtype=jnp.float32) * (
                (old - 1) / max(length - 1, 1)
            )
        else:
            coords = jnp.maximum(
                (jnp.arange(length, dtype=jnp.float32) + 0.5) * old / length - 0.5, 0
            )
        lo = jnp.floor(coords).astype(jnp.int32)
        hi = jnp.minimum(lo + 1, old - 1)
        shape = [1] * x.ndim
        shape[axis] = length
        weight = (coords - lo).reshape(shape).astype(x.dtype)
        x = (
            jnp.take(x, lo, axis=axis) * (1 - weight)
            + jnp.take(x, hi, axis=axis) * weight
        )
    return x


def sample_bilinear(x, coords, *, padding="border"):
    """Sample NHWC maps at pixel coordinates (..., x/y), like grid_sample."""
    b, h, w, _ = x.shape
    output_shape = coords.shape[:-1]
    coords = jax.lax.stop_gradient(coords).astype(x.dtype).reshape(b, -1, 2)
    # Reproduce the normalized-grid round trip of the official bilinear_sampler.
    # Its float32 rounding matters at the tracker’s high spatial frequencies.
    scale = jnp.asarray([2 / max(w - 1, 1), 2 / max(h - 1, 1)], x.dtype)
    normalized = jax.lax.optimization_barrier(coords * scale)
    normalized = jax.lax.optimization_barrier(normalized - 1)
    pixel = jax.lax.optimization_barrier(normalized + 1) * jnp.asarray(
        [(w - 1) / 2, (h - 1) / 2], x.dtype
    )
    cx, cy = pixel[..., 0], pixel[..., 1]
    if padding == "border":
        cx, cy = jnp.clip(cx, 0, w - 1), jnp.clip(cy, 0, h - 1)
    x0, y0 = jnp.floor(cx).astype(jnp.int32), jnp.floor(cy).astype(jnp.int32)
    dx, dy = cx - x0, cy - y0
    batch = jnp.arange(b)[:, None]

    def gather(ix, iy):
        value = x[batch, jnp.clip(iy, 0, h - 1), jnp.clip(ix, 0, w - 1)]
        if padding == "zeros":
            valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
            value = value * valid[..., None]
        return value

    out = (
        gather(x0, y0) * ((1 - dx) * (1 - dy))[..., None]
        + gather(x0 + 1, y0) * (dx * (1 - dy))[..., None]
        + gather(x0, y0 + 1) * ((1 - dx) * dy)[..., None]
        + gather(x0 + 1, y0 + 1) * (dx * dy)[..., None]
    )
    return out.reshape(*output_shape, x.shape[-1])


def sincos_grid(height, width, dim, *, aspect_ratio=None, base=100.0):
    """Compute the fixed embeddings in NumPy float64, as in official VGGT."""
    if aspect_ratio is None:
        xx, yy = np.meshgrid(
            np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
        )
    else:
        span = np.sqrt(aspect_ratio**2 + 1)
        xx, yy = np.meshgrid(
            np.linspace(
                -aspect_ratio / span * (width - 1) / width,
                aspect_ratio / span * (width - 1) / width,
                width,
                dtype=np.float32,
            ),
            np.linspace(
                -1 / span * (height - 1) / height,
                1 / span * (height - 1) / height,
                height,
                dtype=np.float32,
            ),
        )
    omega = 1.0 / base ** (np.arange(dim // 4, dtype=np.float64) / (dim / 4))
    parts = []
    for pos in (xx, yy):
        angle = pos[..., None] * omega
        parts.extend([np.sin(angle), np.cos(angle)])
    return jnp.asarray(np.concatenate(parts, -1).astype(np.float32))[None]
