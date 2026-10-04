# Copyright (c) Meta Platforms, Inc. and affiliates.
# JAX adaptation; see LICENSE and NOTICE.
import jax
import jax.numpy as jnp
from flax import nnx

from .heads import DPTHead
from .layers import (
    LayerNorm,
    Linear,
    Mlp,
    attention,
    parameter,
    sample_bilinear,
    sincos_grid,
)


class MultiheadAttention(nnx.Module):
    def __init__(self, dim, heads, *, rngs):
        self.in_proj_weight = parameter((3 * dim, dim), rngs)
        self.in_proj_bias = nnx.Param(jnp.zeros(3 * dim))
        self.out_proj = Linear(dim, dim, rngs=rngs)
        self.heads = heads

    def __call__(self, x, context=None):
        context = x if context is None else context
        ws = jnp.split(self.in_proj_weight[...], 3, axis=0)
        bs = jnp.split(self.in_proj_bias[...], 3, axis=0)
        q, k, v = [
            jnp.matmul(t, w.T, precision=jax.lax.Precision.HIGHEST) + b
            for t, w, b in zip((x, context, context), ws, bs)
        ]
        b, n, c = q.shape
        q = q.reshape(b, n, self.heads, c // self.heads)
        k = k.reshape(b, k.shape[1], self.heads, c // self.heads)
        v = v.reshape(b, v.shape[1], self.heads, c // self.heads)
        # nn.MultiheadAttention scales Q before matmul (unlike torch SDPA).
        q = jax.lax.optimization_barrier(q * ((c // self.heads) ** -0.5))
        # The high-frequency flow encoding amplifies changes between iterations.
        result = attention(q, k, v, scale=1.0, precision=jax.lax.Precision.HIGHEST)
        return self.out_proj(result.reshape(b, n, c))


class AttnBlock(nnx.Module):
    def __init__(self, dim, heads, *, rngs, cross=False):
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.mlp = Mlp(dim, dim * 4, rngs=rngs)
        self.cross = cross
        if cross:
            self.norm_context = LayerNorm(dim)
            self.cross_attn = MultiheadAttention(dim, heads, rngs=rngs)
        else:
            self.attn = MultiheadAttention(dim, heads, rngs=rngs)

    def __call__(self, x, context=None):
        # Upstream applies the residual to the normalized input.
        x = self.norm1(x)
        update = (
            self.cross_attn(x, self.norm_context(context))
            if self.cross
            else self.attn(x)
        )
        x = x + update
        return x + self.mlp(self.norm2(x))


class EfficientUpdateFormer(nnx.Module):
    def __init__(self, config, dim_in, dim_out, *, rngs):
        hidden = config.track_hidden
        self.input_norm = LayerNorm(dim_in)
        self.input_transform = Linear(dim_in, hidden, rngs=rngs)
        self.output_norm = LayerNorm(hidden)
        self.flow_head = Linear(hidden, dim_out, rngs=rngs)
        # Preserve the spelling used in the official checkpoint.
        self.virual_tracks = parameter((1, config.track_virtual, 1, hidden), rngs, 1.0)
        for name in (
            "time_blocks",
            "space_virtual_blocks",
            "space_point2virtual_blocks",
            "space_virtual2point_blocks",
        ):
            setattr(
                self,
                name,
                nnx.List(
                    [
                        AttnBlock(
                            hidden, config.track_heads, rngs=rngs, cross="2" in name
                        )
                        for _ in range(config.track_depth)
                    ]
                ),
            )

    def __call__(self, x):
        init = self.input_transform(self.input_norm(x))
        b, n, s, c = init.shape
        virtual = jnp.broadcast_to(
            self.virual_tracks[...], (b, self.virual_tracks.shape[1], s, c)
        )
        x = jnp.concatenate([init, virtual], axis=1)
        total = x.shape[1]
        for i, time_block in enumerate(self.time_blocks):
            x = time_block(x.reshape(b * total, s, c)).reshape(b, total, s, c)
            space = x.transpose(0, 2, 1, 3).reshape(b * s, total, c)
            points, virtual = space[:, :n], space[:, n:]
            virtual = self.space_virtual2point_blocks[i](virtual, points)
            virtual = self.space_virtual_blocks[i](virtual)
            points = self.space_point2virtual_blocks[i](points, virtual)
            x = (
                jnp.concatenate([points, virtual], axis=1)
                .reshape(b, s, total, c)
                .transpose(0, 2, 1, 3)
            )
        return self.flow_head(self.output_norm(x[:, :n] + init))


def flow_embedding(flows, dim):
    div = jnp.arange(0, dim, 2, dtype=jnp.float32) * (1000.0 / dim)
    parts = []
    for axis in (0, 1):
        angle = flows[..., axis, None] * div
        parts.append(
            jnp.stack([jnp.sin(angle), jnp.cos(angle)], -1).reshape(
                *flows.shape[:-1], dim
            )
        )
    return jnp.concatenate(parts, -1)


def correlation_pyramid(fmaps, levels):
    pyramid = [fmaps]
    for _ in range(levels - 1):
        b, s, h, w, c = fmaps.shape
        if h < 2 or w < 2:
            raise ValueError(
                f"Tracking requires feature maps of at least {2 ** (levels - 1)} pixels per side"
            )
        fmaps = fmaps[:, :, : h // 2 * 2, : w // 2 * 2]
        fmaps = fmaps.reshape(b, s, h // 2, 2, w // 2, 2, c).mean(axis=(3, 5))
        pyramid.append(fmaps)
    return pyramid


def sample_correlations(pyramid, targets, coords, radius):
    b, s, n, c = targets.shape
    delta = jnp.arange(-radius, radius + 1, dtype=coords.dtype)
    a, d = jnp.meshgrid(delta, delta, indexing="ij")
    grid = jnp.stack([a, d], -1)[None]
    outputs = []
    for i, maps in enumerate(pyramid):
        h, w = maps.shape[2:4]
        corr = (
            jnp.einsum(
                "bsnc,bshwc->bsnhw", targets, maps, precision=jax.lax.Precision.HIGHEST
            )
            / c**0.5
        )
        sample_coords = coords.reshape(b * s * n, 1, 1, 2) / 2**i + grid
        sampled = sample_bilinear(
            corr.reshape(b * s * n, h, w, 1), sample_coords, padding="zeros"
        )
        outputs.append(sampled.reshape(b, s, n, -1))
    return jnp.concatenate(outputs, -1)


class BaseTrackerPredictor(nnx.Module):
    def __init__(self, config, *, rngs):
        dim = config.track_features
        self.corr_mlp = Mlp(
            config.track_corr_levels * (config.track_corr_radius * 2 + 1) ** 2,
            config.track_hidden,
            dim,
            rngs=rngs,
        )
        self.query_ref_token = parameter((1, 2, dim * 3 + 4), rngs, 1.0)
        self.updateformer = EfficientUpdateFormer(
            config, dim * 3 + 4, dim + 2, rngs=rngs
        )
        self.fmap_norm = LayerNorm(dim)
        self.ffeat_norm = LayerNorm(dim)  # GroupNorm(1, C) on [B*N*S, C].
        self.ffeat_updater = nnx.Dict({"0": Linear(dim, dim, rngs=rngs)})
        self.vis_predictor = nnx.Dict({"0": Linear(dim, 1, rngs=rngs)})
        self.conf_predictor = nnx.Dict({"0": Linear(dim, 1, rngs=rngs)})
        self.config = config

    def __call__(self, query_points, fmaps, iters=4):
        fmaps = self.fmap_norm(fmaps.transpose(0, 1, 3, 4, 2))
        b, s, h, w, c = fmaps.shape
        n = query_points.shape[1]
        queries = query_points / 2.0
        coords = jnp.broadcast_to(queries[:, None], (b, s, n, 2))
        features = sample_bilinear(fmaps[:, 0], queries)
        features = jnp.broadcast_to(features[:, None], (b, s, n, c))
        pyramid = correlation_pyramid(fmaps, self.config.track_corr_levels)
        pos = jnp.broadcast_to(
            sincos_grid(h, w, c * 3 + 4, base=10000.0), (b, h, w, c * 3 + 4)
        )
        ref_token = self.query_ref_token[:, jnp.minimum(jnp.arange(s), 1)]
        predictions = []
        for _ in range(iters):
            coords = jax.lax.stop_gradient(coords)
            corr = sample_correlations(
                pyramid, features, coords, self.config.track_corr_radius
            )
            corr = self.corr_mlp(corr.transpose(0, 2, 1, 3).reshape(b * n, s, -1))
            flows = (coords - coords[:, :1]).transpose(0, 2, 1, 3).reshape(b * n, s, 2)
            flow = jnp.concatenate(
                [flow_embedding(flows, c // 2), flows / 518.0, flows / 518.0], -1
            )
            old_features = features.transpose(0, 2, 1, 3).reshape(b * n, s, c)
            x = jnp.concatenate([flow, corr, old_features], -1)
            sampled_pos = sample_bilinear(pos, coords[:, 0]).reshape(b * n, 1, -1)
            x = (x + sampled_pos + ref_token).reshape(b, n, s, -1)
            delta = self.updateformer(x).reshape(b * n, s, c + 2)
            delta_features = self.ffeat_norm(delta[..., 2:].reshape(b * n * s, c))
            new_features = jax.nn.gelu(
                self.ffeat_updater["0"](delta_features), approximate=False
            )
            features = (
                (old_features + new_features.reshape(b * n, s, c))
                .reshape(b, n, s, c)
                .transpose(0, 2, 1, 3)
            )
            coords = coords + delta[..., :2].reshape(b, n, s, 2).transpose(0, 2, 1, 3)
            coords = coords.at[:, 0].set(queries)
            predictions.append(coords * 2)
        vis = jax.nn.sigmoid(self.vis_predictor["0"](features)[..., 0])
        conf = jax.nn.sigmoid(self.conf_predictor["0"](features)[..., 0])
        return predictions, vis, conf


class TrackHead(nnx.Module):
    def __init__(self, config, *, rngs):
        self.feature_extractor = DPTHead(config, feature_only=True, rngs=rngs)
        self.tracker = BaseTrackerPredictor(config, rngs=rngs)

    def __call__(self, tokens, images, patch_start_idx, query_points, iters=4):
        return self.tracker(
            query_points, self.feature_extractor(tokens, images, patch_start_idx), iters
        )
