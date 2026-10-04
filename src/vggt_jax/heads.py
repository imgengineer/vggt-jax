# Copyright (c) Meta Platforms, Inc. and affiliates.
# JAX adaptation; see LICENSE and NOTICE.
import jax
import jax.numpy as jnp
from flax import nnx

from .layers import Block, Conv2d, LayerNorm, Linear, Mlp, resize_bilinear, sincos_grid


class CameraHead(nnx.Module):
    def __init__(self, config, *, rngs):
        dim = config.embed_dim * 2
        self.trunk = nnx.List(
            [
                Block(dim, config.num_heads, rngs=rngs)
                for _ in range(config.camera_depth)
            ]
        )
        self.token_norm = LayerNorm(dim)
        self.trunk_norm = LayerNorm(dim)
        self.empty_pose_tokens = nnx.Param(jnp.zeros((1, 1, 9)))
        self.embed_pose = Linear(9, dim, rngs=rngs)
        self.poseLN_modulation = nnx.Dict({"1": Linear(dim, 3 * dim, rngs=rngs)})
        self.adaln_norm = LayerNorm(dim, affine=False, eps=1e-6)
        self.pose_branch = Mlp(dim, dim // 2, 9, rngs=rngs)

    def __call__(self, tokens, num_iterations=4):
        pose_tokens = self.token_norm(tokens[-1][:, :, 0].astype(jnp.float32))
        pred, outputs = None, []
        for _ in range(num_iterations):
            pose = (
                jnp.broadcast_to(
                    self.empty_pose_tokens[...], (*pose_tokens.shape[:2], 9)
                )
                if pred is None
                else jax.lax.stop_gradient(pred)
            )
            if pred is not None:
                pred = pose
            module_input = self.embed_pose(pose)
            modulation = self.poseLN_modulation["1"](jax.nn.silu(module_input))
            shift, scale, gate = jnp.split(modulation, 3, axis=-1)
            x = pose_tokens + gate * (
                self.adaln_norm(pose_tokens) * (1 + scale) + shift
            )
            for block in self.trunk:
                x = block(x)
            delta = self.pose_branch(self.trunk_norm(x))
            pred = delta if pred is None else pred + delta
            outputs.append(
                jnp.concatenate([pred[..., :7], jax.nn.relu(pred[..., 7:])], -1)
            )
        return outputs


class ResidualConvUnit(nnx.Module):
    def __init__(self, features, *, rngs):
        self.conv1 = Conv2d(features, features, 3, padding=1, rngs=rngs)
        self.conv2 = Conv2d(features, features, 3, padding=1, rngs=rngs)

    def __call__(self, x):
        # Upstream's first ReLU is in-place: the residual also uses relu(x).
        x = jax.nn.relu(x)
        return x + self.conv2(jax.nn.relu(self.conv1(x)))


class FeatureFusionBlock(nnx.Module):
    def __init__(self, features, *, rngs, has_residual=True):
        self.out_conv = Conv2d(features, features, 1, rngs=rngs)
        self.resConfUnit1 = (
            ResidualConvUnit(features, rngs=rngs) if has_residual else None
        )
        self.resConfUnit2 = ResidualConvUnit(features, rngs=rngs)

    def __call__(self, x, residual=None, size=None):
        if self.resConfUnit1 is not None:
            x = x + self.resConfUnit1(residual)
        x = self.resConfUnit2(x)
        size = size or (x.shape[1] * 2, x.shape[2] * 2)
        return self.out_conv(resize_bilinear(x, size))


class Scratch(nnx.Module):
    def __init__(self, channels, features, output_dim, *, rngs, feature_only):
        for i, channel in enumerate(channels, 1):
            setattr(
                self,
                f"layer{i}_rn",
                Conv2d(channel, features, 3, padding=1, bias=False, rngs=rngs),
            )
            setattr(
                self,
                f"refinenet{i}",
                FeatureFusionBlock(features, rngs=rngs, has_residual=i != 4),
            )
        self.output_conv1 = Conv2d(
            features,
            features if feature_only else features // 2,
            3,
            padding=1,
            rngs=rngs,
        )
        self.output_conv2 = (
            None
            if feature_only
            else nnx.Dict(
                {
                    "0": Conv2d(features // 2, 32, 3, padding=1, rngs=rngs),
                    "2": Conv2d(32, output_dim, 1, rngs=rngs),
                }
            )
        )

    def __call__(self, inputs):
        a, b, c, d = [getattr(self, f"layer{i}_rn")(x) for i, x in enumerate(inputs, 1)]
        x = self.refinenet4(d, size=c.shape[1:3])
        x = self.refinenet3(x, c, size=b.shape[1:3])
        x = self.refinenet2(x, b, size=a.shape[1:3])
        x = self.refinenet1(x, a)
        return self.output_conv1(x)


class DPTHead(nnx.Module):
    def __init__(
        self, config, output_dim=4, *, rngs, activation="inv_log", feature_only=False
    ):
        dim = 2 * config.embed_dim
        channels = config.dpt_channels
        features = config.track_features if feature_only else config.dpt_features
        self.norm = LayerNorm(dim)
        self.projects = nnx.List([Conv2d(dim, oc, 1, rngs=rngs) for oc in channels])
        self.resize_layers = nnx.List(
            [
                Conv2d(
                    channels[0], channels[0], 4, stride=4, transpose=True, rngs=rngs
                ),
                Conv2d(
                    channels[1], channels[1], 2, stride=2, transpose=True, rngs=rngs
                ),
                None,
                Conv2d(channels[3], channels[3], 3, stride=2, padding=1, rngs=rngs),
            ]
        )
        self.scratch = Scratch(
            channels, features, output_dim, rngs=rngs, feature_only=feature_only
        )
        self.config = config
        self.activation = activation
        self.feature_only = feature_only

    def _pos_embed(self, x, h, w):
        return x + 0.1 * sincos_grid(
            x.shape[1], x.shape[2], x.shape[-1], aspect_ratio=w / h
        )

    def _forward(self, tokens, images, patch_start_idx):
        b, s, _, h, w = images.shape
        ph, pw = h // self.config.patch_size, w // self.config.patch_size
        features = []
        for i, idx in enumerate(self.config.intermediate_layer_idx):
            x = self.norm(tokens[idx][:, :, patch_start_idx:].astype(jnp.float32))
            x = self.projects[i](x.reshape(b * s, ph, pw, -1))
            if not self.feature_only:
                x = self._pos_embed(x, h, w)
            if self.resize_layers[i] is not None:
                x = self.resize_layers[i](x)
            features.append(x)
        x = self.scratch(features)
        ratio = 2 if self.feature_only else 1
        x = resize_bilinear(
            x,
            (
                ph * self.config.patch_size // ratio,
                pw * self.config.patch_size // ratio,
            ),
        )
        if self.feature_only:
            return x.reshape(b, s, *x.shape[1:]).transpose(0, 1, 4, 2, 3)
        x = self._pos_embed(x, h, w)
        x = self.scratch.output_conv2["2"](
            jax.nn.relu(self.scratch.output_conv2["0"](x))
        )
        xyz, conf = x[..., :-1], 1 + jnp.exp(x[..., -1])
        if self.activation == "exp":
            xyz = jnp.exp(xyz)
        else:
            xyz = jnp.sign(xyz) * jnp.expm1(jnp.abs(xyz))
        return xyz.reshape(b, s, *xyz.shape[1:]), conf.reshape(b, s, *conf.shape[1:])

    def __call__(self, tokens, images, patch_start_idx, frames_chunk_size=8):
        s = images.shape[1]
        if frames_chunk_size is None or frames_chunk_size >= s:
            return self._forward(tokens, images, patch_start_idx)
        if frames_chunk_size < 1:
            raise ValueError("frames_chunk_size must be positive")
        outputs = []
        for start in range(0, s, frames_chunk_size):
            chunk = [
                x[:, start : start + frames_chunk_size] if x is not None else None
                for x in tokens
            ]
            outputs.append(
                self._forward(
                    chunk, images[:, start : start + frames_chunk_size], patch_start_idx
                )
            )
        if self.feature_only:
            return jnp.concatenate(outputs, axis=1)
        return tuple(
            jnp.concatenate([out[i] for out in outputs], axis=1) for i in (0, 1)
        )
