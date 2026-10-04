# Copyright (c) Meta Platforms, Inc. and affiliates.
# JAX adaptation; see LICENSE and NOTICE.
import jax
import jax.numpy as jnp
from flax import nnx

from .layers import Block, Conv2d, LayerNorm, parameter


class PatchEmbed(nnx.Module):
    def __init__(self, config, *, rngs, dtype):
        self.proj = Conv2d(
            3,
            config.embed_dim,
            config.patch_size,
            stride=config.patch_size,
            rngs=rngs,
            dtype=dtype,
        )

    def __call__(self, images):
        x = self.proj(images)
        return x.reshape(x.shape[0], -1, x.shape[-1])


class DinoVisionTransformer(nnx.Module):
    def __init__(self, config, *, rngs, dtype):
        dim = config.embed_dim
        self.patch_embed = PatchEmbed(config, rngs=rngs, dtype=dtype)
        self.cls_token = parameter((1, 1, dim), rngs, 1e-6)
        self.pos_embed = parameter(
            (1, (config.img_size // config.patch_size) ** 2 + 1, dim), rngs
        )
        self.register_tokens = parameter(
            (1, config.num_register_tokens, dim), rngs, 1e-6
        )
        self.mask_token = nnx.Param(jnp.zeros((1, dim)))
        self.blocks = nnx.List(
            [
                Block(
                    dim,
                    config.num_heads,
                    eps=1e-6,
                    init_values=1.0,
                    rngs=rngs,
                    dtype=dtype,
                )
                for _ in range(config.dino_depth)
            ]
        )
        self.norm = LayerNorm(dim, eps=1e-6)
        self.patch_size = config.patch_size
        self.dtype = dtype

    def __call__(self, images):
        b, h, w, _ = images.shape
        ph, pw = h // self.patch_size, w // self.patch_size
        x = self.patch_embed(images)
        cls = jnp.broadcast_to(self.cls_token[...], (b, 1, x.shape[-1]))
        x = jnp.concatenate([cls, x], axis=1)
        pos = self.pos_embed[...]
        side = int((pos.shape[1] - 1) ** 0.5)
        if (ph, pw) != (side, side):
            patch_pos = pos[:, 1:].reshape(1, side, side, -1)
            patch_pos = jax.image.resize(
                patch_pos,
                (1, ph, pw, pos.shape[-1]),
                method="cubic",
                antialias=True,
                precision=jax.lax.Precision.HIGHEST,
            )
            pos = jnp.concatenate(
                [pos[:, :1], patch_pos.reshape(1, ph * pw, -1)], axis=1
            )
        # The official model adds float32 positional embeddings before autocast.
        x = x + pos
        regs = jnp.broadcast_to(
            self.register_tokens[...], (b, self.register_tokens.shape[1], x.shape[-1])
        )
        x = jnp.concatenate([x[:, :1], regs, x[:, 1:]], axis=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 1 + self.register_tokens.shape[1] :]


class Aggregator(nnx.Module):
    def __init__(self, config, *, rngs, dtype):
        self.patch_embed = DinoVisionTransformer(config, rngs=rngs, dtype=dtype)
        self.frame_blocks = nnx.List(
            [
                Block(
                    config.embed_dim,
                    config.num_heads,
                    qk_norm=True,
                    rngs=rngs,
                    dtype=dtype,
                )
                for _ in range(config.depth)
            ]
        )
        self.global_blocks = nnx.List(
            [
                Block(
                    config.embed_dim,
                    config.num_heads,
                    qk_norm=True,
                    rngs=rngs,
                    dtype=dtype,
                )
                for _ in range(config.depth)
            ]
        )
        self.camera_token = parameter((1, 2, 1, config.embed_dim), rngs, 1e-6)
        self.register_token = parameter(
            (1, 2, config.num_register_tokens, config.embed_dim), rngs, 1e-6
        )
        self.config = config
        self.patch_start_idx = 1 + config.num_register_tokens

    def __call__(self, images):
        b, s, c, h, w = images.shape
        mean = jnp.asarray([0.485, 0.456, 0.406])[None, None, :, None, None]
        std = jnp.asarray([0.229, 0.224, 0.225])[None, None, :, None, None]
        images = (
            ((images - mean) / std).transpose(0, 1, 3, 4, 2).reshape(b * s, h, w, c)
        )
        patches = self.patch_embed(images)

        def expand(token):
            indexes = jnp.minimum(jnp.arange(s), 1)
            return jnp.broadcast_to(
                token[:, indexes], (b, s, *token.shape[2:])
            ).reshape(b * s, -1, self.config.embed_dim)

        x = jnp.concatenate(
            [expand(self.camera_token[...]), expand(self.register_token[...]), patches],
            axis=1,
        )
        n, d = x.shape[1:]
        yy, xx = jnp.meshgrid(
            jnp.arange(h // self.config.patch_size),
            jnp.arange(w // self.config.patch_size),
            indexing="ij",
        )
        grid = jnp.stack([yy, xx], -1).reshape(-1, 2) + 1
        grid = jnp.concatenate(
            [jnp.zeros((self.patch_start_idx, 2), jnp.int32), grid], 0
        )
        frame_pos = jnp.broadcast_to(grid, (b * s, n, 2))
        global_pos = frame_pos.reshape(b, s * n, 2)
        cached = set(self.config.intermediate_layer_idx) | {self.config.depth - 1}
        outputs = []
        for i, (frame, global_block) in enumerate(
            zip(self.frame_blocks, self.global_blocks)
        ):
            local = frame(x.reshape(b * s, n, d), frame_pos).reshape(b, s, n, d)
            x = global_block(local.reshape(b, s * n, d), global_pos).reshape(b, s, n, d)
            outputs.append(jnp.concatenate([local, x], -1) if i in cached else None)
        return outputs, self.patch_start_idx
