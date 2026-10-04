# Copyright (c) Meta Platforms, Inc. and affiliates.
# JAX adaptation; see LICENSE and NOTICE.
import jax.numpy as jnp
from flax import nnx

from .aggregator import Aggregator
from .config import VGGTConfig
from .heads import CameraHead, DPTHead
from .tracking import TrackHead


class VGGT(nnx.Module):
    """Official VGGT architecture. Inputs/outputs use the PyTorch API's layouts.

    ``dtype`` controls the aggregator's linear/convolution computations; camera,
    depth, point and tracking heads compute in float32 for numerical stability.
    """

    def __init__(
        self,
        config=None,
        *,
        rngs=None,
        dtype=jnp.float32,
        enable_camera=True,
        enable_depth=True,
        enable_point=True,
        enable_track=True,
    ):
        config = config or VGGTConfig()
        rngs = rngs or nnx.Rngs(0)
        self.config = config
        self.aggregator = Aggregator(config, rngs=rngs, dtype=dtype)
        self.camera_head = CameraHead(config, rngs=rngs) if enable_camera else None
        self.depth_head = (
            DPTHead(config, 2, activation="exp", rngs=rngs) if enable_depth else None
        )
        self.point_head = DPTHead(config, 4, rngs=rngs) if enable_point else None
        self.track_head = TrackHead(config, rngs=rngs) if enable_track else None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path="facebook/VGGT-1B",
        *,
        revision=None,
        **kwargs,
    ):
        """Load official .safetensors/.pt or a Hugging Face repository, without random allocation."""
        from .weights import load_pretrained

        model = nnx.eval_shape(lambda: cls(**kwargs))
        load_pretrained(model, pretrained_model_name_or_path, revision=revision)
        return model

    def load_state_dict(self, state_dict, *, strict=True):
        from .weights import load_state_dict

        return load_state_dict(self, state_dict, strict=strict)

    def jit(self):
        """Create a reusable predictor without repeatedly traversing the NNX graph.

        Call after loading weights. Parameter value updates remain visible; create
        a new predictor after changing the model's structure.
        """
        return nnx.jit_partial(lambda m, x, q=None: m(x, q), self, graph=False)

    def __call__(self, images, query_points=None):
        images = jnp.asarray(images, dtype=jnp.float32)
        if images.ndim == 4:
            images = images[None]
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("images must have shape [S, 3, H, W] or [B, S, 3, H, W]")
        b, s, _, h, w = images.shape
        if b < 1 or s < 1 or min(h, w) < self.config.patch_size:
            raise ValueError("images must contain at least one frame and one patch")
        if h % self.config.patch_size or w % self.config.patch_size:
            raise ValueError("Image height and width must be divisible by patch_size")
        if query_points is not None:
            query_points = jnp.asarray(query_points, jnp.float32)
            if query_points.ndim == 2:
                query_points = query_points[None]
            if (
                query_points.ndim != 3
                or query_points.shape[0] != b
                or query_points.shape[-1] != 2
                or query_points.shape[1] < 1
            ):
                raise ValueError(
                    "query_points must have shape [N, 2] or [B, N, 2] with matching batch size"
                )
        tokens, start = self.aggregator(images)
        predictions = {"images": images}
        if self.camera_head is not None:
            poses = self.camera_head(tokens)
            predictions.update(pose_enc=poses[-1], pose_enc_list=poses)
        if self.depth_head is not None:
            predictions["depth"], predictions["depth_conf"] = self.depth_head(
                tokens, images, start
            )
        if self.point_head is not None:
            predictions["world_points"], predictions["world_points_conf"] = (
                self.point_head(tokens, images, start)
            )
        if self.track_head is not None and query_points is not None:
            tracks, vis, conf = self.track_head(tokens, images, start, query_points)
            predictions.update(track=tracks[-1], vis=vis, conf=conf)
        return predictions
