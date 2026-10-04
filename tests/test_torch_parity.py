import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from vggt_jax import VGGTConfig
from vggt_jax.aggregator import DinoVisionTransformer
from vggt_jax.heads import CameraHead, DPTHead
from vggt_jax.layers import Conv2d, resize_bilinear, sample_bilinear
from vggt_jax.tracking import BaseTrackerPredictor
from vggt_jax.weights import load_checkpoint, load_state_dict

torch = pytest.importorskip("torch")
torch.set_num_threads(4)
REFERENCE = Path(__file__).resolve().parents[1] / ".reference/vggt"


def reference_imports():
    if not REFERENCE.exists():
        pytest.skip(
            "Run scripts/validate_parity.py to fetch the official reference first"
        )
    sys.path.insert(0, str(REFERENCE))


@pytest.mark.parametrize(
    "transpose,stride,padding,kernel",
    [(False, 2, 1, 3), (True, 4, 0, 4), (True, 2, 0, 2)],
)
def test_convolution_layout(transpose, stride, padding, kernel, tmp_path):
    torch.manual_seed(7)
    cls = torch.nn.ConvTranspose2d if transpose else torch.nn.Conv2d
    reference = cls(3, 5, kernel, stride=stride, padding=padding)
    actual = Conv2d(
        3,
        5,
        kernel,
        stride=stride,
        padding=padding,
        transpose=transpose,
        rngs=nnx.Rngs(0),
    )
    path = tmp_path / "model.pt"
    torch.save(reference.state_dict(), path)
    load_checkpoint(actual, path)
    x = torch.randn(2, 3, 7, 9)
    expected = reference(x).detach().numpy().transpose(0, 2, 3, 1)
    result = nnx.jit(lambda m, x: m(x))(
        actual, jnp.asarray(x.numpy().transpose(0, 2, 3, 1))
    )
    np.testing.assert_allclose(result, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("padding", ["zeros", "border"])
@pytest.mark.parametrize("shape", [(5, 7), (1, 3), (3, 1)])
def test_pixel_sampling(padding, shape):
    reference_imports()
    from vggt.heads.track_modules.utils import bilinear_sampler

    rng = np.random.default_rng(3)
    x = rng.normal(size=(2, 3, *shape)).astype(np.float32)
    coords = rng.uniform(-2, 9, size=(2, 5, 4, 2)).astype(np.float32)
    expected = bilinear_sampler(
        torch.from_numpy(x), torch.from_numpy(coords), padding_mode=padding
    ).numpy()
    result = sample_bilinear(
        jnp.asarray(x.transpose(0, 2, 3, 1)), jnp.asarray(coords), padding=padding
    )
    np.testing.assert_allclose(
        np.asarray(result).transpose(0, 3, 1, 2), expected, rtol=2e-5, atol=3e-6
    )


def test_align_corners_resize():
    x = np.random.default_rng(2).normal(size=(2, 3, 7, 9)).astype(np.float32)
    for size in ((13, 17), (3, 4), (1, 1)):
        expected = torch.nn.functional.interpolate(
            torch.from_numpy(x), size, mode="bilinear", align_corners=True
        )
        result = resize_bilinear(jnp.asarray(x.transpose(0, 2, 3, 1)), size)
        np.testing.assert_allclose(
            np.asarray(result).transpose(0, 3, 1, 2),
            expected.numpy(),
            rtol=2e-5,
            atol=3e-6,
        )


def config():
    return VGGTConfig(
        img_size=56,
        embed_dim=32,
        num_heads=4,
        depth=4,
        dino_depth=2,
        camera_depth=1,
        intermediate_layer_idx=(0, 1, 2, 3),
        dpt_features=16,
        dpt_channels=(8, 16, 32, 32),
        track_features=16,
        track_hidden=32,
        track_depth=1,
        track_heads=4,
        track_virtual=4,
        track_corr_levels=2,
        track_corr_radius=1,
    )


def test_dino_rectangular_positional_interpolation():
    reference_imports()
    from vggt.layers.vision_transformer import DinoVisionTransformer as TorchDino

    torch.manual_seed(2)
    c = config()
    reference = TorchDino(
        img_size=c.img_size,
        patch_size=14,
        embed_dim=32,
        depth=2,
        num_heads=4,
        block_chunks=0,
        num_register_tokens=4,
        init_values=1.0,
        interpolate_antialias=True,
        interpolate_offset=0.0,
    ).eval()
    actual = DinoVisionTransformer(c, rngs=nnx.Rngs(0), dtype=jnp.float32)
    load_state_dict(actual, reference.state_dict())
    for h, w in ((56, 56), (28, 42), (70, 84)):
        x = torch.randn(1, 3, h, w)
        with torch.no_grad():
            expected = reference(x)["x_norm_patchtokens"].numpy()
        result = nnx.jit(lambda m, x: m(x))(
            actual, jnp.asarray(x.numpy().transpose(0, 2, 3, 1))
        )
        np.testing.assert_allclose(result, expected, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("branch", ["depth", "point", "feature"])
def test_dpt_heads_and_chunking(branch):
    reference_imports()
    from vggt.heads.dpt_head import DPTHead as TorchDPT

    torch.manual_seed(4)
    c = config()
    feature_only = branch == "feature"
    activation = "exp" if branch == "depth" else "inv_log"
    output_dim = 2 if branch == "depth" else 4
    reference = TorchDPT(
        64,
        features=16,
        out_channels=list(c.dpt_channels),
        intermediate_layer_idx=[0, 1, 2, 3],
        output_dim=output_dim,
        activation=activation,
        feature_only=feature_only,
        pos_embed=not feature_only,
        down_ratio=2 if feature_only else 1,
    ).eval()
    actual = DPTHead(
        c,
        output_dim,
        activation=activation,
        feature_only=feature_only,
        rngs=nnx.Rngs(0),
    )
    load_state_dict(actual, reference.state_dict())
    tokens = [torch.randn(1, 3, 11, 64) for _ in range(4)]
    images = torch.zeros(1, 3, 3, 28, 42)
    with torch.no_grad():
        expected = reference(tokens, images, 5, frames_chunk_size=2)
    result = nnx.jit(lambda m, t, x: m(t, x, 5, frames_chunk_size=2))(
        actual, [jnp.asarray(t.numpy()) for t in tokens], jnp.asarray(images.numpy())
    )
    for a, b in zip(jax.tree.leaves(result), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b.numpy(), rtol=2e-4, atol=2e-5)


def test_iterative_camera():
    reference_imports()
    from vggt.heads.camera_head import CameraHead as TorchCamera

    c = config()
    reference = TorchCamera(dim_in=64, trunk_depth=1, num_heads=4).eval()
    actual = CameraHead(c, rngs=nnx.Rngs(0))
    load_state_dict(actual, reference.state_dict())
    tokens = torch.randn(2, 3, 11, 64)
    with torch.no_grad():
        expected = reference([tokens])
    result = nnx.jit(lambda m, t: m([t]))(actual, jnp.asarray(tokens.numpy()))
    for a, b in zip(result, expected):
        np.testing.assert_allclose(a, b.numpy(), rtol=2e-4, atol=2e-5)


def test_iterative_tracker():
    reference_imports()
    from vggt.heads.track_modules.base_track_predictor import (
        BaseTrackerPredictor as TorchTracker,
    )
    from vggt.heads.track_modules.blocks import EfficientUpdateFormer

    c = config()
    torch.manual_seed(42)
    reference = TorchTracker(
        stride=2, corr_levels=2, corr_radius=1, latent_dim=16, hidden_size=32, depth=1
    )
    reference.updateformer = EfficientUpdateFormer(
        space_depth=1,
        time_depth=1,
        input_dim=52,
        hidden_size=32,
        num_heads=4,
        output_dim=18,
        num_virtual_tracks=4,
    )
    reference.eval()
    actual = BaseTrackerPredictor(c, rngs=nnx.Rngs(0))
    load_state_dict(actual, reference.state_dict())
    fmaps = torch.randn(2, 3, 16, 14, 21)
    query = torch.tensor([[[0.5, 1.5], [20.3, 12.2], [40.0, 25.0]]] * 2)
    with torch.no_grad():
        expected = reference(query, fmaps, iters=4)
    result = nnx.jit(lambda m, q, f: m(q, f, iters=4))(
        actual, jnp.asarray(query.numpy()), jnp.asarray(fmaps.numpy())
    )
    for a, b in zip(jax.tree.leaves(result), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b.numpy(), rtol=3e-4, atol=3e-4)
