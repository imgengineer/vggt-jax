import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from vggt_jax import VGGT, VGGTConfig
from vggt_jax.weights import load_state_dict


@pytest.fixture
def tiny_config():
    return VGGTConfig(
        img_size=28,
        embed_dim=32,
        depth=4,
        num_heads=4,
        dino_depth=1,
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


def test_full_forward_jit_and_query_anchor(tiny_config):
    model = VGGT(tiny_config, rngs=nnx.Rngs(3))
    images = jax.random.uniform(jax.random.key(1), (2, 2, 3, 28, 42))
    queries = jnp.asarray([[[0.5, 1.5], [31.2, 20.7]]] * 2)
    reference = nnx.jit(lambda m, x, q: m(x, q))(model, images, queries)
    output = model.jit()(images, queries)
    for actual, expected in zip(jax.tree.leaves(output), jax.tree.leaves(reference)):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    assert output["depth"].shape == (2, 2, 28, 42, 1)
    assert output["world_points"].shape == (2, 2, 28, 42, 3)
    assert output["pose_enc"].shape == (2, 2, 9)
    assert output["track"].shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(output["track"][:, 0], queries)
    for leaf in jax.tree.leaves(output):
        assert np.isfinite(leaf).all()
    assert np.all(output["depth"] > 0)
    assert np.all(output["depth_conf"] >= 1)
    assert np.all((output["vis"] >= 0) & (output["vis"] <= 1))


@pytest.mark.parametrize("cache", ["jit_partial", "cached_partial"])
def test_cached_predictor_observes_parameter_updates(tiny_config, cache):
    model = VGGT(tiny_config, enable_point=False, enable_track=False)
    images = jnp.full((1, 3, 28, 28), 0.3)
    if cache == "cached_partial":
        forward = nnx.jit(lambda m, x: m(x), graph=True, graph_updates=True)
        predict = nnx.cached_partial(forward, model, graph=True, graph_updates=True)
    else:
        predict = model.jit()
    before = predict(images)["depth"]
    bias = model.depth_head.scratch.output_conv2["2"].bias
    bias[...] = bias[...] + jnp.array([0.5, 0.0])
    after = predict(images)["depth"]
    np.testing.assert_allclose(after, before * np.exp(0.5), rtol=1e-5)


def test_nnx_parameter_gradients(tiny_config):
    model = VGGT(tiny_config, enable_point=False, enable_track=False)
    images = jnp.ones((1, 3, 28, 28), jnp.float32) * 0.3

    def loss(m):
        predictions = m(images)
        return jnp.mean(predictions["depth"]) + jnp.mean(predictions["pose_enc"] ** 2)

    grads = nnx.jit(nnx.grad(loss))(model)
    leaves = jax.tree.leaves(grads)
    assert all(np.isfinite(leaf).all() for leaf in leaves)
    assert float(jnp.linalg.norm(grads.depth_head.projects[0].weight[...])) > 0
    assert (
        float(
            jnp.linalg.norm(grads.aggregator.patch_embed.patch_embed.proj.weight[...])
        )
        > 0
    )


def test_strict_checkpoint_errors(tiny_config):
    model = nnx.eval_shape(lambda: VGGT(tiny_config, enable_track=False))
    with pytest.raises(ValueError, match="missing="):
        load_state_dict(model, {})
    shape = model.aggregator.camera_token.shape
    with pytest.raises(ValueError, match="checkpoint"):
        load_state_dict(
            model, {"aggregator.camera_token": np.zeros((*shape[:-1], 1))}, strict=False
        )


def test_input_validation(tiny_config):
    model = VGGT(tiny_config, enable_track=False)
    with pytest.raises(ValueError, match="divisible"):
        model(jnp.zeros((1, 3, 29, 28)))
    with pytest.raises(ValueError, match="matching batch"):
        model(jnp.zeros((1, 3, 28, 28)), jnp.zeros((2, 1, 2)))
