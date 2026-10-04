import jax.numpy as jnp
import numpy as np
from PIL import Image

from vggt_jax.data import load_and_preprocess_images, scene_dataset
from vggt_jax.geometry import (
    pose_encoding_to_extri_intri,
    unproject_depth_map_to_point_map,
)


def test_rgba_padding_and_grain_resume(tmp_path):
    paths = []
    for i in range(3):
        path = tmp_path / f"{i}.png"
        Image.new("RGBA", (30, 20), (i * 50, 0, 0, 128)).save(path)
        paths.append(str(path))
    images = load_and_preprocess_images(paths, "pad", target_size=28)
    assert images.shape == (3, 3, 28, 28)
    assert images.dtype == np.float32
    assert np.all(images[:, :, 0] == 1)
    np.testing.assert_allclose(images[0, :, 14, 14], [127 / 255] * 3)
    dataset = scene_dataset(
        [[p] for p in paths], shuffle=True, seed=7, target_size=28, num_threads=0
    )
    iterator = iter(dataset)
    next(iterator)
    state = iterator.get_state()
    expected = next(iterator)
    restored = iter(dataset)
    restored.set_state(state)
    np.testing.assert_array_equal(next(restored)["images"], expected["images"])


def test_camera_unprojection_known_geometry():
    pose = jnp.asarray([[[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, np.pi / 2, np.pi / 2]]])
    extrinsics, intrinsics = pose_encoding_to_extri_intri(pose, (4, 6))
    np.testing.assert_allclose(
        intrinsics[0, 0], [[3, 0, 3], [0, 2, 2], [0, 0, 1]], atol=1e-6
    )
    depth = jnp.ones((1, 1, 4, 6, 1)) * 2
    points = unproject_depth_map_to_point_map(depth, extrinsics, intrinsics)
    np.testing.assert_allclose(points[0, 0, 2, 3], [-1, -2, -1], atol=1e-6)
    np.testing.assert_allclose(points[0, 0, 0, 0], [-3, -4, -1], atol=1e-6)


def test_preprocessing_matches_official(tmp_path):
    import sys
    from pathlib import Path

    import pytest

    pytest.importorskip("torch")
    reference = Path(__file__).resolve().parents[1] / ".reference/vggt"
    if not reference.exists():
        pytest.skip("Official reference has not been fetched")
    sys.path.insert(0, str(reference))
    from vggt.utils.load_fn import load_and_preprocess_images as torch_preprocess

    paths = []
    for i, size in enumerate(((40, 20), (20, 40), (40, 30))):
        path = tmp_path / f"{i}.png"
        rng = np.random.default_rng(i)
        Image.fromarray(rng.integers(0, 256, (size[1], size[0], 4), np.uint8)).save(
            path
        )
        paths.append(str(path))
    for mode in ("crop", "pad"):
        np.testing.assert_array_equal(
            load_and_preprocess_images(paths, mode),
            torch_preprocess(paths, mode).numpy(),
        )
