# Copyright (c) Meta Platforms, Inc. and affiliates.
# JAX adaptation; see LICENSE and NOTICE.
import jax.numpy as jnp


def quat_to_mat(quaternions):
    """Quaternion order is XYZW, matching official VGGT."""
    i, j, k, r = jnp.moveaxis(quaternions, -1, 0)
    s = 2 / jnp.sum(quaternions * quaternions, axis=-1)
    values = (
        1 - s * (j * j + k * k),
        s * (i * j - k * r),
        s * (i * k + j * r),
        s * (i * j + k * r),
        1 - s * (i * i + k * k),
        s * (j * k - i * r),
        s * (i * k - j * r),
        s * (j * k + i * r),
        1 - s * (i * i + j * j),
    )
    return jnp.stack(values, -1).reshape(*quaternions.shape[:-1], 3, 3)


def pose_encoding_to_extri_intri(
    pose_encoding, image_size_hw, *, build_intrinsics=True
):
    """Return OpenCV camera-from-world extrinsics [..., 3, 4] and intrinsics."""
    extrinsics = jnp.concatenate(
        [quat_to_mat(pose_encoding[..., 3:7]), pose_encoding[..., :3, None]], -1
    )
    if not build_intrinsics:
        return extrinsics, None
    h, w = image_size_hw
    fy = (h / 2) / jnp.tan(pose_encoding[..., 7] / 2)
    fx = (w / 2) / jnp.tan(pose_encoding[..., 8] / 2)
    intrinsics = jnp.zeros((*pose_encoding.shape[:-1], 3, 3), jnp.float32)
    intrinsics = intrinsics.at[..., 0, 0].set(fx).at[..., 1, 1].set(fy)
    intrinsics = intrinsics.at[..., 0, 2].set(w / 2).at[..., 1, 2].set(h / 2)
    return extrinsics, intrinsics.at[..., 2, 2].set(1)


def unproject_depth_map_to_point_map(depth_map, extrinsics, intrinsics):
    """Convert depth [..., H, W] (or [..., H, W, 1]) to world points [..., H, W, 3]."""
    if depth_map.ndim == extrinsics.ndim + 1:
        depth_map = depth_map[..., 0]
    h, w = depth_map.shape[-2:]
    yy, xx = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    x = (xx - intrinsics[..., 0, 2, None, None]) / intrinsics[..., 0, 0, None, None]
    y = (yy - intrinsics[..., 1, 2, None, None]) / intrinsics[..., 1, 1, None, None]
    camera_points = jnp.stack([x * depth_map, y * depth_map, depth_map], -1)
    centered = camera_points - extrinsics[..., :3, 3][..., None, None, :]
    return jnp.einsum("...hwj,...ji->...hwi", centered, extrinsics[..., :3, :3])
