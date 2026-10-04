# Copyright (c) Meta Platforms, Inc. and affiliates.
# JAX/NumPy adaptation of VGGT image preprocessing; see LICENSE and NOTICE.
from dataclasses import dataclass

import grain
import numpy as np
from PIL import Image


def load_and_preprocess_images(
    image_path_list, mode="crop", target_size=518, patch_size=14
):
    """Return float32 [S, 3, H, W] in [0, 1], matching the official PIL pipeline."""
    if not image_path_list:
        raise ValueError("At least one image is required")
    if mode not in ("crop", "pad"):
        raise ValueError("mode must be 'crop' or 'pad'")
    if target_size < patch_size or target_size % patch_size:
        raise ValueError("target_size must be a positive multiple of patch_size")
    images = []
    for path in image_path_list:
        with Image.open(path) as source:
            if source.mode == "RGBA":
                source = Image.alpha_composite(
                    Image.new("RGBA", source.size, (255, 255, 255, 255)), source
                )
            image = source.convert("RGB")
        width, height = image.size
        if mode == "pad" and height > width:
            nh = target_size
            nw = max(
                patch_size,
                round(width * target_size / height / patch_size) * patch_size,
            )
        else:
            nw = target_size
            nh = max(
                patch_size,
                round(height * target_size / width / patch_size) * patch_size,
            )
        image = image.resize((nw, nh), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
        if mode == "crop" and nh > target_size:
            top = (nh - target_size) // 2
            array = array[:, top : top + target_size]
        if mode == "pad":
            array = _pad(array, target_size, target_size)
        images.append(array)
    height = max(image.shape[1] for image in images)
    width = max(image.shape[2] for image in images)
    return np.stack([_pad(image, height, width) for image in images])


def _pad(image, height, width):
    dh, dw = height - image.shape[1], width - image.shape[2]
    return np.pad(
        image,
        ((0, 0), (dh // 2, dh - dh // 2), (dw // 2, dw - dw // 2)),
        constant_values=1,
    )


@dataclass(frozen=True)
class PreprocessScene:
    mode: str = "pad"
    target_size: int = 518
    patch_size: int = 14

    def __call__(self, paths):
        return {
            "images": load_and_preprocess_images(
                paths, self.mode, self.target_size, self.patch_size
            )
        }


def scene_dataset(
    scenes,
    *,
    batch_size=1,
    shuffle=False,
    seed=0,
    target_size=518,
    mode="pad",
    patch_size=14,
    num_threads=4,
):
    """Grain pipeline over lists of image paths, yielding [B, S, 3, H, W].

    Each batch must contain the same number of views; pad mode gives fixed H/W.
    The returned iterator supports Grain's get_state()/set_state() for resuming.
    """
    dataset = grain.MapDataset.source(scenes)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    dataset = dataset.map(PreprocessScene(mode, target_size, patch_size)).batch(
        batch_size
    )
    return dataset.to_iter_dataset(
        read_options=grain.ReadOptions(num_threads=num_threads, prefetch_buffer_size=16)
    )
