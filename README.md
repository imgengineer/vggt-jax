<div align="center">
<h1>VGGT-JAX</h1>
<p>Visual Geometry Grounded Transformer in JAX and Flax NNX</p>

[![Paper](https://img.shields.io/badge/arXiv-2503.11651-b31b1b)](https://arxiv.org/abs/2503.11651)
[![Upstream](https://img.shields.io/badge/Upstream-VGGT-blue)](https://github.com/facebookresearch/vggt)
[![Weights](https://img.shields.io/badge/Hugging_Face-VGGT--1B-yellow)](https://huggingface.co/facebook/VGGT-1B)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)

**An unofficial JAX implementation of [VGGT](https://github.com/facebookresearch/vggt).**
</div>

## Overview

VGGT-JAX predicts camera parameters, depth maps, world-space point maps and point tracks from RGB images. It implements the original VGGT-1B architecture in Flax NNX, loads the official weights directly, and uses Tokamax for attention and Grain for scene input pipelines.

- **Official checkpoint compatibility:** all 1,797 parameter keys and 1,256,537,516 parameters match the reference model. Load Hugging Face weights, local safetensors, or a PyTorch state dict.
- **Complete prediction heads:** DINOv2 encoder, alternating frame/global attention, iterative camera prediction, depth/point DPT heads, and the iterative tracker.
- **Tokamax high-performance attention:** GPU inference explicitly uses the fused Pallas Triton FlashAttention kernel; CPU uses Tokamax's XLA implementation for tests.
- **JAX O1 by default:** reusable compiled inference through `model.jit()`, with model parameters kept as `nnx.Param`.
- **Measured parity and performance:** the tested FP32 runs meet the documented tolerances. On an RTX 5090, inference takes 140 ms for geometry and 153 ms with five tracked points, versus 156 ms and 395 ms for the official PyTorch eager baseline.

This implementation targets upstream commit [`a288dd0`](https://github.com/facebookresearch/vggt/tree/a288dd0f14786c93483e45524328726ab7b1b4ce). The timings below apply to that reference, the tested inputs, and the recorded environment.

## Quick Start

Clone the repository and restore the environment with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/imgengineer/vggt-jax.git
cd vggt-jax
uv sync --locked
uv run python -c 'import jax; print(jax.devices())'
```

The environment uses Python 3.12+ and JAX CUDA 13 wheels. The tested versions in `uv.lock` are JAX 0.11.2, Flax 0.12.10, Grain 0.2.18, and Tokamax 0.0.15. GPU inference requires a compatible NVIDIA driver; unit tests run on CPU.

For the tested 32 GB GPU configuration, set these variables before starting Python. They prevent JAX preallocation and CUDA command-buffer memory pressure. The CLI and validation scripts set these defaults automatically.

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.60
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
```

Load the official checkpoint and predict geometry:

```python
import jax
import jax.numpy as jnp

from vggt_jax import VGGT
from vggt_jax.data import load_and_preprocess_images

# Downloads the official weights on the first run.
model = VGGT.from_pretrained("facebook/VGGT-1B")
image_names = ["path/to/imageA.png", "path/to/imageB.png"]
images = jnp.asarray(load_and_preprocess_images(image_names, mode="pad"))

# Create once and reuse. The first call includes compilation.
predict = model.jit()
predictions = predict(images)
jax.block_until_ready(predictions)

print(predictions["depth"].shape)
print(predictions["world_points"].shape)
```

PyTorch is optional for safetensors loading and JAX inference. Install `uv sync --locked --extra validation` to load `.pt` checkpoints or compare against the original implementation. PyTorch and torchvision use the CUDA 13 index specified in `pyproject.toml`.

<details>
<summary>Environment creation with uv add</summary>

The dependencies were added with:

```bash
uv add 'jax[cuda13]' flax grain numpy pillow huggingface-hub safetensors tokamax triton
uv add --optional validation torch torchvision einops
uv add --dev pytest ruff
```

Use `uv sync --locked` to reproduce the versions used in the reports.

</details>

## Detailed Usage

### Cameras, depth and point maps

Convert the predicted pose encoding to camera matrices, or reconstruct world points from depth:

```python
from vggt_jax.geometry import (
    pose_encoding_to_extri_intri,
    unproject_depth_map_to_point_map,
)

# OpenCV convention: extrinsic maps world coordinates to camera coordinates.
extrinsic, intrinsic = pose_encoding_to_extri_intri(
    predictions["pose_enc"], images.shape[-2:]
)
points_from_depth = unproject_depth_map_to_point_map(
    predictions["depth"], extrinsic, intrinsic
)
```

Inputs use `[S, 3, H, W]` or `[B, S, 3, H, W]`, with RGB values in `[0, 1]`. Image dimensions must be multiples of 14. Preprocessing follows the official crop/pad pipeline; `mode="pad"` produces 518 × 518 images.

| Output | Shape | Meaning |
|---|---|---|
| `pose_enc` | `[B, S, 9]` | Final camera pose encoding |
| `pose_enc_list` | Four tensors of `[B, S, 9]` | Camera refinement iterations |
| `depth` | `[B, S, H, W, 1]` | Depth map |
| `depth_conf` | `[B, S, H, W]` | Depth confidence |
| `world_points` | `[B, S, H, W, 3]` | World-space point map |
| `world_points_conf` | `[B, S, H, W]` | Point-map confidence |
| `images` | `[B, S, 3, H, W]` | Input images with a batch dimension |
| `track` | `[B, S, N, 2]` | Tracked pixel coordinates, when queries are provided |
| `vis`, `conf` | `[B, S, N]` | Tracking visibility and confidence |

### Point tracking

Query coordinates refer to the **preprocessed first image**, in `(x, y)` pixel order:

```python
query_points = jnp.array([[100.0, 200.0], [60.72, 259.94]])
predictions_with_tracks = predict(images, query_points)
tracks = predictions_with_tracks["track"]
```

Queries may have shape `[N, 2]` for a single scene or `[B, N, 2]`. The first frame's track coordinates equal the queries. The default tracker's seven correlation levels require both image dimensions to be at least 128 pixels; the 518 × 518 pad mode satisfies this.

### Command line

Save predictions and camera matrices to a compressed NumPy archive:

```bash
uv run vggt-jax imageA.png imageB.png --mode pad --output predictions.npz

# Local official checkpoint, with queries saved as a [N, 2] .npy array.
uv run vggt-jax imageA.png imageB.png \
    --checkpoint .cache/model.safetensors --query-points queries.npy \
    --mode pad --output tracked.npz
```

The archive contains the final prediction tensors plus `extrinsic` and `intrinsic`. With no queries, the CLI omits the tracker parameters when loading the checkpoint.

### Checkpoint loading

```python
# Official parameter names and layouts are preserved.
model = VGGT.from_pretrained("/path/to/model.safetensors")
# Requires the validation extra:
model = VGGT.from_pretrained("/path/to/model.pt")
```

To load a state dict from an existing PyTorch model:

```python
from flax import nnx

model = nnx.eval_shape(VGGT)
report = model.load_state_dict(torch_model.state_dict())
print(report)
```

Loading is strict by default: missing keys, unexpected keys and shape mismatches are rejected. Explicitly disabled heads are skipped. Model construction uses `nnx.eval_shape` so loading does not first allocate random weights for the 1B model.

The tested checkpoint SHA-256 is `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e`.

### Grain scene pipeline

Each sample is a list of image paths for one scene. Batches must contain the same number of views; pad mode gives a fixed spatial shape.

```python
from vggt_jax.data import scene_dataset

dataset = scene_dataset(
    [["scene1/a.png", "scene1/b.png"], ["scene2/a.png", "scene2/b.png"]],
    batch_size=2,
    shuffle=True,
    seed=42,
)
iterator = iter(dataset)
batch = next(iterator)  # batch["images"]: [2, 2, 3, 518, 518]
state = iterator.get_state()
iterator.set_state(state)
```

The pipeline provides deterministic shuffling, threaded preprocessing and iterator checkpointing. Model parameters support NNX differentiation and transformations. Tests check finite, nonzero gradients in the backbone and prediction heads. A complete training program, training losses, COLMAP export and the upstream visualization demos are not included.

## Accuracy and Performance

### Official PyTorch comparison

The benchmark uses an RTX 5090, the same official checkpoint, and two kitchen images with input shape `[1, 2, 3, 350, 518]`. Each backend runs in a separate process, with three warmup calls followed by 20 GPU-synchronized calls. Timings are for both images together and exclude loading, compilation, preprocessing, transfers and file output.

The reference is the official PyTorch eager `model.forward` with `torch.no_grad()` and SDPA. FP32 runs disable PyTorch TF32. JAX uses `model.jit()`, default O1, Tokamax attention and disabled CUDA command buffers. BF16 follows the upstream backbone autocast configuration; camera/depth/point heads stay FP32. `torch.compile` has not been benchmarked.

| Mode | Official PyTorch | JAX / Tokamax O1 | Speedup | Parity |
|---|---:|---:|---:|---|
| FP32 cameras, depth and points | 156.25 ms | **139.70 ms** | **1.12×** | Pass |
| FP32 geometry + five tracked points | 395.12 ms | **153.05 ms** | **2.58×** | Pass |
| BF16 cameras, depth and points | 82.41 ms | **60.76 ms** | **1.36×** | Fail |

**FP32 is the default for numerical parity.** BF16 is faster but fails the selected geometry tolerances. The first JAX compilations took approximately 34 s, 124 s and 31 s respectively; these times are excluded from the table.

After switching from automatic dispatch to explicit `implementation="triton"`, a fresh five-repeat run measured 139.92 ms for geometry and 152.24 ms with tracking; parity passed in both cases. The full run is recorded in [the explicit-Triton report](reports/speed_explicit_triton.json).

Versions, settings, all 20 latency samples and output errors are available in [the full benchmark report](reports/speed_benchmark.json). [The earlier baseline](reports/speed_baseline.json) records JAX timings of 401 ms for FP32 geometry and 416 ms with tracking, before the attention and NNX cache optimizations.

### Numerical parity

Acceptance criteria are geometry/intermediate-feature **NRMSE ≤ 1e-3**, tracking **RMSE ≤ 1 pixel**, and visibility/confidence **MAE ≤ 0.05**. NRMSE is `sqrt(mean((jax - torch)^2)) / sqrt(mean(torch^2))`, with a small denominator floor for zero-valued tensors.

The FP32 geometry-plus-tracking benchmark above gives:

| Output | Metric | Measured error |
|---|---|---:|
| Camera pose | NRMSE | 4.24e-7 |
| Depth | NRMSE | 1.55e-6 |
| World points | NRMSE | 4.70e-6 |
| Tracks | RMSE | 0.338 pixels |
| Visibility | MAE | 0.01696 |
| Tracking confidence | MAE | 0.00111 |

The optimized implementation also passes [flower two-view validation](reports/parity_flower_optimized.json), [single-view validation](reports/parity_single_optimized.json), and [end-to-end CLI validation](reports/cli_optimized.json). Across the three FP32 input sets, the largest geometry NRMSE is approximately `1.4e-5`. These are measured examples rather than guarantees for every scene.

The tracker amplifies small floating-point differences through iterative high-frequency displacement encoding. Tracking therefore uses pixel RMSE and score MAE for acceptance; individual errors may exceed those averages. Stricter validation is available with `--strict-tracking --track-rmse-limit 0.05 --score-mae-limit 0.001`; the current kitchen two-view results fail those stricter limits.

<details>
<summary>Attention precision and NNX caching</summary>

Attention calls `tokamax.dot_product_attention(..., implementation="triton")` on GPU and `implementation="xla"` on CPU. This explicitly selects Tokamax's fused Pallas Triton FlashAttention kernel for GPU inference instead of relying on automatic dispatch. FP32 attention on NVIDIA SM80+ uses `TF32_TF32_F32_X3`, combining three Tensor Core products with FP32 accumulation. Tracker attention, other devices, BF16 attention, linear layers and convolutions use `HIGHEST`. BF16 mode changes backbone linear/convolution computations; prediction heads remain FP32.

Importing `vggt_jax` sets [JAX's optimization level](https://docs.jax.dev/en/latest/config_options.html#optimization-level) to O1 if no explicit configuration exists. Set `JAX_OPTIMIZATION_LEVEL=O2` before running, or call `jax.config.update` to override it.

Reuse `model.jit()` to cache the flattened parameter structure with `nnx.jit_partial(graph=False)`. Parameter value updates remain visible. Create a new predictor after changing the model structure; new image shapes or query counts cause compilation for those shapes.

You can also use [`nnx.cached_partial`](https://flax.readthedocs.io/en/latest/guides/performance.html#caching-graph-node-traversals) with `nnx.jit`:

```python
from flax import nnx


@nnx.jit(graph=True, graph_updates=True)
def forward(model, images, query_points=None):
    return model(images, query_points)


predict = nnx.cached_partial(forward, model, graph=True, graph_updates=True)
predictions = predict(images, query_points)
```

This caches graph traversals in the current process and shares the original parameter variables. It requires graph mode and graph updates, preserves parameter value updates, and still needs the first XLA compilation.

On the same FP32 geometry input, 20 alternating measurements produced identical predictions:

| NNX call path | Median inference |
|---|---:|
| `nnx.jit` | 179.44 ms |
| `nnx.jit` + `nnx.cached_partial` | 156.84 ms |
| `nnx.jit_partial(graph=False)` — default | **139.84 ms** |

Both cache variants pass the parameter-update test. See [the cache comparison](reports/nnx_cache_comparison.json), [GPU profiling](reports/performance_profile.json), and [the attention microbenchmark](reports/attention_benchmark.json). The attention microbenchmark is a separate kernel comparison and does not establish whole-model speedup.

</details>

## Reproducing the Results

For a fresh clone, fetch the pinned reference to enable all PyTorch parity tests, then run the suite and source checks:

```bash
git clone https://github.com/facebookresearch/vggt.git .reference/vggt
git -C .reference/vggt checkout a288dd0f14786c93483e45524328726ab7b1b4ce
uv sync --locked --extra validation
uv run --extra validation pytest -q
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
```

The validated suite contains 26 tests, including CPU PyTorch parity, input preprocessing, geometric operations, checkpoint checks, gradients and cached parameter updates. Reference-dependent tests are skipped if the checkout is absent. GPU attention forward/backward checks are included in the attention benchmark.

Download the official checkpoint to the path used by the benchmarks:

```bash
uv run python - <<'PY'
from huggingface_hub import hf_hub_download

hf_hub_download("facebook/VGGT-1B", "model.safetensors", local_dir=".cache")
PY
```

Compare independent PyTorch/JAX runs, then benchmark using the saved inputs:

```bash
uv run --extra validation python scripts/validate_parity.py \
    --checkpoint .cache/model.safetensors --report reports/parity_reproduced.json
uv run --extra validation python scripts/benchmark_vggt.py \
    --work-dir .cache/speed_reproduced --report reports/speed_reproduced.json
uv run python scripts/benchmark_nnx_cache.py
uv run python scripts/benchmark_attention.py
```

Validation downloads the reference source at the pinned commit and uses two real kitchen images with five subpixel/boundary queries. It compares every prediction, all four camera refinements, and intermediate features at layers 4, 11, 17 and 23. Reports record input hashes, shapes, dependency versions, RMSE, NRMSE, MAE and maximum absolute error.

Use `--images`, `--frames`, `--batch-size`, or `--mode pad` for other validation inputs. The speed benchmark supports `--case float32`, `--case float32-tracking`, and `--case bfloat16`. The parity validator exits with a nonzero status when tolerances fail; the speed benchmark records the parity result alongside its timings.

Weights, reference checkouts, environments and prediction arrays are excluded from Git; only source, tests, the lockfile and JSON reports are published.

## Acknowledgements and Citation

This port builds on [VGGT](https://github.com/facebookresearch/vggt) and its original authors' work. Its implementation also uses [JAX](https://github.com/jax-ml/jax), [Flax NNX](https://github.com/google/flax), [Tokamax](https://github.com/openxla/tokamax), and [Grain](https://github.com/google/grain).

If you use the model in research, cite the [original VGGT paper](https://arxiv.org/abs/2503.11651):

```bibtex
@inproceedings{wang2025vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## License

The upstream VGGT license is retained in [LICENSE](LICENSE). Derived DINOv2 components retain their Apache-2.0 notices and [license text](LICENSE-APACHE-2.0); see [NOTICE](NOTICE) for attribution.

Checkpoint terms are separate from the code license. The original [VGGT-1B](https://huggingface.co/facebook/VGGT-1B) checkpoint is non-commercial; the gated [VGGT-1B-Commercial](https://huggingface.co/facebook/VGGT-1B-Commercial) checkpoint has its own terms, as described in the [upstream license section](https://github.com/facebookresearch/vggt#license). All published parity and speed results here use the original VGGT-1B checkpoint.
