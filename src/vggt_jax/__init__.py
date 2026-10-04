"""VGGT implemented with JAX and Flax NNX."""

import os

from jax import config as jax_config

from .models import VGGT, VGGTConfig

if (
    "JAX_OPTIMIZATION_LEVEL" not in os.environ
    and jax_config.jax_optimization_level == "UNKNOWN"
):
    jax_config.update("jax_optimization_level", "O1")

__all__ = ["VGGT", "VGGTConfig"]
