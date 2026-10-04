from dataclasses import dataclass


@dataclass(frozen=True)
class VGGTConfig:
    """Official VGGT-1B defaults; smaller dimensions also support unit tests."""

    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    dino_depth: int = 24
    num_register_tokens: int = 4
    camera_depth: int = 4
    intermediate_layer_idx: tuple[int, ...] = (4, 11, 17, 23)
    dpt_features: int = 256
    dpt_channels: tuple[int, ...] = (256, 512, 1024, 1024)
    track_features: int = 128
    track_hidden: int = 384
    track_depth: int = 6
    track_heads: int = 8
    track_virtual: int = 64
    track_corr_levels: int = 7
    track_corr_radius: int = 4

    def __post_init__(self):
        if self.embed_dim % self.num_heads or (self.embed_dim // self.num_heads) % 4:
            raise ValueError("embed_dim / num_heads must be divisible by 4 for RoPE")
        if len(self.intermediate_layer_idx) != 4 or len(self.dpt_channels) != 4:
            raise ValueError("DPT requires four intermediate layers and channel sizes")
        if any(i < 0 or i >= self.depth for i in self.intermediate_layer_idx):
            raise ValueError("intermediate_layer_idx must index the aggregator blocks")
