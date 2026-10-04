"""Text-guided visual token selection."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PruningConfig:
    layers: tuple[int, ...] = (0, 9, 19)
    keep_ratios: tuple[float, ...] = (0.17, 0.6, 0.3)

    def __post_init__(self):
        if len(self.layers) != len(self.keep_ratios):
            raise ValueError("Each pruning layer needs a keep ratio.")
        if any(layer < 0 for layer in self.layers):
            raise ValueError("Pruning layers must be nonnegative.")
        if tuple(sorted(set(self.layers))) != self.layers:
            raise ValueError("Pruning layers must be distinct and increasing.")
        if any(not 0 < ratio <= 1 for ratio in self.keep_ratios):
            raise ValueError("Keep ratios must be in (0, 1].")

    def validate_layers(self, num_layers):
        if self.layers and self.layers[-1] >= num_layers:
            raise ValueError(f"Pruning layers must be below {num_layers}.")


def select_visual_tokens(attention, visual_start, visual_count, keep_ratio):
    """Weight text-to-vision attention by symmetric text-to-text importance."""
    text_start = visual_start + visual_count
    if text_start >= attention.shape[-2]:
        raise ValueError("Text guidance must follow the visual tokens.")
    text_attention = attention[:, :, text_start:, text_start:]
    symmetric_attention = (
        text_attention + text_attention.transpose(-1, -2)
        - torch.diag_embed(torch.diagonal(text_attention, dim1=-2, dim2=-1))
    )
    text_importance = symmetric_attention.sum(dim=-2)
    text_to_vision = attention[:, :, text_start:, visual_start:text_start]
    scores = (text_importance.unsqueeze(-2) @ text_to_vision).squeeze(-2).sum(dim=1)
    keep_count = max(1, min(int(keep_ratio * visual_count), visual_count))
    indices = scores.topk(keep_count, dim=-1).indices
    keep = torch.zeros_like(scores, dtype=torch.bool)
    return keep.scatter_(1, indices, True)
