import torch
from torch import nn

from alibi.config import ALiBiConfig
from alibi.layers import ALiBiTransformerLayer


class ALiBiTransformer(nn.Module):
    def __init__(self, config: ALiBiConfig, device) -> None:
        super().__init__()
        self.max_len = config.max_len
        # BUG FIX: Changed from nn.Sequential to nn.ModuleList.
        # nn.Sequential only passes a single tensor between layers —
        # it cannot forward the ``padding_mask`` argument.  With
        # ModuleList we iterate explicitly and pass both arguments.
        self.layers = nn.ModuleList(
            [ALiBiTransformerLayer(config, device) for _ in range(config.num_layers)]
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        _, seq_len, _ = x.shape
        assert seq_len <= self.max_len, "sequence length exceeds `max_len`"
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)
        return x
