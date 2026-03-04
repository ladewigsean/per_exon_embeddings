import math

import torch
from torch import nn
from torch.nn import functional as F

from alibi.config import ALiBiConfig


def get_relative_positions(seq_len: int, device) -> torch.Tensor:
    x = torch.arange(seq_len, device=device)[None, :]
    y = torch.arange(seq_len, device=device)[:, None]
    return x - y


def get_alibi_slope(num_heads, device) -> torch.Tensor:
    x = (2 ** 8) ** (1 / num_heads)
    return (
        torch.tensor([1 / x ** (i + 1) for i in range(num_heads)], device=device)
        .unsqueeze(-1)
        .unsqueeze(-1)
    )


class ALiBiMultiHeadAttention(nn.Module):
    def __init__(self, config: ALiBiConfig, device) -> None:
        super().__init__()
        self.device = device
        self.causal = config.causal
        self.num_heads = config.num_heads
        self.scale = math.sqrt(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("m", get_alibi_slope(self.num_heads, self.device))
        self.kqv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        if config.causal:
            self.register_buffer(
                "mask", torch.tril(torch.ones(1, 1, config.max_len, config.max_len))
            )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            padding_mask: bool [batch_size, seq_len], True at padded positions (optional)
        """
        batch_size, seq_len, _ = x.shape

        key, query, value = self.kqv(x).chunk(3, dim=-1)
        key = key.view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        query = query.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        bias = (self.m * get_relative_positions(seq_len, device=x.device)).unsqueeze(0)

        score = torch.matmul(query, key) / self.scale + bias

        if self.causal:
            score = score.masked_fill(
                self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf")
            )

        # Apply padding mask: mask out attention to/from padded positions
        if padding_mask is not None:
            # padding_mask: [B, S] bool, True at pad -> expand to [B, 1, 1, S]
            score = score.masked_fill(
                padding_mask[:, None, None, :], float("-inf")
            )

        attn = F.softmax(score, dim=-1)
        out = torch.matmul(attn, value)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.dropout(out)

        return out
