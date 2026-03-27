"""Tests for the ALiBi attention and transformer modules."""
import torch
import pytest

from alibi.config import ALiBiConfig
from alibi.attention import (
    get_relative_positions,
    get_alibi_slope,
    ALiBiMultiHeadAttention,
)
from alibi.layers import ALiBiTransformerLayer, FeedForward
from alibi.model import ALiBiTransformer


class TestRelativePositions:
    def test_shape(self):
        pos = get_relative_positions(5, device="cpu")
        assert pos.shape == (5, 5)

    def test_diagonal_is_zero(self):
        pos = get_relative_positions(4, device="cpu")
        assert torch.equal(pos.diag(), torch.zeros(4))

    def test_antisymmetric(self):
        pos = get_relative_positions(6, device="cpu")
        assert torch.equal(pos, -pos.T)


class TestALiBiSlope:
    def test_shape(self):
        slopes = get_alibi_slope(4, device="cpu")
        assert slopes.shape == (4, 1, 1)

    def test_decreasing(self):
        slopes = get_alibi_slope(8, device="cpu").squeeze()
        for i in range(len(slopes) - 1):
            assert slopes[i] > slopes[i + 1]


class TestALiBiMultiHeadAttention:
    @pytest.fixture
    def config(self):
        return ALiBiConfig(
            num_layers=1, d_model=32, num_heads=2,
            max_len=20, dropout=0.0, causal=False,
        )

    def test_output_shape(self, config):
        attn = ALiBiMultiHeadAttention(config, device="cpu")
        x = torch.randn(4, 10, 32)
        out = attn(x)
        assert out.shape == (4, 10, 32)

    def test_with_padding_mask(self, config):
        attn = ALiBiMultiHeadAttention(config, device="cpu")
        x = torch.randn(2, 8, 32)
        mask = torch.tensor([
            [False, False, False, True, True, True, True, True],
            [False, False, False, False, False, True, True, True],
        ])
        out = attn(x, padding_mask=mask)
        assert out.shape == (2, 8, 32)

    def test_causal_mode(self):
        config = ALiBiConfig(
            num_layers=1, d_model=32, num_heads=2,
            max_len=20, dropout=0.0, causal=True,
        )
        attn = ALiBiMultiHeadAttention(config, device="cpu")
        x = torch.randn(2, 10, 32)
        out = attn(x)
        assert out.shape == (2, 10, 32)


class TestALiBiTransformerLayer:
    def test_output_shape(self):
        config = ALiBiConfig(
            num_layers=1, d_model=32, num_heads=2,
            max_len=20, dropout=0.0, causal=False,
        )
        layer = ALiBiTransformerLayer(config, device="cpu")
        x = torch.randn(4, 10, 32)
        out = layer(x)
        assert out.shape == (4, 10, 32)

    def test_with_padding_mask(self):
        config = ALiBiConfig(
            num_layers=1, d_model=32, num_heads=2,
            max_len=20, dropout=0.0, causal=False,
        )
        layer = ALiBiTransformerLayer(config, device="cpu")
        x = torch.randn(2, 8, 32)
        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[0, 5:] = True
        out = layer(x, padding_mask=mask)
        assert out.shape == (2, 8, 32)


class TestALiBiTransformer:
    def test_output_shape(self):
        config = ALiBiConfig(
            num_layers=2, d_model=32, num_heads=2,
            max_len=20, dropout=0.0, causal=False,
        )
        model = ALiBiTransformer(config, device="cpu")
        x = torch.randn(4, 10, 32)
        out = model(x)
        assert out.shape == (4, 10, 32)

    def test_seq_len_exceeding_max_raises(self):
        config = ALiBiConfig(
            num_layers=1, d_model=32, num_heads=2,
            max_len=5, dropout=0.0, causal=False,
        )
        model = ALiBiTransformer(config, device="cpu")
        with pytest.raises(AssertionError):
            model(torch.randn(1, 10, 32))

    def test_with_padding_mask(self):
        config = ALiBiConfig(
            num_layers=2, d_model=32, num_heads=2,
            max_len=20, dropout=0.0, causal=False,
        )
        model = ALiBiTransformer(config, device="cpu")
        x = torch.randn(2, 8, 32)
        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[0, 5:] = True
        out = model(x, padding_mask=mask)
        assert out.shape == (2, 8, 32)

    def test_gradient_flows(self):
        config = ALiBiConfig(
            num_layers=2, d_model=32, num_heads=2,
            max_len=20, dropout=0.0, causal=False,
        )
        model = ALiBiTransformer(config, device="cpu")
        x = torch.randn(2, 5, 32, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None


class TestFeedForward:
    def test_output_shape(self):
        config = ALiBiConfig(d_model=32, dropout=0.0, expansion_factor=2)
        ff = FeedForward(config)
        x = torch.randn(4, 10, 32)
        out = ff(x)
        assert out.shape == (4, 10, 32)
