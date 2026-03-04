"""Tests for positional encoding modules."""
import torch
import pytest

from classifier_runner import PositionalEncoding, LearnedPositionalEmbedding


class TestPositionalEncoding:
    def test_output_shape(self, batch_size, seq_len, embed_dim):
        pe = PositionalEncoding(d_model=embed_dim, max_length=100, dropout=0.0)
        x = torch.randn(batch_size, seq_len, embed_dim)
        out = pe(x)
        assert out.shape == (batch_size, seq_len, embed_dim)

    def test_encoding_is_float32(self, embed_dim):
        pe = PositionalEncoding(d_model=embed_dim, max_length=100, dropout=0.0)
        assert pe.pe.dtype == torch.float32

    def test_different_positions_produce_different_encodings(self, embed_dim):
        pe = PositionalEncoding(d_model=embed_dim, max_length=100, dropout=0.0, factor=1.0)
        # The raw PE values at position 0 and position 1 must differ
        assert not torch.allclose(pe.pe[0, 0, :], pe.pe[0, 1, :])

    def test_factor_scales_contribution(self, embed_dim):
        pe_small = PositionalEncoding(d_model=embed_dim, max_length=100, dropout=0.0, factor=0.1)
        pe_large = PositionalEncoding(d_model=embed_dim, max_length=100, dropout=0.0, factor=1.0)
        x = torch.zeros(1, 5, embed_dim)
        out_small = pe_small(x)
        out_large = pe_large(x)
        # The large factor output should be ~10x the small factor output
        ratio = out_large.abs().mean() / out_small.abs().mean()
        assert 9.0 < ratio < 11.0

    def test_pe_actually_modifies_input(self, embed_dim):
        """Verify PE adds a non-zero signal to the input — the core bug."""
        pe = PositionalEncoding(d_model=embed_dim, max_length=100, dropout=0.0, factor=1.0)
        x = torch.randn(2, 10, embed_dim)
        out = pe(x)
        # The output should differ from the input
        assert not torch.allclose(x, out)
        # The difference should be the PE values (scaled by factor)
        diff = out - x
        expected = pe.pe[:, :10, :] * pe.factor
        assert torch.allclose(diff, expected.expand_as(diff), atol=1e-6)

    def test_pe_survives_float32_addition_with_large_values(self, embed_dim):
        """The original bug: float16 PE gets lost when added to large embeddings."""
        pe = PositionalEncoding(d_model=embed_dim, max_length=100, dropout=0.0, factor=0.01)
        # Simulate large pre-trained embeddings (magnitude ~5-10)
        x = torch.randn(2, 10, embed_dim) * 5.0
        out = pe(x)
        diff = (out - x).abs().max().item()
        # In float32, even small PE values should be preserved
        assert diff > 0.001, f"PE contribution too small: {diff}"

    def test_variable_seq_len(self, embed_dim):
        pe = PositionalEncoding(d_model=embed_dim, max_length=100, dropout=0.0)
        for sl in [1, 5, 50, 100]:
            out = pe(torch.randn(1, sl, embed_dim))
            assert out.shape == (1, sl, embed_dim)

    def test_seq_len_exceeding_max_length_raises(self, embed_dim):
        pe = PositionalEncoding(d_model=embed_dim, max_length=10, dropout=0.0)
        with pytest.raises(RuntimeError):
            pe(torch.randn(1, 20, embed_dim))


class TestLearnedPositionalEmbedding:
    def test_output_shape(self, batch_size, seq_len, embed_dim):
        lpe = LearnedPositionalEmbedding(d_model=embed_dim, max_len=100, dropout=0.0)
        x = torch.randn(batch_size, seq_len, embed_dim)
        out = lpe(x)
        assert out.shape == (batch_size, seq_len, embed_dim)

    def test_slices_to_seq_len(self, embed_dim):
        """The original bug: position_ids were not sliced to seq_len."""
        lpe = LearnedPositionalEmbedding(d_model=embed_dim, max_len=100, dropout=0.0)
        x = torch.randn(2, 5, embed_dim)
        # This should not raise a dimension mismatch error
        out = lpe(x)
        assert out.shape == (2, 5, embed_dim)

    def test_different_seq_lengths_work(self, embed_dim):
        lpe = LearnedPositionalEmbedding(d_model=embed_dim, max_len=100, dropout=0.0)
        for sl in [1, 10, 50, 100]:
            out = lpe(torch.randn(1, sl, embed_dim))
            assert out.shape == (1, sl, embed_dim)

    def test_embeddings_are_trainable(self, embed_dim):
        lpe = LearnedPositionalEmbedding(d_model=embed_dim, max_len=10, dropout=0.0)
        x = torch.randn(2, 5, embed_dim, requires_grad=True)
        out = lpe(x)
        loss = out.sum()
        loss.backward()
        assert lpe.position_embeddings.weight.grad is not None
        assert lpe.position_embeddings.weight.grad.abs().sum() > 0
