"""Tests for model forward passes and shapes."""
import torch
import pytest

from classifier_runner import (
    TransformerClassifier,
    NominalClassifier,
    RNNClassifier,
)


@pytest.fixture
def model_kwargs(embed_dim, num_classes):
    return {
        "num_classes": num_classes,
        "embed_size": embed_dim,
        "hidden_dim1": 32,
        "dropout_rate": 0.0,
        "max_length": 20,
    }


class TestTransformerClassifier:
    def test_output_shape(self, model_kwargs, batch_size, embed_dim, num_classes):
        model = TransformerClassifier(
            **model_kwargs, nhead=2, num_layers_transformer=1,
            dim_feedforward=64, device="cpu", pe_factor=1.0,
        )
        model.eval()
        x = torch.randn(batch_size, 10, embed_dim)
        lengths = torch.full((batch_size,), 10)
        out = model(x, lengths)
        assert out.shape == (batch_size, num_classes)

    def test_variable_lengths_in_batch(self, model_kwargs, embed_dim, num_classes):
        model = TransformerClassifier(
            **model_kwargs, nhead=2, num_layers_transformer=1,
            dim_feedforward=64, device="cpu", pe_factor=1.0,
        )
        model.eval()
        B, max_sl = 4, 10
        x = torch.randn(B, max_sl, embed_dim)
        lengths = torch.tensor([3, 7, 10, 5])
        out = model(x, lengths)
        assert out.shape == (B, num_classes)

    def test_padding_does_not_affect_output(self, model_kwargs, embed_dim, num_classes):
        """Padded positions should not change the output for valid positions."""
        model = TransformerClassifier(
            **model_kwargs, nhead=2, num_layers_transformer=1,
            dim_feedforward=64, device="cpu", pe_factor=1.0,
        )
        model.eval()

        # Create input with 3 valid positions
        valid_data = torch.randn(1, 3, embed_dim)

        # Run with no padding
        lengths_no_pad = torch.tensor([3])
        out_no_pad = model(valid_data, lengths_no_pad)

        # Run with padding appended
        padded = torch.cat([valid_data, torch.zeros(1, 7, embed_dim)], dim=1)
        lengths_padded = torch.tensor([3])
        out_padded = model(padded, lengths_padded)

        assert torch.allclose(out_no_pad, out_padded, atol=1e-5)

    def test_gradient_flows(self, model_kwargs, embed_dim):
        model = TransformerClassifier(
            **model_kwargs, nhead=2, num_layers_transformer=1,
            dim_feedforward=64, device="cpu", pe_factor=1.0,
        )
        x = torch.randn(2, 5, embed_dim, requires_grad=True)
        lengths = torch.tensor([5, 3])
        out = model(x, lengths)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_single_sequence_length(self, model_kwargs, embed_dim, num_classes):
        model_kwargs_copy = {**model_kwargs, "max_length": 5}
        model = TransformerClassifier(
            **model_kwargs_copy, nhead=2, num_layers_transformer=1,
            dim_feedforward=64, device="cpu", pe_factor=1.0,
        )
        model.eval()
        x = torch.randn(2, 1, embed_dim)
        lengths = torch.tensor([1, 1])
        out = model(x, lengths)
        assert out.shape == (2, num_classes)


class TestTransformerClassifierALiBi:
    def test_alibi_forward(self, model_kwargs, batch_size, embed_dim, num_classes):
        model = TransformerClassifier(
            **model_kwargs, nhead=2, num_layers_transformer=1,
            dim_feedforward=64, device="cpu", use_alibi=True, pe_factor=1.0,
        )
        model.eval()
        x = torch.randn(batch_size, 10, embed_dim)
        lengths = torch.full((batch_size,), 10)
        out = model(x, lengths)
        assert out.shape == (batch_size, num_classes)

    def test_alibi_with_padding(self, model_kwargs, embed_dim, num_classes):
        model = TransformerClassifier(
            **model_kwargs, nhead=2, num_layers_transformer=1,
            dim_feedforward=64, device="cpu", use_alibi=True, pe_factor=1.0,
        )
        model.eval()
        x = torch.randn(2, 8, embed_dim)
        lengths = torch.tensor([5, 3])
        out = model(x, lengths)
        assert out.shape == (2, num_classes)


class TestNominalClassifier:
    def test_output_shape(self, model_kwargs, batch_size, embed_dim, num_classes):
        model = NominalClassifier(**model_kwargs)
        model.eval()
        x = torch.randn(batch_size, 5, embed_dim)
        lengths = torch.full((batch_size,), 5)
        out = model(x, lengths)
        assert out.shape == (batch_size, num_classes)

    def test_no_softmax_in_output(self, model_kwargs, batch_size, embed_dim):
        """Output should be raw logits, NOT probabilities (no softmax)."""
        model = NominalClassifier(**model_kwargs)
        model.eval()
        x = torch.randn(batch_size, 3, embed_dim)
        lengths = torch.full((batch_size,), 3)
        out = model(x, lengths)
        # Raw logits can be negative and don't sum to 1
        has_negative = (out < 0).any().item()
        sums_to_one = torch.allclose(out.sum(dim=1), torch.ones(batch_size))
        # At least one of these should be true for raw logits
        assert has_negative or not sums_to_one

    def test_accepts_lengths_parameter(self, model_kwargs, embed_dim):
        """Verify the forward signature accepts `lengths` (was a bug)."""
        model = NominalClassifier(**model_kwargs)
        model.eval()
        x = torch.randn(2, 5, embed_dim)
        lengths = torch.tensor([5, 3])
        # Should not raise TypeError
        out = model(x, lengths)
        assert out.shape[0] == 2


class TestRNNClassifier:
    def test_output_shape(self, embed_dim, num_classes):
        model = RNNClassifier(
            num_classes=num_classes, embed_size=embed_dim,
            hidden_dim1=32, dropout_rate=0.0, num_rnn_layers=2,
        )
        model.eval()
        x = torch.randn(4, 10, embed_dim)
        lengths = torch.full((4,), 10)
        out = model(x, lengths)
        assert out.shape == (4, num_classes)

    def test_no_softmax_in_output(self, embed_dim, num_classes):
        """Output should be raw logits."""
        model = RNNClassifier(
            num_classes=num_classes, embed_size=embed_dim,
            hidden_dim1=32, dropout_rate=0.0, num_rnn_layers=2,
        )
        model.eval()
        x = torch.randn(4, 10, embed_dim)
        lengths = torch.full((4,), 10)
        out = model(x, lengths)
        has_negative = (out < 0).any().item()
        sums_to_one = torch.allclose(out.sum(dim=1), torch.ones(4))
        assert has_negative or not sums_to_one

    def test_accepts_lengths_parameter(self, embed_dim, num_classes):
        model = RNNClassifier(
            num_classes=num_classes, embed_size=embed_dim,
            hidden_dim1=32, dropout_rate=0.0, num_rnn_layers=2,
        )
        model.eval()
        x = torch.randn(2, 5, embed_dim)
        lengths = torch.tensor([5, 3])
        out = model(x, lengths)
        assert out.shape[0] == 2

    def test_num_rnn_layers_is_small(self, embed_dim, num_classes):
        """Verify num_layers is not accidentally set to max_length (was a bug)."""
        model = RNNClassifier(
            num_classes=num_classes, embed_size=embed_dim,
            hidden_dim1=32, num_rnn_layers=2, max_length=5000,
        )
        assert model.rnn.num_layers == 2
