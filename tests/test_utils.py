"""Tests for utility functions: gen_pad_mask_bool, masked_mean."""
import torch
import pytest

from classifier_runner import gen_pad_mask_bool, masked_mean


class TestGenPadMaskBool:
    def test_basic_shape(self):
        lengths = torch.tensor([3, 5, 2])
        mask = gen_pad_mask_bool(max_length=6, lengths=lengths, device="cpu")
        assert mask.shape == (3, 6)
        assert mask.dtype == torch.bool

    def test_values_correct(self):
        lengths = torch.tensor([3, 1])
        mask = gen_pad_mask_bool(max_length=5, lengths=lengths, device="cpu")
        # First sample: 3 valid positions, 2 padded
        expected_0 = torch.tensor([False, False, False, True, True])
        assert torch.equal(mask[0], expected_0)
        # Second sample: 1 valid position, 4 padded
        expected_1 = torch.tensor([False, True, True, True, True])
        assert torch.equal(mask[1], expected_1)

    def test_no_padding(self):
        lengths = torch.tensor([5, 5])
        mask = gen_pad_mask_bool(max_length=5, lengths=lengths, device="cpu")
        assert not mask.any()

    def test_all_padding_except_one(self):
        lengths = torch.tensor([1])
        mask = gen_pad_mask_bool(max_length=10, lengths=lengths, device="cpu")
        assert mask[0, 0] == False
        assert mask[0, 1:].all()


class TestMaskedMean:
    def test_without_padding(self):
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])  # [1, 3, 2]
        mask = torch.tensor([[False, False, False]])  # no padding
        result = masked_mean(x, mask)
        expected = torch.tensor([[3.0, 4.0]])  # mean of all 3 positions
        assert torch.allclose(result, expected)

    def test_with_padding(self):
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [999.0, 999.0]]])  # [1, 3, 2]
        mask = torch.tensor([[False, False, True]])  # last position padded
        result = masked_mean(x, mask)
        expected = torch.tensor([[2.0, 3.0]])  # mean of first 2 only
        assert torch.allclose(result, expected)

    def test_single_valid_position(self):
        x = torch.tensor([[[7.0, 8.0], [0.0, 0.0], [0.0, 0.0]]])
        mask = torch.tensor([[False, True, True]])
        result = masked_mean(x, mask)
        expected = torch.tensor([[7.0, 8.0]])
        assert torch.allclose(result, expected)

    def test_batch_dimension(self):
        x = torch.tensor([
            [[1.0, 0.0], [3.0, 0.0], [0.0, 0.0]],
            [[2.0, 0.0], [4.0, 0.0], [6.0, 0.0]],
        ])
        mask = torch.tensor([
            [False, False, True],   # 2 valid
            [False, False, False],  # 3 valid
        ])
        result = masked_mean(x, mask)
        assert torch.allclose(result[0], torch.tensor([2.0, 0.0]))
        assert torch.allclose(result[1], torch.tensor([4.0, 0.0]))

    def test_output_shape(self):
        B, S, E = 4, 10, 32
        x = torch.randn(B, S, E)
        mask = torch.zeros(B, S, dtype=torch.bool)
        result = masked_mean(x, mask)
        assert result.shape == (B, E)
