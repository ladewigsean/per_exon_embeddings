"""Tests for MultiClassDataset and MultiClassSubset."""
import numpy as np
import torch
import pytest

from classifier_runner import MultiClassDataset, MultiClassSubset


class TestMultiClassDataset:
    def test_dataset_loads(self, mock_dataset_dir):
        h5_path, csv_path, num_classes, _ = mock_dataset_dir
        ds = MultiClassDataset(h5_path, csv_path)
        assert len(ds) == 40
        assert ds.num_classes == num_classes

    def test_getitem_returns_correct_types(self, mock_dataset_dir, embed_dim):
        h5_path, csv_path, _, _ = mock_dataset_dir
        ds = MultiClassDataset(h5_path, csv_path)
        embedding, label, identifier, seq_len = ds[0]
        assert isinstance(embedding, torch.Tensor)
        assert embedding.dtype == torch.float32
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.float32
        assert isinstance(identifier, str)
        assert isinstance(seq_len, int)

    def test_getitem_embedding_shape(self, mock_dataset_dir, embed_dim):
        h5_path, csv_path, _, max_seq_len = mock_dataset_dir
        ds = MultiClassDataset(h5_path, csv_path)
        embedding, _, _, seq_len = ds[0]
        # Should be padded to max_length
        assert embedding.shape == (ds.max_length, embed_dim)
        assert seq_len >= 1
        assert seq_len <= max_seq_len

    def test_label_is_one_hot(self, mock_dataset_dir, num_classes):
        h5_path, csv_path, _, _ = mock_dataset_dir
        ds = MultiClassDataset(h5_path, csv_path)
        _, label, _, _ = ds[0]
        assert label.shape == (num_classes,)
        assert label.sum().item() == pytest.approx(1.0)
        assert set(label.tolist()).issubset({0.0, 1.0})

    def test_max_length_computed_correctly(self, mock_dataset_dir):
        h5_path, csv_path, _, max_seq_len = mock_dataset_dir
        ds = MultiClassDataset(h5_path, csv_path)
        assert ds.max_length == max_seq_len

    def test_embedding_dim_detected(self, mock_dataset_dir, embed_dim):
        h5_path, csv_path, _, _ = mock_dataset_dir
        ds = MultiClassDataset(h5_path, csv_path)
        assert ds.embedding_dim == embed_dim

    def test_with_existing_label_encoder(self, mock_dataset_dir):
        h5_path, csv_path, _, _ = mock_dataset_dir
        ds1 = MultiClassDataset(h5_path, csv_path)
        ds2 = MultiClassDataset(h5_path, csv_path, label_encoder=ds1.label_encoder)
        assert ds2.num_classes == ds1.num_classes

    def test_missing_columns_raises(self, tmp_path, embed_dim):
        import h5py
        import pandas as pd

        h5_path = tmp_path / "bad.h5"
        csv_path = tmp_path / "bad.csv"
        with h5py.File(h5_path, "w") as h5f:
            h5f.create_dataset("s0", data=np.zeros((3, embed_dim), dtype=np.float32))
        pd.DataFrame({"id": ["s0"], "label": ["X"]}).to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="identifier"):
            MultiClassDataset(str(h5_path), str(csv_path))


class TestMultiClassSubset:
    def test_subset_creation(self, mock_dataset_dir):
        h5_path, csv_path, _, _ = mock_dataset_dir
        ds = MultiClassDataset(h5_path, csv_path)
        indices = np.arange(10)
        subset = MultiClassSubset(ds, indices)
        assert len(subset) == 10
        assert subset.num_classes == ds.num_classes

    def test_subset_getitem(self, mock_dataset_dir, embed_dim):
        h5_path, csv_path, _, _ = mock_dataset_dir
        ds = MultiClassDataset(h5_path, csv_path)
        indices = np.arange(5)
        subset = MultiClassSubset(ds, indices)
        embedding, label, identifier, seq_len = subset[0]
        assert embedding.shape[1] == embed_dim
        assert label.sum().item() == pytest.approx(1.0)
