"""Shared fixtures for all tests."""
import os
import tempfile

import h5py
import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def embed_dim():
    return 64


@pytest.fixture
def num_classes():
    return 4


@pytest.fixture
def batch_size():
    return 8


@pytest.fixture
def seq_len():
    return 10


@pytest.fixture
def mock_dataset_dir(tmp_path, embed_dim):
    """Create a temporary directory with a small HDF5 + CSV dataset.

    Returns (h5_path, csv_path, num_classes, max_seq_len).
    """
    num_samples = 40
    genes = ["BRCA1", "TP53", "EGFR", "KRAS"]
    max_seq_len = 6

    csv_path = tmp_path / "train.csv"
    h5_path = tmp_path / "train.h5"

    rows = []
    rng = np.random.default_rng(42)
    with h5py.File(h5_path, "w") as h5f:
        for i in range(num_samples):
            identifier = f"sample_{i:04d}"
            gene = genes[i % len(genes)]
            # Vary sequence length between 1 and max_seq_len
            sl = rng.integers(1, max_seq_len + 1)
            embedding = rng.standard_normal((sl, embed_dim)).astype(np.float32)
            h5f.create_dataset(identifier, data=embedding)
            rows.append({"identifier": identifier, "gene": gene})

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    return str(h5_path), str(csv_path), len(genes), max_seq_len
