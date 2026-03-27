"""Tests for MultiClassTrainer training loop."""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import pytest

from classifier_runner import (
    MultiClassDataset,
    MultiClassTrainer,
)


@pytest.fixture
def small_dataset(mock_dataset_dir, embed_dim):
    h5_path, csv_path, num_classes, max_seq_len = mock_dataset_dir
    ds = MultiClassDataset(h5_path, csv_path)
    return ds


@pytest.fixture
def train_val_loaders(small_dataset):
    n = len(small_dataset)
    train_ds = Subset(small_dataset, list(range(0, n - 10)))
    val_ds = Subset(small_dataset, list(range(n - 10, n)))
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    return train_loader, val_loader


class TestMultiClassTrainer:
    def test_transformer_training_runs(self, small_dataset, train_val_loaders, tmp_path, embed_dim):
        train_loader, val_loader = train_val_loaders
        model_config = {
            "num_classes": small_dataset.num_classes,
            "embed_size": embed_dim,
            "hidden_dim1": 32,
            "dropout_rate": 0.0,
            "max_length": small_dataset.max_length,
            "nhead": 2,
            "dim_feedforward": 64,
            "num_layers_transformer": 1,
            "device": "cpu",
            "pe_factor": 1.0,
        }
        trainer = MultiClassTrainer(
            model_config, learning_rate=1e-3, weight_decay=0,
            class_weights_tensor=None, model="Transformer",
            criterion="CEL_weightless", optimizer="Adam", scheduler="None",
        )
        checkpoint_path = str(tmp_path / "test_ckpt.pt")
        metrics, epochs = trainer.train_and_validate(
            train_loader, val_loader, num_epochs=3, patience=10,
            checkpoint_path=checkpoint_path,
            label_encoder=small_dataset.label_encoder,
        )
        assert "macro avg" in metrics
        assert epochs >= 1

    def test_basic_model_training_runs(self, small_dataset, train_val_loaders, tmp_path, embed_dim):
        train_loader, val_loader = train_val_loaders
        model_config = {
            "num_classes": small_dataset.num_classes,
            "embed_size": embed_dim,
            "hidden_dim1": 32,
            "dropout_rate": 0.0,
            "max_length": small_dataset.max_length,
        }
        trainer = MultiClassTrainer(
            model_config, learning_rate=1e-3, weight_decay=0,
            class_weights_tensor=None, model="Basic",
            criterion="CEL_weightless", optimizer="Adam", scheduler="None",
        )
        checkpoint_path = str(tmp_path / "test_ckpt.pt")
        metrics, epochs = trainer.train_and_validate(
            train_loader, val_loader, num_epochs=2, patience=10,
            checkpoint_path=checkpoint_path,
            label_encoder=small_dataset.label_encoder,
        )
        assert "accuracy" in metrics

    

    def test_early_stopping(self, small_dataset, train_val_loaders, tmp_path, embed_dim):
        train_loader, val_loader = train_val_loaders
        model_config = {
            "num_classes": small_dataset.num_classes,
            "embed_size": embed_dim,
            "hidden_dim1": 32,
            "dropout_rate": 0.0,
            "max_length": small_dataset.max_length,
            "nhead": 2,
            "dim_feedforward": 64,
            "num_layers_transformer": 1,
            "device": "cpu",
            "pe_factor": 1.0,
        }
        trainer = MultiClassTrainer(
            model_config, learning_rate=1e-3, weight_decay=0,
            class_weights_tensor=None, model="Transformer",
            criterion="CEL_weightless", optimizer="Adam", scheduler="None",
        )
        checkpoint_path = str(tmp_path / "test_ckpt.pt")
        # patience=1 should trigger early stopping quickly
        metrics, epochs = trainer.train_and_validate(
            train_loader, val_loader, num_epochs=100, patience=1,
            checkpoint_path=checkpoint_path,
            label_encoder=small_dataset.label_encoder,
        )
        # Should stop well before 100 epochs
        assert epochs < 100

    def test_checkpoint_saved_and_loaded(self, small_dataset, train_val_loaders, tmp_path, embed_dim):
        train_loader, val_loader = train_val_loaders
        model_config = {
            "num_classes": small_dataset.num_classes,
            "embed_size": embed_dim,
            "hidden_dim1": 32,
            "dropout_rate": 0.0,
            "max_length": small_dataset.max_length,
            "nhead": 2,
            "dim_feedforward": 64,
            "num_layers_transformer": 1,
            "device": "cpu",
            "pe_factor": 1.0,
        }
        trainer = MultiClassTrainer(
            model_config, learning_rate=1e-3, weight_decay=0,
            class_weights_tensor=None, model="Transformer",
            criterion="CEL_weightless", optimizer="Adam", scheduler="None",
        )
        checkpoint_path = str(tmp_path / "test_ckpt.pt")
        trainer.train_and_validate(
            train_loader, val_loader, num_epochs=3, patience=10,
            checkpoint_path=checkpoint_path,
            label_encoder=small_dataset.label_encoder,
        )
        assert os.path.exists(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        assert "model_state_dict" in ckpt
        assert "val_f1_macro" in ckpt

    def test_evaluate_on_loader(self, small_dataset, train_val_loaders, embed_dim):
        _, val_loader = train_val_loaders
        model_config = {
            "num_classes": small_dataset.num_classes,
            "embed_size": embed_dim,
            "hidden_dim1": 32,
            "dropout_rate": 0.0,
            "max_length": small_dataset.max_length,
            "nhead": 2,
            "dim_feedforward": 64,
            "num_layers_transformer": 1,
            "device": "cpu",
            "pe_factor": 1.0,
        }
        trainer = MultiClassTrainer(
            model_config, learning_rate=1e-3, weight_decay=0,
            class_weights_tensor=None, model="Transformer",
            criterion="CEL_weightless", optimizer="Adam", scheduler="None",
        )
        report, preds, ids, labels = trainer.evaluate_on_loader(
            val_loader, small_dataset.label_encoder,
        )
        assert "accuracy" in report
        assert len(preds) == len(labels)
        assert len(ids) == len(labels)
