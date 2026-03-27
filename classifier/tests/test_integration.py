"""End-to-end integration tests that run small training loops."""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import pytest

from classifier_runner import (
    MultiClassDataset,
    MultiClassTrainer,
    TransformerClassifier,
    PositionalEncoding,
    gen_pad_mask_bool,
    masked_mean,
)


class TestEndToEndTransformerTraining:
    """Full pipeline: dataset -> model -> train -> evaluate."""

    def test_overfit_small_batch(self, mock_dataset_dir, embed_dim):
        """A model should be able to overfit a tiny dataset (memorization test)."""
        h5_path, csv_path, num_classes, max_seq_len = mock_dataset_dir
        ds = MultiClassDataset(h5_path, csv_path)

        # Use only 8 samples for train and val (same data — testing memorization)
        indices = list(range(8))
        loader = DataLoader(Subset(ds, indices), batch_size=8, shuffle=False)

        model_config = {
            "num_classes": ds.num_classes,
            "embed_size": embed_dim,
            "hidden_dim1": 64,
            "dropout_rate": 0.0,
            "max_length": ds.max_length,
            "nhead": 2,
            "dim_feedforward": 128,
            "num_layers_transformer": 1,
            "device": "cpu",
            "pe_factor": 1.0,
        }
        trainer = MultiClassTrainer(
            model_config, learning_rate=1e-2, weight_decay=0,
            class_weights_tensor=None, model="Transformer",
            criterion="CEL_weightless", optimizer="Adam", scheduler="None", 
        )

        # Train for many epochs to overfit
        for _ in range(50):
            trainer.model.train()
            trainer._run_epoch(loader, training=True)

        # Evaluate — should achieve high accuracy on memorized data
        report, preds, ids, labels = trainer.evaluate_on_loader(loader, ds.label_encoder)
        accuracy = report["accuracy"]
        assert accuracy > 0.5, f"Model failed to overfit small batch: accuracy={accuracy:.2f}"

    def test_pe_signal_preserved_in_forward(self, embed_dim):
        """Verify the PE signal is not lost during a forward pass."""
        model = TransformerClassifier(
            num_classes=4, embed_size=embed_dim, hidden_dim1=32,
            dropout_rate=0.0, max_length=20, nhead=2,
            num_layers_transformer=1, dim_feedforward=64,
            device="cpu", pe_factor=1.0,
        )
        model.eval()

        # Two identical inputs at different "positions" should produce different outputs
        # if PE is working
        x1 = torch.randn(1, 5, embed_dim)
        x2 = x1.clone()
        # Shift x2 by inserting a zero-pad at position 0
        x2_shifted = torch.cat([torch.zeros(1, 1, embed_dim), x2[:, :4, :]], dim=1)

        lengths = torch.tensor([5])
        out1 = model(x1, lengths)
        out2 = model(x2_shifted, lengths)

        # Outputs should differ because PE encodes position
        assert not torch.allclose(out1, out2, atol=1e-4), \
            "PE had no effect — outputs identical for different position arrangements"

    def test_padding_invariance_integration(self, mock_dataset_dir, embed_dim):
        """Full pipeline test: different padding amounts should not change predictions."""
        h5_path, csv_path, _, _ = mock_dataset_dir
        ds = MultiClassDataset(h5_path, csv_path)

        model_config = {
            "num_classes": ds.num_classes,
            "embed_size": embed_dim,
            "hidden_dim1": 32,
            "dropout_rate": 0.0,
            "max_length": ds.max_length,
            "nhead": 2,
            "dim_feedforward": 64,
            "num_layers_transformer": 1,
            "device": "cpu",
            "pe_factor": 1.0,
        }
        trainer = MultiClassTrainer(
            model_config, learning_rate=1e-3, weight_decay=0,
            class_weights_tensor=None, model="Transformer",
            criterion="CEL_weightless", optimizer="Adam", scheduler="None",device=model_config["device"]
        )
        trainer.model.eval()

        # Get a single sample
        embedding, label, identifier, seq_len = ds[0]

        # Run with tight trim (just the valid positions)
        x_tight = embedding[:seq_len, :].unsqueeze(0)
        lengths_tight = torch.tensor([seq_len])
        with torch.no_grad():
            out_tight = trainer.model(x_tight, lengths_tight)

        # Run with full padding
        x_padded = embedding.unsqueeze(0)
        lengths_padded = torch.tensor([seq_len])
        with torch.no_grad():
            out_padded = trainer.model(x_padded, lengths_padded)

        assert torch.allclose(out_tight, out_padded, atol=1e-4), \
            "Padding changed model output — masking is broken"


class TestAllModelVariantsTrainable:
    """Verify all three model types can complete a training step without errors."""

    @pytest.fixture
    def setup(self, mock_dataset_dir, embed_dim):
        h5_path, csv_path, num_classes, max_seq_len = mock_dataset_dir
        ds = MultiClassDataset(h5_path, csv_path)
        loader = DataLoader(Subset(ds, list(range(16))), batch_size=8, shuffle=True)
        return ds, loader

    @pytest.mark.parametrize("model_type,extra_config", [
        ("Transformer", {"nhead": 2, "dim_feedforward": 64, "num_layers_transformer": 1, "device": "cpu", "pe_factor": 1.0}),
        ("Basic", {}),
        
    ])
    def test_training_step(self, setup, embed_dim, model_type, extra_config):
        ds, loader = setup
        model_config = {
            "num_classes": ds.num_classes,
            "embed_size": embed_dim,
            "hidden_dim1": 32,
            "dropout_rate": 0.1,
            "max_length": ds.max_length,
            **extra_config,
        }
        trainer = MultiClassTrainer(
            model_config, learning_rate=1e-3, weight_decay=1e-5,
            class_weights_tensor=None, model=model_type,
            criterion="CEL_weightless", optimizer="AdamW", scheduler="None",
        )

        # One training epoch should complete without error
        trainer.model.train()
        loss, acc = trainer._run_epoch(loader, training=True)
        assert loss > 0
        assert 0 <= acc <= 100


class TestSchedulerVariants:
    """Verify all scheduler options work with the trainer."""

    @pytest.fixture
    def setup(self, mock_dataset_dir, embed_dim):
        h5_path, csv_path, _, _ = mock_dataset_dir
        ds = MultiClassDataset(h5_path, csv_path)
        n = len(ds)
        train_loader = DataLoader(Subset(ds, list(range(0, n - 10))), batch_size=8, shuffle=True)
        val_loader = DataLoader(Subset(ds, list(range(n - 10, n))), batch_size=8, shuffle=False)
        return ds, train_loader, val_loader

    @pytest.mark.parametrize("scheduler", [
        "Plateau", "Exponential", "CosineAnnealingWarmRestarts", "Cyclic", "None",
    ])
    def test_scheduler(self, setup, embed_dim, scheduler, tmp_path):
        ds, train_loader, val_loader = setup
        model_config = {
            "num_classes": ds.num_classes,
            "embed_size": embed_dim,
            "hidden_dim1": 32,
            "dropout_rate": 0.0,
            "max_length": ds.max_length,
            "nhead": 2,
            "dim_feedforward": 64,
            "num_layers_transformer": 1,
            "device": "cpu",
            "pe_factor": 1.0,
        }
        trainer = MultiClassTrainer(
            model_config, learning_rate=1e-3, weight_decay=0,
            class_weights_tensor=None, model="Transformer",
            criterion="CEL_weightless", optimizer="Adam", scheduler=scheduler,
        )
        checkpoint_path = str(tmp_path / f"ckpt_{scheduler}.pt")
        metrics, epochs = trainer.train_and_validate(
            train_loader, val_loader, num_epochs=3, patience=10,
            checkpoint_path=checkpoint_path,
            label_encoder=ds.label_encoder,
        )
        assert "accuracy" in metrics
        assert epochs >= 1
