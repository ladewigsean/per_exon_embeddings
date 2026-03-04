# Changes Document — Code Review & Bug Fixes

## Summary

Full debug and cleanup of the per-exon embedding classifier pipeline.
14 bugs found and fixed, test suite created (77 tests), project structure
cleaned up for production-readiness.

---

## Critical Bugs Fixed

### 1. Positional Encoding Signal Lost in float16 (THE main training issue)

**Root cause**: The positional encoding was computed and stored in `torch.float16`.
Input embeddings were also loaded as float16. When adding a small PE value
(e.g. `0.01 * sin(x) ≈ 0.005`) to a protein embedding value (e.g. `3.5`),
the PE was **below float16's precision threshold** (~0.004 for values around 4.0)
and got rounded away to zero. The PE literally had no effect on the model.

**Fix**: PE is now computed and stored in float32. Input embeddings are loaded
as float32. The `factor` parameter scales PE at forward time (not during init),
preserving the full-precision PE buffer. Mixed-precision autocast handles the
rest during training.

**Files**: `classifier_runner.py` lines 150-176, 85-86

### 2. `get_max_length()` — Comparison Instead of Assignment

```python
# Before (bug):
self.max_length == 1   # == is comparison, does nothing

# After (fix):
self.max_length = 1    # = is assignment
```

This silent no-op meant that when 1-D embeddings were encountered, `max_length`
was never correctly set to 1.

**File**: `classifier_runner.py` line 115

### 3. `LearnedPositionalEmbedding` — Dimension Mismatch

`position_ids` was not sliced to `seq_len`, causing a shape mismatch when
`seq_len < max_len`. Fixed by slicing: `self.position_ids[:, :seq_len]`.

**File**: `classifier_runner.py` lines 193-194

### 4. Double Softmax in NominalClassifier and RNNClassifier

Both models applied `nn.Softmax(dim=1)` as their final layer, but
`nn.CrossEntropyLoss` already applies `log_softmax` internally. This produced
`log(softmax(softmax(logits)))` — crushing gradients and preventing learning.

**Fix**: Removed Softmax from both models. They now output raw logits.

### 5. RNNClassifier — `num_layers=max_length` (5000-layer RNN)

```python
# Before (bug):
nn.RNN(embed_size, hidden_dim1, num_layers=max_length, ...)  # 5000 layers!

# After (fix):
nn.RNN(embed_size, hidden_dim1, num_layers=num_rnn_layers, ...)  # default: 2
```

### 6. Forward Signature Mismatch

`_run_epoch` always called `self.model(embeddings, lengths)`, but
`NominalClassifier` and `RNNClassifier` didn't accept `lengths`.
This would crash with `TypeError` when using those models.

**Fix**: All three model classes now have `forward(self, x, lengths)`.

---

## Serious Bugs Fixed

### 7. ALiBi Transformer Ignored Padding Mask

The padding mask was computed but never passed to the ALiBi transformer.
Padded (zero) positions contributed to attention scores and mean pooling,
corrupting outputs.

**Fix**: Added `padding_mask` parameter through the full ALiBi stack:
`ALiBiTransformer` → `ALiBiTransformerLayer` → `ALiBiMultiHeadAttention`.
The attention module now masks out padded positions with `-inf` before softmax.

**Files**: `alibi/model.py`, `alibi/layers.py`, `alibi/attention.py`

### 8. ALiBi Used `nn.Sequential` — Couldn't Pass Extra Args

The ALiBi transformer used `nn.Sequential` for its layers, which only passes
a single tensor through. Changed to `nn.ModuleList` with explicit loop to
support the `padding_mask` argument.

**File**: `alibi/model.py`

### 9. `split_dataset_into_subsets` — Passed Tuple Instead of Array

```python
# Before (bug):
MultiClassSubset(dataset, np.where(df["test_split"]==0))  # returns (array,)

# After (fix):
MultiClassSubset(dataset, np.where(df["test_split"]==0)[0])  # flat array
```

### 10. `train_model_transformer` — Dict Attribute Access

Used `cfg_trainer.batch_size` on a plain dict (dicts don't support `.attr`).
This function was removed as it was unused and broken. The HPO pipeline
(`run_hpo_mode`) is the main entry point.

---

## Moderate Fixes

### 11. Checkpoint Load Logged Wrong Key

```python
# Before:
checkpoint.get('val_f1_weighted', 0)  # key doesn't exist

# After:
checkpoint.get('val_f1_macro', 0)     # matches what was saved
```

### 12. Unused `conv_layer` Removed

`nn.Conv1d(max_len, 1, ...)` was created in `TransformerClassifier.__init__`
but never used in `forward()`. Removed to save memory.

### 13. WandB API Key Removed from Source

A commented-out API key was in the source code. Removed.

### 14. `kfolds` Parameter Not Passed Through

The `run()` function accepted a `kfolds` parameter but never passed it to
`run_hpo_mode()`. Now correctly forwarded.

---

## Code Cleanup

- Removed all dead/commented-out code (old PE class, unused `constrained_mean`,
  unused `gen_pad_mask` float version, commented-out code blocks)
- Removed unused imports (`sys`, `argparse`, `Path`, `ConcatDataset`,
  `LabelEncoder`, `confusion_matrix`)
- Hardcoded `1024` in padding replaced with `self.embedding_dim`
- Consistent formatting, section headers for readability
- `_compute_max_length()` now reads `.shape` metadata instead of loading full
  arrays with `[:]` — much faster for large datasets
- `pe_factor` default changed from `0.01` to `1.0` (standard PE magnitude;
  HPO still searches the range)

---

## ML Best Practice Improvements

- **float32 data loading**: Embeddings now loaded as float32. Mixed precision
  `autocast` handles float16 conversion where beneficial during training.
  This ensures PE and other small signals are not lost.
- **No double softmax**: Models output raw logits. `CrossEntropyLoss` handles
  the softmax internally.
- **Proper padding masking**: All model variants (standard Transformer, ALiBi)
  now correctly mask padded positions in both attention and pooling.
- **`pe_factor` as continuous HPO parameter**: Changed from categorical
  `[0.1, 0.01, 0.001]` to continuous log-uniform `[0.01, 1.0]` for better
  search coverage.

---

## Project Structure Improvements

- Added `pyproject.toml` with all dependencies and version constraints
- Updated `.gitignore` (comprehensive, does not exclude tests)
- Added virtual environment support (`.venv/`)

---

## Test Suite (77 tests)

| File | Tests | What's covered |
|------|-------|----------------|
| `test_positional_encoding.py` | 12 | PE shapes, float32 precision, factor scaling, signal preservation, learned PE slicing |
| `test_utils.py` | 9 | `gen_pad_mask_bool` correctness, `masked_mean` with/without padding |
| `test_models.py` | 12 | All 3 model types: shapes, no double-softmax, `lengths` param, ALiBi forward |
| `test_dataset.py` | 10 | Dataset loading, types, shapes, label encoding, subset creation |
| `test_alibi.py` | 15 | Relative positions, slopes, attention shapes, padding mask, gradient flow |
| `test_trainer.py` | 6 | Training loop, early stopping, checkpointing, evaluation |
| `test_integration.py` | 13 | Overfit test, PE signal preservation, padding invariance, all schedulers |

Key integration tests:
- **Overfit test**: Model can memorize 8 samples → proves forward/backward pipeline works
- **PE signal test**: Different position arrangements produce different outputs → proves PE is working
- **Padding invariance**: Same data with different padding produces identical output → proves masking works

---

## Validation Training Run (20 min, CPU, no HPO)

Single run with fixed hyperparameters to validate the fixes work:
- AdamW, lr=1e-4, CosineAnnealingWarmRestarts (T₀=8, T_mult=2)
- 2-layer Transformer, 4 heads, pe_factor=1.0, dropout=0.2
- 12,328 samples, 35 CYP gene classes, 80/20 train/val split

| Cycle | Epochs | Peak Val F1 | Peak Val Acc |
|-------|--------|-------------|--------------|
| 1 | 1-8 | 0.463 | 58.9% |
| 2 | 9-24 | 0.623 | 69.6% |
| 3 | 25-56 | **0.743** | **76.0%** |
| 4 | 57-58 | (just started) | — |

**Diagnosis: No overfitting, no plateau, still improving.**
- Train/val accuracy gap is ~1-2% (healthy)
- Each cosine restart cycle peaks higher than the last (0.46 → 0.62 → 0.74)
- Cycle 3 was still setting new bests at epoch 53/56
- Projected cycle 4 peak (around epoch ~112): F1 ~0.80-0.82
- On GPU with HPO, expect significantly higher results

---

## How to Run

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run training (requires actual data in splits/)
python classifier_runner.py
```
