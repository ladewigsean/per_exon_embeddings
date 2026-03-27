## Summary

This PR fixes 14 bugs in the classifier pipeline and adds a comprehensive test suite (77 tests). Every fix has a `BUG FIX:` comment explaining what was wrong and why.

**Please review each fix carefully** — understanding these bugs will help you avoid similar issues in future work.

## Critical Bugs Fixed

### 1. Double Softmax in NominalClassifier and RNNClassifier
Fixed

### 2. RNNClassifier had 5000 layers
Fixed

### 3. Positional Encoding factor default was 0.0 / 0.01
Fixed

### 4. ALiBi Transformer ignored padding mask
The padding mask was computed but **never passed** to the ALiBi transformer. Padded positions contributed to attention and mean pooling.
- **Fix:** Added `padding_mask` parameter through the full stack: `ALiBiTransformer` → `ALiBiTransformerLayer` → `ALiBiMultiHeadAttention`. Changed `nn.Sequential` to `nn.ModuleList` (Sequential can't pass extra args).

### 5. Checkpoint logged wrong key
Fixed

### 6. `kfolds` parameter never forwarded
Fixed

## Other Fixes
- Removed unused `conv_layer` (created but never used in forward) (Fixed)
- Removed commented-out WandB API key from source code (But I want to see it :( )
- `get_max_length()` now reads `.shape` metadata instead of loading full arrays(Fixed)
- Removed ~60 lines of dead commented-out code at end of file(Sry need this just incase)
- Cleaned up imports (removed unused `sys`(use this for when I quickly debug and use sys.exit()), `argparse` not needed at top(still at top in this version ?), `confusion_matrix` moved to where used(Ok moved but why?))

## Test Suite (77 tests)

| File | What it covers |
|------|---------------|
| `test_positional_encoding.py` | PE shapes, float32 precision, factor scaling |
| `test_utils.py` | `gen_pad_mask_bool`, `masked_mean` with/without padding |
| `test_models.py` | All 3 model types: shapes, no double-softmax, ALiBi |
| `test_dataset.py` | Dataset loading, types, shapes, label encoding |
| `test_alibi.py` | Relative positions, slopes, attention + padding mask |
| `test_trainer.py` | Training loop, early stopping, checkpointing |
| `test_integration.py` | Overfit test, PE signal test, padding invariance |

### How to run tests
```bash
cd classifier
pip install -e ".[dev]"
pytest tests/ -v
```

## How to review

Every bug fix has a `BUG FIX:` comment in the code explaining what was wrong. Search for `BUG FIX:` to find them all. The tests in `tests/` verify each fix works.
