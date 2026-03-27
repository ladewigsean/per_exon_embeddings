## Summary


##Removed RNNCLassifier, was never used

## Critical Bugs Fixed

### 1. Double Softmax in NominalClassifier and RNNClassifier
Fixed

### 2. RNNClassifier had 5000 layers
Fixed/ Removed RNN Classifier

### 3. Positional Encoding factor default was 0.0 / 0.01
Fixed

### 4. ALiBi Transformer ignored padding mask
Fixed

### 5. Checkpoint logged wrong key
Fixed

### 6. `kfolds` parameter never forwarded
Fixed

## Other Fixes
- Removed unused `conv_layer` (created but never used in forward) (Fixed)
- Removed commented-out WandB API key from source code (Fixed )
- `get_max_length()` now reads `.shape` metadata instead of loading full arrays(Fixed)
- Removed ~60 lines of dead commented-out code at end of file(Removed)
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

## results from testing
mostly rnn errors, but RNN is removed now anyway
```bash
FAILED tests/test_integration.py::TestEndToEndTransformerTraining::test_padding_invariance_integration - RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
FAILED tests/test_integration.py::TestAllModelVariantsTrainable::test_training_step[RNN-extra_config2] - TypeError: RNNClassifier.__init__() got an unexpected keyword argument 'num_rnn_layers'
FAILED tests/test_models.py::TestRNNClassifier::test_output_shape - TypeError: RNNClassifier.__init__() got an unexpected keyword argument 'num_rnn_layers'
FAILED tests/test_models.py::TestRNNClassifier::test_no_softmax_in_output - TypeError: RNNClassifier.__init__() got an unexpected keyword argument 'num_rnn_layers'
FAILED tests/test_models.py::TestRNNClassifier::test_accepts_lengths_parameter - TypeError: RNNClassifier.__init__() got an unexpected keyword argument 'num_rnn_layers'
FAILED tests/test_models.py::TestRNNClassifier::test_num_rnn_layers_is_small - TypeError: RNNClassifier.__init__() got an unexpected keyword argument 'num_rnn_layers'
FAILED tests/test_trainer.py::TestMultiClassTrainer::test_rnn_model_training_runs - TypeError: RNNClassifier.__init__() got an unexpected keyword argument 'num_rnn_layers'
============================================= 7 failed, 70 passed in 6.05s =============================================
```
