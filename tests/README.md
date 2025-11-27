# Test Suite

This directory contains comprehensive tests for models and training, focusing on:
- Model building from configs
- Class interactions and component relationships
- Component graph generation
- Checkpoint save/load structure
- Training loop structure

## Test Structure

```
tests/
├── conftest.py              # Shared pytest fixtures
├── test_models/             # Model tests
│   ├── test_autoencoder.py
│   ├── test_diffusion.py
│   ├── test_components.py
│   ├── test_checkpointing.py
│   └── test_component_graphs.py
├── test_training/           # Training tests
│   ├── test_train_autoencoder.py
│   ├── test_train_diffusion.py
│   └── test_trainer.py
└── test_experiments/        # Experiment config tests
    ├── test_config_loading.py
    └── test_experiment_builds.py
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test file
```bash
pytest tests/test_models/test_autoencoder.py
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Run specific test class
```bash
pytest tests/test_models/test_autoencoder.py::TestAutoencoderBuilding
```

## Test Fixtures

Test fixtures are defined in `conftest.py`:
- `autoencoder_model` - Autoencoder model instance
- `diffusion_model` - Diffusion model instance
- `test_autoencoder_config` - Minimal autoencoder config
- `test_diffusion_config` - Minimal diffusion config
- `temp_dir` - Temporary directory for test outputs

## Test Configs

Minimal test configs are in `tests/fixtures/configs/`:
- `test_autoencoder_minimal.yaml` - Minimal autoencoder config
- `test_vae_minimal.yaml` - Minimal VAE config
- `test_diffusion_minimal.yaml` - Minimal diffusion config

These configs use:
- Small batch sizes (2)
- Minimal epochs (1)
- CPU-friendly settings
- No external dependencies

## What's Tested

### Model Tests
- Model initialization from configs
- Component existence and types
- Component relationships
- Component graph generation
- Checkpoint save/load structure

### Training Tests
- Trainer initialization
- Training loop structure
- Checkpoint saving during training
- Resume from checkpoint

### Experiment Tests
- Config loading and validation
- Building models from experiment configs
- Config compatibility

## Note

These tests focus on **structure and building**, not correctness of model outputs or training results. They verify that:
- Models can be built from configs
- Components are properly connected
- Checkpoints can be saved and loaded
- Training infrastructure works

For correctness testing, see integration tests or manual evaluation.

