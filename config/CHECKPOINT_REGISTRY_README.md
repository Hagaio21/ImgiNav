# Checkpoint Registry System

The checkpoint registry system allows you to reference checkpoints by name instead of hardcoded paths, making your configs portable across different environments (local development and HPC clusters).

## Overview

Instead of hardcoding paths like:
```yaml
autoencoder:
  checkpoint: checkpoints/vae_checkpoint_best.pt
```

You can use registry references:
```yaml
autoencoder:
  checkpoint: "@vae_best"
```

The registry automatically resolves these references to the correct path based on your environment (local or HPC).

## Environment Detection

The system automatically detects your environment:

- **Local**: Default when running on your development machine
- **HPC**: Detected when:
  - `IMGINAV_ENV=hpc` is set
  - SLURM or PBS job scheduler is detected
  - Hostname contains HPC-related patterns

You can also explicitly set the environment:
```bash
export IMGINAV_ENV=local  # or hpc
```

## Registry File

The registry is defined in `config/checkpoint_registry.yaml`:

```yaml
checkpoints:
  vae_best:
    local: checkpoints/vae_checkpoint_best.pt
    hpc: ${IMGINAV_ROOT}/checkpoints/vae_checkpoint_best.pt
    description: "Standalone VAE checkpoint"
```

## Usage in Configs

### Basic Usage

Reference checkpoints using `@checkpoint_name`:

```yaml
# Experiment config
autoencoder:
  checkpoint: "@vae_best"

embedding_projection:
  type: CLIPEmbeddingToSpatial
  clip_projections: "@clip_projection_best"
```

### Automatic Resolution

When you load a config using `load_config()` or `load_config_with_profile()`, registry references are automatically resolved:

```python
from common.utils import load_config

config = load_config("experiments/diffusion/my_experiment.yaml")
# Registry references like "@vae_best" are automatically resolved to full paths
```

### Manual Resolution

You can also resolve paths manually:

```python
from common.checkpoint_registry import get_checkpoint_path

# Get checkpoint path
checkpoint_path = get_checkpoint_path("@vae_best")
# Returns: Path("checkpoints/vae_checkpoint_best.pt") on local
#       or: Path("/work3/.../checkpoints/vae_checkpoint_best.pt") on HPC
```

## Adding Checkpoints to Registry

Edit `config/checkpoint_registry.yaml`:

```yaml
checkpoints:
  my_new_checkpoint:
    local: checkpoints/my_checkpoint.pt
    hpc: ${IMGINAV_ROOT}/checkpoints/my_checkpoint.pt
    description: "Description of what this checkpoint is"
```

## Available Checkpoints

List all available checkpoints:

```python
from common.checkpoint_registry import list_checkpoints

checkpoints = list_checkpoints()
for name, info in checkpoints.items():
    print(f"{name}: {info.get('description', 'No description')}")
```

## Environment Variables in Paths

You can use environment variables in registry paths:

```yaml
checkpoints:
  my_checkpoint:
    local: checkpoints/my_checkpoint.pt
    hpc: ${IMGINAV_ROOT}/checkpoints/my_checkpoint.pt
```

Common variables:
- `${IMGINAV_ROOT}`: Project root directory
- Any other environment variable: `${VAR_NAME}`

## Benefits

1. **Portability**: Same configs work on local and HPC without modification
2. **Maintainability**: Update checkpoint paths in one place (registry)
3. **Clarity**: Checkpoint names are more readable than long paths
4. **Flexibility**: Easy to switch between environments

## Example: Updating an Experiment Config

**Before** (hardcoded paths):
```yaml
autoencoder:
  checkpoint: checkpoints/vae_checkpoint_best.pt

embedding_projection:
  type: CLIPEmbeddingToSpatial
  clip_projections: checkpoints/clip_projection_checkpoint_best.pt
```

**After** (using registry):
```yaml
autoencoder:
  checkpoint: "@vae_best"

embedding_projection:
  type: CLIPEmbeddingToSpatial
  clip_projections: "@clip_projection_best"
```

The config now works on both local and HPC environments automatically!

## Troubleshooting

### Checkpoint not found

If you get an error about a checkpoint not being in the registry:
1. Check that the checkpoint name exists in `config/checkpoint_registry.yaml`
2. Verify the name matches exactly (case-sensitive)
3. Make sure you're using `@checkpoint_name` format

### Path doesn't exist warning

If you see a warning that a checkpoint path doesn't exist:
- This is just a warning - the path will be resolved correctly
- The checkpoint may be created later during training
- Verify the path is correct in the registry

### Environment not detected correctly

To force a specific environment:
```bash
export IMGINAV_ENV=local  # or hpc
```

Or in Python:
```python
from common.checkpoint_registry import get_checkpoint_path

# Override environment
path = get_checkpoint_path("@vae_best", environment="hpc")
```

