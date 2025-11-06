# Models Module

This module contains all neural network model definitions, including autoencoders, diffusion models, and their components.

## Overview

The `models` module is organized into several submodules:
- **Core Models**: Autoencoder, Diffusion, ControlNet
- **Components**: UNet, Encoder, Decoder, Heads, Schedulers, Blocks
- **Datasets**: Data loading and preprocessing
- **Losses**: Loss function implementations

## Directory Structure

```
models/
├── autoencoder.py          # Main autoencoder model
├── encoder.py              # Encoder network
├── decoder.py              # Decoder network
├── diffusion.py            # Diffusion model
├── controlnet_diffusion.py # ControlNet diffusion model
├── components/             # Reusable model components
│   ├── unet.py            # UNet architecture
│   ├── blocks.py          # Building blocks (ResNet, Attention, etc.)
│   ├── heads.py           # Output heads (RGB, Segmentation, Classification)
│   ├── scheduler.py       # Noise schedulers (Linear, Cosine, Quadratic)
│   ├── controlnet.py      # ControlNet adapter
│   ├── fusion.py          # Feature fusion mechanisms
│   └── base_model.py      # Base model class
├── datasets/               # Dataset implementations
│   ├── datasets.py        # Main dataset classes
│   └── collate.py         # Custom collate functions
└── losses/                # Loss functions
    ├── base_loss.py       # Base loss class
    └── loss_utils.py      # Loss utilities
```

## Core Models

### Autoencoder (`autoencoder.py`)

Main autoencoder model combining encoder and decoder.

**Key Features:**
- Supports both deterministic and variational (VAE) encoders
- Multi-head decoder (RGB, Segmentation, Classification)
- Configurable architecture via YAML configs
- Checkpoint loading/saving

**Usage:**
```python
from models.autoencoder import Autoencoder

# Build from config
model = Autoencoder.from_config(config["autoencoder"])

# Forward pass
output = model(x)  # Returns dict with all head outputs

# Encode only
latent = model.encode(x)  # Returns {"latent": z} or {"mu": mu, "logvar": logvar}

# Decode only
reconstruction = model.decode(latent)
```

### Encoder (`encoder.py`)

Encoder network that compresses input images to latent representations.

**Architecture:**
- Convolutional downsampling blocks
- Optional variational output (VAE)
- Group normalization
- SiLU activation

**Key Parameters:**
- `in_channels`: Input channels (typically 3 for RGB)
- `latent_channels`: Latent space channels
- `base_channels`: Base channel count
- `downsampling_steps`: Number of downsampling operations
- `variational`: Whether to use VAE (outputs mu, logvar)

### Decoder (`decoder.py`)

Decoder network that reconstructs images from latent representations.

**Architecture:**
- Convolutional upsampling blocks
- Multiple output heads (RGB, Segmentation, Classification)
- Group normalization
- SiLU activation

**Key Parameters:**
- `latent_channels`: Latent space channels
- `base_channels`: Base channel count
- `upsampling_steps`: Number of upsampling operations
- `heads`: List of output head configurations

### Diffusion Model (`diffusion.py`)

Diffusion model for generating latents, with optional decoder for image reconstruction.

**Key Features:**
- UNet-based noise prediction
- Multiple noise schedulers (Linear, Cosine, Quadratic)
- Pre-embedded latent support (faster training)
- Frozen decoder support (uses pre-trained autoencoder)

**Usage:**
```python
from models.diffusion import DiffusionModel

# Build from config
model = DiffusionModel.from_config(config)

# Training forward pass
pred_noise = model(latent, timestep)  # Predict noise at timestep

# Sampling
generated_latent = model.sample(batch_size, device)  # Generate from noise
image = model.decode_to_image(generated_latent)  # Decode to image
```

### ControlNet Diffusion (`controlnet_diffusion.py`)

ControlNet-based conditional diffusion model for guided generation.

**Key Features:**
- Conditional generation based on control signals
- ControlNet adapter for feature injection
- Multiple fusion modes

## Components

### UNet (`components/unet.py`)

U-Net architecture for diffusion models.

**Key Parameters:**
- `in_channels`: Input channels (matches latent_channels)
- `out_channels`: Output channels
- `base_channels`: Base channel count (32, 48, 64, 128, 256)
- `depth`: Network depth (3, 4, 5)
- `num_res_blocks`: Residual blocks per level
- `time_dim`: Time embedding dimension
- `cond_channels`: Conditioning channels (0 for unconditional)

### Blocks (`components/blocks.py`)

Reusable building blocks:
- `ResNetBlock`: Residual block with group normalization
- `AttentionBlock`: Self-attention mechanism
- `DownsampleBlock`: Downsampling convolution
- `UpsampleBlock`: Upsampling convolution

### Heads (`components/heads.py`)

Output heads for multi-task learning:
- `RGBHead`: RGB reconstruction head
- `SegmentationHead`: Semantic segmentation head
- `ClassificationHead`: Classification head

### Schedulers (`components/scheduler.py`)

Noise schedulers for diffusion:
- `LinearScheduler`: Linear noise schedule
- `CosineScheduler`: Cosine noise schedule (recommended)
- `QuadraticScheduler`: Quadratic noise schedule

**Usage:**
```python
from models.components.scheduler import SCHEDULER_REGISTRY

scheduler = SCHEDULER_REGISTRY["CosineScheduler"](num_steps=500)
noise = scheduler.add_noise(latent, timestep)
```

## Datasets

### ManifestDataset (`datasets/datasets.py`)

Main dataset class for loading data from CSV manifests.

**Features:**
- CSV manifest-based data loading
- Multiple output types (RGB, segmentation, labels)
- Filtering support (e.g., non-empty samples)
- Pre-embedded latent support
- Weighted sampling for class balancing

**Manifest Format:**
CSV with columns like:
- `layout_path`: Path to layout image
- `latent_path`: Path to pre-embedded latent (optional)
- `type`: Classification label
- `is_empty`: Boolean flag

## Losses

### Base Loss (`losses/base_loss.py`)

Base class for all loss functions with configurable weighting.

**Available Losses:**
- `MSELoss`: Mean squared error
- `KLDLoss`: KL divergence (for VAE)
- `CrossEntropyLoss`: Cross-entropy for classification/segmentation
- `LatentStandardizationLoss`: Encourages latents ~N(0,1)
- `CompositeLoss`: Combines multiple losses

**Usage:**
```python
from models.losses.base_loss import MSELoss, CompositeLoss

# Single loss
loss_fn = MSELoss(key="rgb", target="rgb", weight=1.0)

# Composite loss
loss_fn = CompositeLoss(losses=[
    MSELoss(key="rgb", target="rgb", weight=1.0),
    CrossEntropyLoss(key="segmentation", target="segmentation", weight=0.05)
])
```

## Configuration

Models are configured via YAML files. See `experiments/` directory for examples.

**Example Autoencoder Config:**
```yaml
autoencoder:
  encoder:
    in_channels: 3
    latent_channels: 16
    base_channels: 32
    downsampling_steps: 4
    variational: false
  decoder:
    latent_channels: 16
    base_channels: 32
    upsampling_steps: 4
    heads:
      - type: RGBHead
        name: rgb
        out_channels: 3
```

## Notes

- All models inherit from `BaseModel` for consistent checkpoint handling
- Models support both config-based and checkpoint-based initialization
- Frozen components (e.g., frozen decoder in diffusion) are supported
- Mixed precision training (AMP) is supported

