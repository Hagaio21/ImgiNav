#!/usr/bin/env python3
"""
Simple test script for train_diffusion.py.
Just verifies imports and basic functionality without running training or saving files.
"""

import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.diffusion import DiffusionModel
from models.autoencoder import Autoencoder
from models.components.discriminator import LatentDiscriminator
from models.losses import LOSS_REGISTRY
from training.utils import build_loss, build_optimizer, build_scheduler


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    assert DiffusionModel is not None
    assert Autoencoder is not None
    assert LatentDiscriminator is not None
    assert LOSS_REGISTRY is not None
    print("  ✓ All imports successful")


def test_loss_registry():
    """Test that loss components can be created."""
    print("Testing loss registry...")
    
    # Test SNRWeightedNoiseLoss
    noise_loss = LOSS_REGISTRY["SNRWeightedNoiseLoss"](key="pred_noise", target="noise", weight=1.0)
    assert noise_loss is not None
    print("  ✓ SNRWeightedNoiseLoss created")
    
    # Test SemanticLoss
    semantic_loss = LOSS_REGISTRY["SemanticLoss"](
        weight=0.1,
        segmentation_loss={"type": "CrossEntropyLoss", "weight": 1.0},
        perceptual_loss={"type": "PerceptualLoss", "weight": 0.5}
    )
    assert semantic_loss is not None
    print("  ✓ SemanticLoss created")
    
    # Test DiscriminatorLoss
    disc_loss = LOSS_REGISTRY["DiscriminatorLoss"](key="latent", weight=0.1)
    assert disc_loss is not None
    print("  ✓ DiscriminatorLoss created")
    
    # Test CompositeLoss (with config dicts, not objects)
    composite = LOSS_REGISTRY["CompositeLoss"](
        losses=[
            {"type": "SNRWeightedNoiseLoss", "key": "pred_noise", "target": "noise", "weight": 1.0},
            {"type": "SemanticLoss", "weight": 0.1, "segmentation_loss": {"type": "CrossEntropyLoss", "weight": 1.0}, "perceptual_loss": {"type": "PerceptualLoss", "weight": 0.5}}
        ]
    )
    assert composite is not None
    print("  ✓ CompositeLoss created")


def test_model_creation():
    """Test that models can be created (without saving)."""
    print("Testing model creation...")
    
    # Create minimal autoencoder config
    encoder_cfg = {
        "type": "Encoder",
        "in_channels": 3,
        "latent_channels": 16,
        "base_channels": 16,
        "downsampling_steps": 2,
        "variational": False
    }
    
    decoder_cfg = {
        "type": "Decoder",
        "latent_channels": 16,
        "base_channels": 16,
        "upsampling_steps": 2,
        "heads": [
            {"type": "RGBHead", "out_channels": 3},
            {"type": "SegmentationHead", "out_channels": 2}
        ]
    }
    
    # Create autoencoder (in memory only)
    autoencoder = Autoencoder(encoder=encoder_cfg, decoder=decoder_cfg)
    assert autoencoder is not None
    print("  ✓ Autoencoder created")
    
    # Create diffusion model (in memory only)
    diffusion_cfg = {
        "autoencoder": {"checkpoint": None, "frozen": True},
        "unet": {"type": "Unet", "in_channels": 16, "base_channels": 32, "downsample_steps": 2, "num_resblocks": 1},
        "scheduler": {"type": "CosineScheduler", "num_steps": 100}
    }
    # Can't create without checkpoint, so skip this
    print("  ✓ Model creation test skipped (requires checkpoint)")
    
    # Create discriminator (in memory only)
    discriminator = LatentDiscriminator(latent_channels=16, base_channels=64, num_layers=4)
    assert discriminator is not None
    print("  ✓ Discriminator created")


def test_loss_computation():
    """Test that losses can compute (with fake tensors, no saving)."""
    print("Testing loss computation...")
    
    # Create fake tensors (match actual system: 32x32 with 16 channels)
    batch_size = 2
    pred_noise = torch.randn(batch_size, 16, 32, 32)
    target_noise = torch.randn(batch_size, 16, 32, 32)
    
    # Test SNRWeightedNoiseLoss
    noise_loss = LOSS_REGISTRY["SNRWeightedNoiseLoss"](key="pred_noise", target="noise", weight=1.0)
    preds = {"pred_noise": pred_noise, "scheduler": None, "timesteps": None}
    targets = {"noise": target_noise}
    loss_val, logs = noise_loss(preds, targets)
    assert isinstance(loss_val, torch.Tensor)
    assert loss_val.item() >= 0
    print("  ✓ SNRWeightedNoiseLoss computation works")
    
    # Test DiscriminatorLoss (match actual system: 32x32 with 16 channels)
    disc_loss = LOSS_REGISTRY["DiscriminatorLoss"](key="latent", weight=0.1)
    latents = torch.randn(batch_size, 16, 32, 32)
    discriminator = LatentDiscriminator(latent_channels=16, base_channels=64, num_layers=4)
    preds = {"latent": latents, "discriminator": discriminator}
    targets = {}
    loss_val, logs = disc_loss(preds, targets)
    assert isinstance(loss_val, torch.Tensor)
    assert loss_val.item() >= 0  # Loss should be non-negative
    print("  ✓ DiscriminatorLoss computation works")


def main():
    """Run all tests (no file I/O)."""
    print("="*60)
    print("Testing train_diffusion.py components")
    print("="*60)
    
    try:
        test_imports()
        test_loss_registry()
        test_model_creation()
        test_loss_computation()
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
    except Exception as e:
        print("\n" + "="*60)
        print(f"Tests failed: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
