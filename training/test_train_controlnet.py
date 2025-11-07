#!/usr/bin/env python3
"""
Simple test script for train_controlnet.py.
Just verifies imports and basic functionality without running training or saving files.
"""

import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.controlnet_diffusion import ControlNetDiffusionModel
from models.components.controlnet import ControlNet
from models.components.control_adapter import SimpleAdapter
from models.diffusion import DiffusionModel
from models.autoencoder import Autoencoder


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    assert ControlNetDiffusionModel is not None
    assert ControlNet is not None
    assert SimpleAdapter is not None
    assert DiffusionModel is not None
    assert Autoencoder is not None
    print("  ✓ All imports successful")


def test_controlnet_adapter_creation():
    """Test that ControlNet adapter can be created (without saving)."""
    print("Testing ControlNet adapter creation...")
    
    # Create adapter (in memory only)
    adapter = SimpleAdapter(
        text_dim=768,
        pov_dim=256,
        base_channels=32,
        depth=2
    )
    assert adapter is not None
    print("  ✓ ControlNet adapter created")
    
    # Test forward pass with fake inputs
    batch_size = 2
    text_emb = torch.randn(batch_size, 768)
    pov_emb = torch.randn(batch_size, 256, 8, 8)
    
    output = adapter(text_emb, pov_emb)
    assert isinstance(output, list)  # Adapter returns list of features
    assert len(output) > 0
    print("  ✓ ControlNet adapter forward pass works")


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
    
    # ControlNet model creation requires checkpoint, so skip
    print("  ✓ Model creation test skipped (requires checkpoint)")


def main():
    """Run all tests (no file I/O)."""
    print("="*60)
    print("Testing train_controlnet.py components")
    print("="*60)
    
    try:
        test_imports()
        test_controlnet_adapter_creation()
        test_model_creation()
        
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
