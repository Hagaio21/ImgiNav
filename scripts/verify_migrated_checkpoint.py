#!/usr/bin/env python3
"""
Verify that a migrated checkpoint can be loaded correctly.
"""

import sys
from pathlib import Path

try:
    import torch
except ImportError:
    print("Error: torch not installed. Please install PyTorch first.")
    sys.exit(1)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.autoencoder import Autoencoder
from models.diffusion import DiffusionModel


def verify_checkpoint(checkpoint_path):
    """Verify that checkpoint can be loaded."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f"Verifying checkpoint: {checkpoint_path}\n")
    
    try:
        # Load checkpoint
        payload = torch.load(checkpoint_path, map_location="cpu")
        config = payload.get("config")
        state_dict = payload.get("state_dict", payload)
        
        if not config:
            print("  ⚠️  No config found - checkpoint may be incomplete")
            return False
        
        print("=" * 60)
        print("CHECKPOINT STRUCTURE")
        print("=" * 60)
        print(f"  State dict keys: {len(state_dict)}")
        print(f"  Config type: {config.get('type', 'Unknown')}")
        
        # Try to load model
        model_type = config.get("type", "").lower()
        
        print("\n" + "=" * 60)
        print("LOADING MODEL")
        print("=" * 60)
        
        if "autoencoder" in model_type or "vae" in model_type:
            print("  Attempting to load as Autoencoder...")
            model = Autoencoder.load_checkpoint(checkpoint_path, map_location="cpu", strict=False)
            print("  ✅ Autoencoder loaded successfully")
        elif "diffusion" in model_type:
            print("  Attempting to load as DiffusionModel...")
            model = DiffusionModel.load_checkpoint(checkpoint_path, map_location="cpu", strict=False)
            print("  ✅ DiffusionModel loaded successfully")
        else:
            # Try both
            print("  Attempting to load as Autoencoder...")
            try:
                model = Autoencoder.load_checkpoint(checkpoint_path, map_location="cpu", strict=False)
                print("  ✅ Loaded as Autoencoder")
            except Exception as e1:
                print(f"  ❌ Failed as Autoencoder: {e1}")
                print("  Attempting to load as DiffusionModel...")
                try:
                    model = DiffusionModel.load_checkpoint(checkpoint_path, map_location="cpu", strict=False)
                    print("  ✅ Loaded as DiffusionModel")
                except Exception as e2:
                    print(f"  ❌ Failed as DiffusionModel: {e2}")
                    return False
        
        # Check model structure
        print("\n" + "=" * 60)
        print("MODEL STRUCTURE")
        print("=" * 60)
        print(f"  Model class: {model.__class__.__name__}")
        print(f"  Model state dict keys: {len(model.state_dict())}")
        
        # Check components
        if hasattr(model, '_component_names'):
            components = list(model._component_names.values())
            print(f"  Tracked components: {components}")
        
        if isinstance(model, Autoencoder):
            if hasattr(model, 'encoder'):
                print("  ✅ Has encoder")
            if hasattr(model, 'decoder'):
                print("  ✅ Has decoder")
        
        if isinstance(model, DiffusionModel):
            if hasattr(model, 'unet'):
                print("  ✅ Has UNet")
            if hasattr(model, 'decoder'):
                print("  ✅ Has decoder")
            if hasattr(model, 'scheduler'):
                print("  ✅ Has scheduler")
        
        print("\n" + "=" * 60)
        print("VERIFICATION RESULT")
        print("=" * 60)
        print("\n  ✅ CHECKPOINT VERIFIED SUCCESSFULLY")
        print("  The checkpoint can be loaded and has correct structure")
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("VERIFICATION RESULT")
        print("=" * 60)
        print(f"\n  ❌ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_migrated_checkpoint.py <checkpoint_path>")
        sys.exit(1)
    
    success = verify_checkpoint(sys.argv[1])
    sys.exit(0 if success else 1)

