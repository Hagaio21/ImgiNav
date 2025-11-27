#!/usr/bin/env python3
"""
Test checkpoint registry system.
"""

import sys
from pathlib import Path
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.checkpoint_registry import (
    get_checkpoint_path,
    get_checkpoint,
    list_checkpoints,
    detect_environment,
    resolve_checkpoint_in_config
)


def test_registry_basic():
    """Test basic registry functionality."""
    print("="*60)
    print("Testing Checkpoint Registry")
    print("="*60)
    
    # Test environment detection
    env = detect_environment()
    print(f"\n1. Environment Detection: {env}")
    
    # List checkpoints
    print("\n2. Available Checkpoints:")
    checkpoints = list_checkpoints()
    for name, info in checkpoints.items():
        desc = info.get("description", "No description")
        print(f"   - {name}: {desc}")
    
    # Test resolving checkpoint paths
    print("\n3. Resolving Checkpoint Paths:")
    test_checkpoints = ["vae_best", "clip_projection_best", "diffusion_small_text_only"]
    
    for name in test_checkpoints:
        try:
            path = get_checkpoint(f"@{name}")
            exists = path.exists()
            status = "✓" if exists else "⚠"
            print(f"   {status} @{name} → {path}")
            if not exists:
                print(f"      (Path does not exist - may be created later)")
        except ValueError as e:
            print(f"   ✗ @{name}: {e}")
    
    # Test direct path (should work as-is)
    print("\n4. Direct Path (no registry):")
    direct_path = get_checkpoint("checkpoints/vae_checkpoint_best.pt")
    print(f"   {direct_path}")
    
    # Test config resolution
    print("\n5. Config Resolution:")
    test_config = {
        "autoencoder": {
            "checkpoint": "@vae_best"
        },
        "embedding_projection": {
            "clip_projections": "@clip_projection_best"
        },
        "some_other_path": "checkpoints/something.pt"  # Should not be changed
    }
    
    resolved = resolve_checkpoint_in_config(test_config)
    print(f"   Original: {test_config['autoencoder']['checkpoint']}")
    print(f"   Resolved: {resolved['autoencoder']['checkpoint']}")
    print(f"   Other path unchanged: {resolved['some_other_path']}")
    
    print("\n" + "="*60)
    print("Registry Test Complete!")
    print("="*60)


def test_config_loading():
    """Test config loading with registry resolution."""
    print("\n" + "="*60)
    print("Testing Config Loading with Registry")
    print("="*60)
    
    # Create a test config file
    test_config_path = Path("config/test_registry_config.yaml")
    test_config = {
        "type": "DiffusionModel",
        "autoencoder": {
            "checkpoint": "@vae_best"
        },
        "embedding_projection": {
            "type": "CLIPEmbeddingToSpatial",
            "clip_projections": "@clip_projection_best"
        }
    }
    
    # Save test config
    test_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(test_config_path, "w") as f:
        yaml.dump(test_config, f)
    
    print(f"\nCreated test config: {test_config_path}")
    
    # Load config (should auto-resolve)
    from common.utils import load_config_with_profile
    loaded = load_config_with_profile(str(test_config_path))
    
    print("\nLoaded config:")
    print(f"  autoencoder.checkpoint: {loaded.get('autoencoder', {}).get('checkpoint')}")
    print(f"  embedding_projection.clip_projections: {loaded.get('embedding_projection', {}).get('clip_projections')}")
    
    # Check if resolved
    autoencoder_checkpoint = loaded.get('autoencoder', {}).get('checkpoint', '')
    if autoencoder_checkpoint.startswith('@'):
        print("\n  ⚠ Registry references not resolved!")
    else:
        print("\n  ✓ Registry references resolved!")
    
    # Cleanup
    if test_config_path.exists():
        test_config_path.unlink()
        print(f"\nCleaned up test config: {test_config_path}")


if __name__ == "__main__":
    try:
        test_registry_basic()
        test_config_loading()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

