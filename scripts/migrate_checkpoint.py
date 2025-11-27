#!/usr/bin/env python3
"""
Migrate old checkpoints to new architecture.

Handles:
- Projection component key changes (if applicable)
- Preserves all extra state (optimizer, epoch, etc.)
- Verifies migrated checkpoint can be loaded
"""

import sys
import argparse
import shutil
from pathlib import Path
import re

try:
    import torch
except ImportError:
    print("Error: torch not installed. Please install PyTorch first.")
    sys.exit(1)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.autoencoder import Autoencoder
from models.diffusion import DiffusionModel


def inspect_checkpoint_structure(checkpoint_path):
    """Inspect checkpoint structure and identify migration needs."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload.get("state_dict", payload)
    config = payload.get("config")
    
    # Identify old format projection keys
    old_format_keys = []
    new_format_keys = []
    other_keys = []
    
    for key in state_dict.keys():
        if "clip_projections" in key:
            if any(proj in key for proj in ["text_proj.", "pov_proj.", "latent_proj."]):
                if ".proj." in key:
                    new_format_keys.append(key)
                elif re.search(r'(text|pov|latent)_proj\.\d+', key):
                    old_format_keys.append(key)
                else:
                    other_keys.append(key)
            else:
                other_keys.append(key)
        else:
            other_keys.append(key)
    
    return {
        "payload": payload,
        "state_dict": state_dict,
        "config": config,
        "old_format_keys": old_format_keys,
        "new_format_keys": new_format_keys,
        "other_keys": other_keys,
        "extra_state": {k: v for k, v in payload.items() if k not in ["state_dict", "config"]}
    }


def migrate_projection_keys(state_dict, old_format_keys):
    """Migrate old format projection keys to new format."""
    migrated_state_dict = state_dict.copy()
    key_mappings = {}
    
    for old_key in old_format_keys:
        # Pattern: clip_projections.text_proj.0.weight -> clip_projections.text_proj.proj.0.weight
        # Pattern: clip_projections.pov_proj.0.weight -> clip_projections.pov_proj.proj.0.weight
        # Pattern: clip_projections.latent_proj.0.weight -> clip_projections.latent_proj.proj.0.weight
        
        # Find the projection type and layer number
        match = re.match(r'(clip_projections\.(?:text|pov|latent)_proj)\.(\d+)(.*)', old_key)
        if match:
            prefix = match.group(1)
            layer_num = match.group(2)
            suffix = match.group(3)
            
            # Create new key with .proj. inserted
            new_key = f"{prefix}.proj.{layer_num}{suffix}"
            key_mappings[old_key] = new_key
    
    # Apply migrations
    for old_key, new_key in key_mappings.items():
        if old_key in migrated_state_dict:
            migrated_state_dict[new_key] = migrated_state_dict.pop(old_key)
    
    return migrated_state_dict, key_mappings


def migrate_checkpoint(checkpoint_path, output_path=None, dry_run=False, verify=True):
    """
    Migrate checkpoint to new architecture.
    
    Args:
        checkpoint_path: Path to old checkpoint
        output_path: Path to save migrated checkpoint (if None, creates migrated_<name>)
        dry_run: If True, only show what would be migrated
        verify: If True, verify migrated checkpoint can be loaded
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Inspecting checkpoint: {checkpoint_path}\n")
    
    # Inspect checkpoint
    inspection = inspect_checkpoint_structure(checkpoint_path)
    
    old_format_keys = inspection["old_format_keys"]
    config = inspection["config"]
    extra_state = inspection["extra_state"]
    
    print("=" * 60)
    print("CHECKPOINT INSPECTION")
    print("=" * 60)
    print(f"Total state dict keys: {len(inspection['state_dict'])}")
    print(f"Old format projection keys: {len(old_format_keys)}")
    print(f"New format projection keys: {len(inspection['new_format_keys'])}")
    print(f"Extra state keys: {list(extra_state.keys())}")
    
    if config:
        model_type = config.get("type", "Unknown")
        print(f"Model type: {model_type}")
    
    # Check if migration is needed
    needs_migration = len(old_format_keys) > 0
    
    if not needs_migration:
        print("\n" + "=" * 60)
        print("MIGRATION STATUS")
        print("=" * 60)
        print("\n[OK] CHECKPOINT IS ALREADY IN NEW FORMAT")
        print("  No migration needed!")
        return False
    
    print("\n" + "=" * 60)
    print("MIGRATION NEEDED")
    print("=" * 60)
    print(f"\nFound {len(old_format_keys)} keys that need migration:")
    for key in old_format_keys[:10]:
        print(f"  {key}")
    if len(old_format_keys) > 10:
        print(f"  ... and {len(old_format_keys) - 10} more")
    
    if dry_run:
        print("\n[DRY RUN] Would migrate keys:")
        for old_key in old_format_keys[:10]:
            match = re.match(r'(clip_projections\.(?:text|pov|latent)_proj)\.(\d+)(.*)', old_key)
            if match:
                prefix = match.group(1)
                layer_num = match.group(2)
                suffix = match.group(3)
                new_key = f"{prefix}.proj.{layer_num}{suffix}"
                print(f"  {old_key} -> {new_key}")
        return True
    
    # Perform migration
    print("\n" + "=" * 60)
    print("MIGRATING CHECKPOINT")
    print("=" * 60)
    
    # Migrate projection keys
    migrated_state_dict, key_mappings = migrate_projection_keys(
        inspection["state_dict"],
        old_format_keys
    )
    
    print(f"\nMigrated {len(key_mappings)} keys")
    
    # Create new payload
    migrated_payload = {
        "state_dict": migrated_state_dict,
        "config": config,
        **extra_state
    }
    
    # Determine output path
    if output_path is None:
        output_path = checkpoint_path.parent / f"migrated_{checkpoint_path.name}"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save migrated checkpoint
    print(f"\nSaving migrated checkpoint to: {output_path}")
    torch.save(migrated_payload, output_path)
    
    # Verify migrated checkpoint
    if verify:
        print("\n" + "=" * 60)
        print("VERIFYING MIGRATED CHECKPOINT")
        print("=" * 60)
        
        try:
            if config:
                model_type = config.get("type", "").lower()
                
                if "autoencoder" in model_type or "vae" in model_type:
                    model = Autoencoder.load_checkpoint(output_path, map_location="cpu", strict=False)
                    print("\n[OK] Autoencoder checkpoint loaded successfully")
                elif "diffusion" in model_type:
                    model = DiffusionModel.load_checkpoint(output_path, map_location="cpu", strict=False)
                    print("\n[OK] Diffusion checkpoint loaded successfully")
                else:
                    # Try both
                    try:
                        model = Autoencoder.load_checkpoint(output_path, map_location="cpu", strict=False)
                        print("\n[OK] Checkpoint loaded as Autoencoder")
                    except:
                        model = DiffusionModel.load_checkpoint(output_path, map_location="cpu", strict=False)
                        print("\n[OK] Checkpoint loaded as DiffusionModel")
                
                print(f"  Model type: {model.__class__.__name__}")
                print(f"  State dict keys: {len(model.state_dict())}")
                
        except Exception as e:
            print(f"\n[WARNING] Verification failed: {e}")
            print("  Checkpoint was saved but may have issues loading")
            return False
    
    print("\n" + "=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)
    print(f"\nOriginal: {checkpoint_path}")
    print(f"Migrated: {output_path}")
    print(f"\nMigrated {len(key_mappings)} projection keys")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate old checkpoints to new architecture"
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to checkpoint to migrate"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save migrated checkpoint (default: migrated_<name>)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actually migrating"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification of migrated checkpoint"
    )
    
    args = parser.parse_args()
    
    try:
        migrate_checkpoint(
            args.checkpoint,
            output_path=args.output,
            dry_run=args.dry_run,
            verify=not args.no_verify
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

