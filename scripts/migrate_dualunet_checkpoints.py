#!/usr/bin/env python3
"""
Migration script to convert DualUNet checkpoints to Unet format.

This script:
1. Loads checkpoints that reference DualUNet
2. Converts the UNet component to the new Unet format (removes conditioning logic)
3. Saves the checkpoint with updated config and class name

Usage:
    python scripts/migrate_dualunet_checkpoints.py \
        --checkpoint path/to/checkpoint.pt \
        --output path/to/output.pt
    
    Or process directory:
    python scripts/migrate_dualunet_checkpoints.py \
        --checkpoint-dir path/to/checkpoints \
        --output-dir path/to/migrated \
        --pattern "*checkpoint*.pt"
"""

import argparse
import sys
from pathlib import Path
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.diffusion import DiffusionModel
from models.components.unet import Unet, DualUNet  # DualUNet is backward-compatible alias


def migrate_checkpoint(checkpoint_path, output_path, dry_run=False):
    """
    Migrate a single checkpoint from DualUNet to Unet format.
    
    Args:
        checkpoint_path: Path to input checkpoint
        output_path: Path to save migrated checkpoint
        dry_run: If True, only validate without saving
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Migrating: {checkpoint_path}")
    print(f"{'='*60}")
    
    try:
        # Load checkpoint
        print("Loading checkpoint...")
        payload = torch.load(checkpoint_path, map_location="cpu")
        
        # Get config
        config = payload.get("config")
        if config is None:
            print("  Warning: No config found in checkpoint, skipping...")
            return False
        
        # Check if UNet config needs migration
        unet_config = None
        if "unet" in config:
            unet_config = config["unet"]
        elif "model" in config and "unet" in config["model"]:
            unet_config = config["model"]["unet"]
        
        if unet_config is None:
            print("  Warning: No UNet config found, skipping...")
            return False
        
        # Check if already migrated
        unet_type = unet_config.get("type", "")
        if unet_type == "Unet":
            print("  Already migrated (type: Unet)")
            return True
        
        if unet_type not in ("DualUNet", "dualunet", ""):
            print(f"  Warning: Unknown UNet type '{unet_type}', proceeding anyway...")
        
        # Verify conditioning is not used
        cond_channels = unet_config.get("cond_channels", 0)
        fusion_mode = unet_config.get("fusion_mode", "none")
        if cond_channels > 0 or fusion_mode not in ("none", None):
            print(f"  WARNING: Conditioning is enabled (cond_channels={cond_channels}, fusion_mode={fusion_mode})")
            print(f"  This checkpoint uses conditioning features that will be removed!")
            response = input("  Continue anyway? (yes/no): ")
            if response.lower() != "yes":
                print("  Migration cancelled.")
                return False
        
        # Update UNet config: remove conditioning-related fields
        print("  Updating UNet config...")
        new_unet_config = unet_config.copy()
        new_unet_config["type"] = "Unet"
        # Remove conditioning-related fields (they're not used anyway)
        new_unet_config.pop("cond_channels", None)
        new_unet_config.pop("fusion_mode", None)
        new_unet_config.pop("cond_mult", None)
        
        # Update config
        if "unet" in config:
            config["unet"] = new_unet_config
        elif "model" in config:
            config["model"]["unet"] = new_unet_config
        
        # Update payload
        payload["config"] = config
        
        # Load model to verify and get state dict
        print("  Loading model to verify compatibility...")
        try:
            # DualUNet is now an alias for Unet, so it can load old checkpoints
            # Load model - it will use DualUNet class (backward compatible)
            model = DiffusionModel.load_checkpoint(
                checkpoint_path,
                map_location="cpu",
                config=None  # Use saved config
            )
            
            # Get old UNet state dict (may have conditioning-related keys)
            old_unet = model.unet
            old_state_dict = old_unet.state_dict()
            
            # Create new UNet with same config (minus conditioning)
            new_unet = Unet.from_config(new_unet_config)
            new_state_dict = new_unet.state_dict()
            
            # Transfer weights (only matching keys - conditioning keys will be skipped)
            matched_keys = []
            skipped_keys = []
            for key in new_state_dict.keys():
                if key in old_state_dict:
                    if old_state_dict[key].shape == new_state_dict[key].shape:
                        new_state_dict[key] = old_state_dict[key]
                        matched_keys.append(key)
                    else:
                        skipped_keys.append(f"{key} (shape mismatch)")
                else:
                    skipped_keys.append(f"{key} (not found)")
            
            print(f"  Matched {len(matched_keys)}/{len(new_state_dict)} parameters")
            if skipped_keys:
                print(f"  Skipped {len(skipped_keys)} parameters (conditioning-related)")
            
            # Update model's UNet
            model.unet = new_unet
            model.unet.load_state_dict(new_state_dict, strict=False)
            
            # Update payload with new state dict
            payload["state_dict"] = model.state_dict()
            
        except Exception as e:
            print(f"  Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        if dry_run:
            print("  [DRY RUN] Would save to:", output_path)
            return True
        
        # Save migrated checkpoint
        print(f"  Saving migrated checkpoint to: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_path)
        
        print("  ✓ Migration successful!")
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Migrate DualUNet checkpoints to Unet format"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint file to migrate"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Directory containing checkpoints to migrate"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for migrated checkpoint (required if --checkpoint used)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for migrated checkpoints (required if --checkpoint-dir used)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*checkpoint*.pt",
        help="Glob pattern for finding checkpoints in directory (default: '*checkpoint*.pt')"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate checkpoints without saving"
    )
    
    args = parser.parse_args()
    
    if args.checkpoint:
        if not args.output:
            print("Error: --output required when using --checkpoint")
            return
        migrate_checkpoint(args.checkpoint, args.output, dry_run=args.dry_run)
    elif args.checkpoint_dir:
        if not args.output_dir:
            print("Error: --output-dir required when using --checkpoint-dir")
            return
        
        checkpoint_dir = Path(args.checkpoint_dir)
        output_dir = Path(args.output_dir)
        
        checkpoints = list(checkpoint_dir.glob(args.pattern))
        print(f"Found {len(checkpoints)} checkpoints to migrate")
        
        success_count = 0
        for checkpoint_path in checkpoints:
            # Preserve directory structure
            rel_path = checkpoint_path.relative_to(checkpoint_dir)
            output_path = output_dir / rel_path
            
            if migrate_checkpoint(checkpoint_path, output_path, dry_run=args.dry_run):
                success_count += 1
        
        print(f"\n{'='*60}")
        print(f"Migration complete: {success_count}/{len(checkpoints)} successful")
        print(f"{'='*60}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

