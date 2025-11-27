#!/usr/bin/env python3
"""
Quick script to inspect checkpoint structure without running full migration.
"""

import sys
from pathlib import Path

try:
    import torch
except ImportError:
    print("Error: torch not installed. Please install PyTorch first.")
    print("You can install it with: pip install torch")
    sys.exit(1)


def inspect_checkpoint(checkpoint_path):
    """Inspect checkpoint structure."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"Inspecting checkpoint: {checkpoint_path}\n")
    
    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Check structure
    state_dict = payload.get("state_dict", payload)
    config = payload.get("config")
    
    print("=" * 60)
    print("CHECKPOINT STRUCTURE")
    print("=" * 60)
    
    # Count keys
    total_keys = len(state_dict)
    clip_proj_keys = [k for k in state_dict.keys() if "clip_projections" in k]
    
    print(f"\nTotal state dict keys: {total_keys}")
    print(f"CLIP projection keys: {len(clip_proj_keys)}")
    
    # Check for old vs new format
    old_format_keys = []
    new_format_keys = []
    other_clip_keys = []
    
    for key in clip_proj_keys:
        if any(proj in key for proj in ["text_proj.", "pov_proj.", "latent_proj."]):
            if ".proj." in key:
                new_format_keys.append(key)
            else:
                # Check if it's old format (has numbers directly after proj name)
                import re
                if re.search(r'(text|pov|latent)_proj\.\d+', key):
                    old_format_keys.append(key)
                else:
                    other_clip_keys.append(key)
        else:
            other_clip_keys.append(key)
    
    print(f"\nOld format keys (need migration): {len(old_format_keys)}")
    print(f"New format keys (already migrated): {len(new_format_keys)}")
    print(f"Other CLIP keys: {len(other_clip_keys)}")
    
    if old_format_keys:
        print("\nSample old format keys (first 5):")
        for key in old_format_keys[:5]:
            print(f"  {key}")
        if len(old_format_keys) > 5:
            print(f"  ... and {len(old_format_keys) - 5} more")
    
    if new_format_keys:
        print("\nSample new format keys (first 5):")
        for key in new_format_keys[:5]:
            print(f"  {key}")
        if len(new_format_keys) > 5:
            print(f"  ... and {len(new_format_keys) - 5} more")
    
    # Check config
    print("\n" + "=" * 60)
    print("CONFIG STRUCTURE")
    print("=" * 60)
    
    if config:
        if isinstance(config, dict):
            if "clip_projection" in config:
                clip_cfg = config["clip_projection"]
                print("\nCLIP projection config found:")
                print(f"  Keys: {list(clip_cfg.keys())}")
                
                has_text_proj = "text_projection" in clip_cfg
                has_image_proj = "image_projection" in clip_cfg
                has_latent_proj = "latent_projection" in clip_cfg
                
                print(f"\n  Has text_projection: {has_text_proj}")
                print(f"  Has image_projection: {has_image_proj}")
                print(f"  Has latent_projection: {has_latent_proj}")
                
                if not (has_text_proj and has_image_proj):
                    print("\n  WARNING: Config needs migration (missing projection configs)")
                else:
                    print("\n  [OK] Config structure looks good")
            else:
                print("\nNo CLIP projection config found")
        else:
            print(f"\nConfig is not a dict: {type(config)}")
    else:
        print("\nNo config found in checkpoint")
    
    # Summary
    print("\n" + "=" * 60)
    print("MIGRATION STATUS")
    print("=" * 60)
    
    needs_migration = len(old_format_keys) > 0 or (config and isinstance(config, dict) and 
                                                    "clip_projection" in config and 
                                                    ("text_projection" not in config["clip_projection"] or 
                                                     "image_projection" not in config["clip_projection"]))
    
    if needs_migration:
        print("\nWARNING: CHECKPOINT NEEDS MIGRATION")
        print("\nTo migrate, run:")
        print(f'  python scripts/migrate_projections.py "{checkpoint_path}" --output "migrated_{checkpoint_path.name}"')
    else:
        print("\n[OK] CHECKPOINT IS ALREADY IN NEW FORMAT")
        print("  No migration needed!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint.py <checkpoint_path>")
        sys.exit(1)
    
    inspect_checkpoint(sys.argv[1])

