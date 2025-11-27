"""
Migration script to separate CLIP projections from diffusion checkpoint.

This script:
1. Loads the old diffusion checkpoint
2. Extracts CLIP projections from embedding_projection.clip_projections
3. Extracts spatial projection from embedding_projection.spatial_proj
4. Saves CLIP projections separately
5. Reassembles diffusion checkpoint with updated config pointing to external CLIP projections
"""
import torch
import yaml
from pathlib import Path
import sys


def inspect_checkpoint(checkpoint_path: str):
    """Inspect checkpoint structure."""
    print(f"Inspecting checkpoint: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    
    state_dict = payload.get("state_dict", payload)
    config = payload.get("config", {})
    
    print(f"\nCheckpoint keys:")
    print(f"  - state_dict: {len(state_dict)} keys")
    print(f"  - config: {'present' if config else 'missing'}")
    
    # Analyze state_dict keys
    component_keys = {}
    for key in state_dict.keys():
        if "." in key:
            component = key.split(".")[0]
            if component not in component_keys:
                component_keys[component] = []
            component_keys[component].append(key)
    
    print(f"\nComponents found in state_dict:")
    for component, keys in sorted(component_keys.items()):
        print(f"  - {component}: {len(keys)} keys")
        if component == "embedding_projection":
            # Check for CLIP projection keys
            clip_keys = [k for k in keys if "clip_projections" in k]
            spatial_keys = [k for k in keys if "spatial_proj" in k]
            if clip_keys:
                print(f"    - clip_projections: {len(clip_keys)} keys")
            if spatial_keys:
                print(f"    - spatial_proj: {len(spatial_keys)} keys")
    
    if config:
        print(f"\nConfig structure:")
        for ep_key in ["embedding_projection", "embedding_proj"]:
            if ep_key in config:
                ep_cfg = config[ep_key]
                if isinstance(ep_cfg, dict):
                    print(f"  - {ep_key} type: {ep_cfg.get('type', 'unknown')}")
                    if "clip_projections" in ep_cfg:
                        print(f"    - clip_projections: {ep_cfg['clip_projections']}")
                break
    
    return payload, state_dict, config


def migrate_diffusion_checkpoint(
    old_checkpoint_path: str,
    new_diffusion_checkpoint_path: str,
    clip_projection_checkpoint_path: str
):
    """
    Migrate diffusion checkpoint by separating CLIP projections.
    
    Args:
        old_checkpoint_path: Path to old diffusion checkpoint
        new_diffusion_checkpoint_path: Path to save new diffusion checkpoint (without CLIP projections in embedding_projection)
        clip_projection_checkpoint_path: Path to save CLIP projection checkpoint
    """
    print(f"Loading old checkpoint: {old_checkpoint_path}")
    payload = torch.load(old_checkpoint_path, map_location="cpu")
    
    state_dict = payload.get("state_dict", payload)
    config = payload.get("config", {})
    
    # Separate keys
    # Note: The component might be named "embedding_proj" or "embedding_projection"
    diffusion_state_dict = {}
    clip_proj_state_dict = {}
    embedding_proj_state_dict = {}
    
    # Detect the actual component name
    embedding_proj_prefix = None
    for key in state_dict.keys():
        if key.startswith("embedding_proj."):
            embedding_proj_prefix = "embedding_proj"
            break
        elif key.startswith("embedding_projection."):
            embedding_proj_prefix = "embedding_projection"
            break
    
    if embedding_proj_prefix:
        print(f"  Detected embedding component name: {embedding_proj_prefix}")
    
    for key, value in state_dict.items():
        if embedding_proj_prefix and key.startswith(f"{embedding_proj_prefix}.clip_projections."):
            # Extract CLIP projection keys
            clip_key = key.replace(f"{embedding_proj_prefix}.clip_projections.", "clip_projections.")
            clip_proj_state_dict[clip_key] = value
            # Also keep with unprefixed key for standalone loading
            unprefixed_key = key.replace(f"{embedding_proj_prefix}.clip_projections.", "")
            clip_proj_state_dict[unprefixed_key] = value
        elif embedding_proj_prefix and key.startswith(f"{embedding_proj_prefix}.spatial_proj."):
            # Keep spatial projection in embedding_proj (it's part of CLIPEmbeddingToSpatial)
            embedding_proj_state_dict[key] = value
        elif embedding_proj_prefix and key.startswith(f"{embedding_proj_prefix}."):
            # Other embedding_proj keys (shouldn't be many)
            embedding_proj_state_dict[key] = value
        else:
            # All other components (decoder, unet, scheduler, etc.)
            diffusion_state_dict[key] = value
    
    print(f"\nSeparated keys:")
    print(f"  - Diffusion (decoder, unet, scheduler, etc.): {len(diffusion_state_dict)} keys")
    print(f"  - Embedding projection (spatial_proj, etc.): {len(embedding_proj_state_dict)} keys")
    print(f"  - CLIP projections: {len([k for k in clip_proj_state_dict.keys() if k.startswith('clip_projections')])} keys")
    
    # Extract CLIP projection config if present
    clip_proj_config = None
    for ep_key in ["embedding_projection", "embedding_proj"]:
        if config and ep_key in config:
            ep_cfg = config[ep_key]
            if isinstance(ep_cfg, dict) and "clip_projections" in ep_cfg:
                clip_proj_config = ep_cfg["clip_projections"]
                # If it's a string (path), we'll keep it as is
                # If it's a dict, we'll save it as config
            break
    
    # Save CLIP projection checkpoint
    if clip_proj_state_dict:
        print(f"\nSaving CLIP projection checkpoint to: {clip_projection_checkpoint_path}")
        Path(clip_projection_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        clip_proj_payload = {"state_dict": clip_proj_state_dict}
        if clip_proj_config and isinstance(clip_proj_config, dict):
            clip_proj_payload["config"] = clip_proj_config
        # Preserve any extra state from original checkpoint
        for key in ["epoch", "step", "best_val_loss", "training_history"]:
            if key in payload:
                clip_proj_payload[key] = payload[key]
        torch.save(clip_proj_payload, clip_projection_checkpoint_path)
        print(f"✓ Saved CLIP projection checkpoint")
    else:
        print("⚠ No CLIP projection keys found in checkpoint")
    
    # Create new diffusion config - standalone (no external paths)
    new_config = config.copy() if config else {}
    
    # For standalone checkpoints, CLIP projections are embedded in state_dict
    # Mark this in the config so CLIPEmbeddingToSpatial knows to create minimal component
    # Check both "embedding_projection" and "embedding_proj" (component name might vary)
    for ep_key in ["embedding_projection", "embedding_proj"]:
        if ep_key in new_config:
            ep_cfg = new_config[ep_key].copy() if isinstance(new_config[ep_key], dict) else {}
            if isinstance(ep_cfg, dict):
                # Remove any checkpoint path - CLIP projections are embedded in state_dict
                if "clip_projections" in ep_cfg:
                    if isinstance(ep_cfg["clip_projections"], str):
                        # It was a path - remove it, mark as embedded
                        del ep_cfg["clip_projections"]
                        ep_cfg["_clip_projections_embedded"] = True
                    elif isinstance(ep_cfg["clip_projections"], dict):
                        # It's a dict config - mark as embedded
                        ep_cfg["_clip_projections_embedded"] = True
                else:
                    # No clip_projections in config - mark as embedded
                    ep_cfg["_clip_projections_embedded"] = True
                new_config[ep_key] = ep_cfg
            break
    
    # Rename embedding_proj keys to embedding_projection to match model structure
    # The model uses "embedding_projection" as the component name
    renamed_embedding_proj_state_dict = {}
    for key, value in embedding_proj_state_dict.items():
        if key.startswith("embedding_proj."):
            # Rename to embedding_projection
            new_key = key.replace("embedding_proj.", "embedding_projection.", 1)
            renamed_embedding_proj_state_dict[new_key] = value
        else:
            renamed_embedding_proj_state_dict[key] = value
    
    # Embed CLIP projections back into the diffusion checkpoint (standalone)
    # Map old checkpoint structure to new model structure
    # Old: embedding_proj.clip_projections.text_proj.0.weight
    # New: embedding_projection.clip_projections.text_proj.proj.0.weight
    embedded_clip_proj_state_dict = {}
    
    # First, collect all CLIP projection keys from the original checkpoint
    original_clip_keys = {}
    for key, value in state_dict.items():
        if embedding_proj_prefix and key.startswith(f"{embedding_proj_prefix}.clip_projections."):
            original_clip_keys[key] = value
    
    # Map to new structure: add .proj. for text_proj, pov_proj, latent_proj
    for old_key, value in original_clip_keys.items():
        # Remove embedding_proj prefix, we'll add embedding_projection later
        key_without_prefix = old_key.replace(f"{embedding_proj_prefix}.", "")
        
        # Map old structure to new structure
        # Old: clip_projections.text_proj.0.weight
        # New: clip_projections.text_proj.proj.0.weight
        new_key = key_without_prefix
        for proj_type in ["text_proj", "pov_proj", "latent_proj"]:
            # Check if key has pattern: clip_projections.{proj_type}.{number}.{param}
            pattern = f"clip_projections.{proj_type}."
            if new_key.startswith(pattern):
                # Insert .proj. after the projection type
                # clip_projections.text_proj.0.weight -> clip_projections.text_proj.proj.0.weight
                rest = new_key[len(pattern):]
                if rest[0].isdigit():  # Next part is a number (layer index)
                    new_key = f"{pattern}proj.{rest}"
                    break
        
        # Add embedding_projection prefix
        embedded_key = f"embedding_projection.{new_key}"
        embedded_clip_proj_state_dict[embedded_key] = value
    
    # Merge everything back into diffusion state (standalone checkpoint)
    final_state_dict = {
        **diffusion_state_dict,
        **renamed_embedding_proj_state_dict,
        **embedded_clip_proj_state_dict
    }
    
    # Save new diffusion checkpoint
    print(f"\nSaving new diffusion checkpoint to: {new_diffusion_checkpoint_path}")
    Path(new_diffusion_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    new_payload = {
        "state_dict": final_state_dict,
        "config": new_config
    }
    # Preserve any extra state from original checkpoint
    for key in ["epoch", "step", "best_val_loss", "training_history", "optimizer_state", "scheduler_state", "scaler_state"]:
        if key in payload:
            new_payload[key] = payload[key]
    torch.save(new_payload, new_diffusion_checkpoint_path)
    print(f"✓ Saved new diffusion checkpoint")
    
    return len(final_state_dict), len([k for k in clip_proj_state_dict.keys() if k.startswith('clip_projections')])


def test_checkpoint_load(checkpoint_path: str, clip_projection_path: str = None):
    """Test that checkpoint can be loaded successfully."""
    print(f"\nTesting checkpoint load: {checkpoint_path}")
    try:
        import sys
        from pathlib import Path
        # Add project root to path
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from models.diffusion import DiffusionModel
        
        payload = torch.load(checkpoint_path, map_location="cpu")
        config = payload.get("config")
        
        if not config:
            print("  ✗ No config in checkpoint")
            return False
        
        # Build model from config
        model = DiffusionModel.from_config(config)
        
        # Load state dict
        state_dict = payload.get("state_dict", payload)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"  ⚠ Missing keys: {len(missing_keys)}")
            if len(missing_keys) <= 5:
                for key in missing_keys:
                    print(f"    - {key}")
        
        if unexpected_keys:
            print(f"  ⚠ Unexpected keys: {len(unexpected_keys)}")
            if len(unexpected_keys) <= 5:
                for key in unexpected_keys:
                    print(f"    - {key}")
        
        print(f"  ✓ Checkpoint loads successfully")
        
        # Verify components
        has_decoder = hasattr(model, 'decoder') and model.decoder is not None
        has_unet = hasattr(model, 'unet') and model.unet is not None
        has_scheduler = hasattr(model, 'scheduler') and model.scheduler is not None
        has_embedding_proj = hasattr(model, 'embedding_projection') and model.embedding_projection is not None
        
        print(f"  Components:")
        print(f"    - Decoder: {'✓' if has_decoder else '✗'}")
        print(f"    - UNet: {'✓' if has_unet else '✗'}")
        print(f"    - Scheduler: {'✓' if has_scheduler else '✗'}")
        print(f"    - Embedding Projection: {'✓' if has_embedding_proj else '✗'}")
        
        if has_embedding_proj:
            has_clip_proj = hasattr(model.embedding_projection, 'clip_projections') and model.embedding_projection.clip_projections is not None
            print(f"      - CLIP Projections: {'✓' if has_clip_proj else '✗'}")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate diffusion checkpoint")
    parser.add_argument(
        "--old-checkpoint",
        type=str,
        default="checkpoints/backups/diff_clip_regular_rooms_small_down_bottleneck_text_only_checkpoint_best.pt",
        help="Path to old diffusion checkpoint"
    )
    parser.add_argument(
        "--new-diffusion-checkpoint",
        type=str,
        default="checkpoints/diff_clip_regular_rooms_small_down_bottleneck_text_only_checkpoint_best.pt",
        help="Path to save new diffusion checkpoint"
    )
    parser.add_argument(
        "--clip-projection-checkpoint",
        type=str,
        default="checkpoints/diff_clip_projection_checkpoint_best.pt",
        help="Path to save CLIP projection checkpoint (from embedding_projection)"
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only inspect checkpoint, don't migrate"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test checkpoint loading after migration"
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    old_checkpoint = Path(args.old_checkpoint).resolve()
    new_diffusion_checkpoint = Path(args.new_diffusion_checkpoint).resolve()
    clip_projection_checkpoint = Path(args.clip_projection_checkpoint).resolve()
    
    if not old_checkpoint.exists():
        print(f"Error: Old checkpoint not found: {old_checkpoint}")
        sys.exit(1)
    
    print("="*60)
    print("Diffusion Checkpoint Migration")
    print("="*60)
    
    # Inspect checkpoint
    payload, state_dict, config = inspect_checkpoint(str(old_checkpoint))
    
    if args.inspect_only:
        print("\nInspection complete (--inspect-only mode)")
        sys.exit(0)
    
    # Migrate checkpoint
    print("\n" + "="*60)
    print("Migrating Checkpoint")
    print("="*60)
    
    diffusion_keys, clip_keys = migrate_diffusion_checkpoint(
        str(old_checkpoint),
        str(new_diffusion_checkpoint),
        str(clip_projection_checkpoint)
    )
    
    # Test checkpoint loading if requested
    if args.test:
        print("\n" + "="*60)
        print("Testing Checkpoint Load")
        print("="*60)
        test_checkpoint_load(str(new_diffusion_checkpoint), str(clip_projection_checkpoint))
    
    print("\n" + "="*60)
    print("Migration complete!")
    print("="*60)
    print(f"New diffusion checkpoint: {new_diffusion_checkpoint}")
    print(f"CLIP projection checkpoint: {clip_projection_checkpoint}")

