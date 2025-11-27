"""
Migration script to separate CLIP projection from VAE checkpoint.

This script:
1. Loads the old VAE checkpoint with CLIP projections
2. Extracts CLIP projection weights and saves them separately
3. Saves a new VAE checkpoint without CLIP projections
4. Updates all diffusion experiment configs to use the new checkpoint path
"""
import torch
import yaml
from pathlib import Path
import sys
import shutil


def migrate_vae_checkpoint(
    old_checkpoint_path: str,
    new_vae_checkpoint_path: str,
    clip_projection_checkpoint_path: str
):
    """
    Migrate VAE checkpoint by separating CLIP projections.
    
    Args:
        old_checkpoint_path: Path to old VAE checkpoint with CLIP projections
        new_vae_checkpoint_path: Path to save new VAE checkpoint (without CLIP projections)
        clip_projection_checkpoint_path: Path to save CLIP projection checkpoint
    """
    print(f"Loading old checkpoint: {old_checkpoint_path}")
    payload = torch.load(old_checkpoint_path, map_location="cpu")
    
    state_dict = payload.get("state_dict", payload)
    config = payload.get("config", {})
    
    # Separate CLIP projection keys from VAE keys
    vae_state_dict = {}
    clip_proj_state_dict = {}
    
    clip_proj_prefixes = ["clip_projection.", "clip_projections."]
    
    for key, value in state_dict.items():
        is_clip_proj = any(key.startswith(prefix) for prefix in clip_proj_prefixes)
        if is_clip_proj:
            # Remove prefix for standalone checkpoint
            # Keep both prefixed and unprefixed versions for compatibility
            clip_proj_state_dict[key] = value
            # Also save without prefix for standalone loading
            unprefixed_key = key
            for prefix in clip_proj_prefixes:
                if key.startswith(prefix):
                    unprefixed_key = key[len(prefix):]
                    break
            if unprefixed_key != key:
                clip_proj_state_dict[unprefixed_key] = value
        else:
            vae_state_dict[key] = value
    
    print(f"Found {len(vae_state_dict)} VAE keys")
    print(f"Found {len([k for k in clip_proj_state_dict.keys() if k.startswith('clip_projection') or k.startswith('clip_projections')])} CLIP projection keys")
    
    # Extract CLIP projection config if present
    clip_proj_config = None
    if "clip_projection" in config:
        clip_proj_config = config["clip_projection"]
    elif "clip_projections" in config:
        clip_proj_config = config["clip_projections"]
    
    # Save CLIP projection checkpoint
    if clip_proj_state_dict:
        print(f"\nSaving CLIP projection checkpoint to: {clip_projection_checkpoint_path}")
        Path(clip_projection_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        clip_proj_payload = {"state_dict": clip_proj_state_dict}
        if clip_proj_config:
            clip_proj_payload["config"] = clip_proj_config
        # Preserve any extra state from original checkpoint
        for key in ["epoch", "step", "best_val_loss", "training_history"]:
            if key in payload:
                clip_proj_payload[key] = payload[key]
        torch.save(clip_proj_payload, clip_projection_checkpoint_path)
        print(f"✓ Saved CLIP projection checkpoint")
    else:
        print("⚠ No CLIP projection keys found in checkpoint")
    
    # Create new VAE config - standalone (no external paths)
    # Remove CLIP projection from config since it's now separate
    # But keep the structure clean (no path references)
    new_config = config.copy()
    new_config.pop("clip_projection", None)
    new_config.pop("clip_projections", None)
    
    # Also remove any checkpoint paths from encoder/decoder configs if they exist
    # The weights are embedded in the state_dict
    for component_key in ["encoder", "decoder"]:
        if component_key in new_config and isinstance(new_config[component_key], dict):
            if "checkpoint" in new_config[component_key]:
                # Keep the checkpoint path only if it's for a different model
                # Otherwise, weights are in this checkpoint's state_dict
                pass  # For now, keep checkpoint paths in encoder/decoder configs
    
    # Save new VAE checkpoint
    print(f"\nSaving new VAE checkpoint to: {new_vae_checkpoint_path}")
    Path(new_vae_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    new_payload = {
        "state_dict": vae_state_dict,
        "config": new_config
    }
    # Preserve any extra state from original checkpoint
    for key in ["epoch", "step", "best_val_loss", "training_history"]:
        if key in payload:
            new_payload[key] = payload[key]
    torch.save(new_payload, new_vae_checkpoint_path)
    print(f"✓ Saved new VAE checkpoint")
    
    return len(vae_state_dict), len([k for k in clip_proj_state_dict.keys() if k.startswith('clip_projection') or k.startswith('clip_projections')])


def update_embedding_projection_configs(
    experiments_dir: str,
    clip_projection_checkpoint_path: str = None,
    mark_embedded: bool = True
):
    """
    Update all diffusion experiment configs for standalone checkpoints.
    
    Args:
        experiments_dir: Directory containing experiment configs
        clip_projection_checkpoint_path: Path to CLIP projection checkpoint (optional, for reference)
        mark_embedded: If True, mark CLIP projections as embedded (for standalone checkpoints)
                      If False, add path to CLIP projection checkpoint
    """
    experiments_dir = Path(experiments_dir)
    config_files = list(experiments_dir.rglob("*.yaml"))
    
    updated_count = 0
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                content = f.read()
                config = yaml.safe_load(content)
            
            # Check if this config has CLIPEmbeddingToSpatial
            needs_update = False
            if "embedding_projection" in config:
                ep_config = config["embedding_projection"]
                if isinstance(ep_config, dict):
                    ep_type = ep_config.get("type", "")
                    if ep_type == "CLIPEmbeddingToSpatial":
                        if mark_embedded:
                            # For standalone checkpoints: mark as embedded, remove any paths
                            if "clip_projections" in ep_config and isinstance(ep_config["clip_projections"], str):
                                # Remove path - CLIP projections are embedded in checkpoint
                                del ep_config["clip_projections"]
                            if "_clip_projections_embedded" not in ep_config:
                                ep_config["_clip_projections_embedded"] = True
                                needs_update = True
                        else:
                            # Add path to CLIP projection checkpoint (for non-standalone)
                            if "clip_projections" not in ep_config and clip_projection_checkpoint_path:
                                ep_config["clip_projections"] = str(clip_projection_checkpoint_path)
                                needs_update = True
            
            if needs_update:
                with open(config_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
                print(f"✓ Updated embedding_projection: {config_file}")
                updated_count += 1
        except Exception as e:
            print(f"⚠ Error updating {config_file}: {e}")
    
    print(f"\nUpdated {updated_count} embedding_projection configs")
    return updated_count


def update_experiment_configs(
    experiments_dir: str,
    old_checkpoint_paths: list,
    new_checkpoint_path: str
):
    """
    Update all diffusion experiment configs to use the new checkpoint path.
    
    Args:
        experiments_dir: Directory containing experiment configs
        old_checkpoint_paths: List of old checkpoint paths to replace (can be partial matches)
        new_checkpoint_path: New checkpoint path to use
    """
    experiments_dir = Path(experiments_dir)
    config_files = list(experiments_dir.rglob("*.yaml"))
    
    # Normalize paths for comparison
    old_checkpoint_paths = [str(Path(p).resolve()) for p in old_checkpoint_paths]
    new_checkpoint_path = str(Path(new_checkpoint_path).resolve())
    
    updated_count = 0
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                content = f.read()
                config = yaml.safe_load(content)
            
            # Check if this config references any of the old checkpoint paths
            needs_update = False
            if "autoencoder" in config:
                if isinstance(config["autoencoder"], dict):
                    current_checkpoint = config["autoencoder"].get("checkpoint", "")
                    # Check if current checkpoint matches any old path (by filename or full path)
                    for old_path in old_checkpoint_paths:
                        old_filename = Path(old_path).name
                        if current_checkpoint == old_path or current_checkpoint.endswith(old_filename):
                            config["autoencoder"]["checkpoint"] = new_checkpoint_path
                            needs_update = True
                            break
                elif isinstance(config["autoencoder"], str):
                    current_checkpoint = config["autoencoder"]
                    for old_path in old_checkpoint_paths:
                        old_filename = Path(old_path).name
                        if current_checkpoint == old_path or current_checkpoint.endswith(old_filename):
                            config["autoencoder"] = new_checkpoint_path
                            needs_update = True
                            break
            
            if needs_update:
                with open(config_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
                print(f"✓ Updated: {config_file}")
                updated_count += 1
        except Exception as e:
            print(f"⚠ Error updating {config_file}: {e}")
    
    print(f"\nUpdated {updated_count} experiment configs")
    return updated_count


def test_experiment_build(experiment_config_path: str, project_root: str = None):
    """
    Test if an experiment config can be built successfully.
    
    Args:
        experiment_config_path: Path to experiment YAML config
        project_root: Root directory of the project (for adding to Python path)
        
    Returns:
        True if build successful, False otherwise
    """
    try:
        import sys
        if project_root:
            project_root = Path(project_root).resolve()
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
        
        from models.diffusion import DiffusionModel
        
        config_path = Path(experiment_config_path)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Build model from config
        model = DiffusionModel.from_config(config)
        
        print(f"✓ {config_path.name}: Build successful")
        return True
    except Exception as e:
        print(f"✗ {config_path.name}: Build failed - {e}")
        return False


def test_all_experiments(experiments_dir: str, project_root: str = None):
    """
    Test all diffusion experiments can be built.
    
    Args:
        experiments_dir: Directory containing experiment configs
        project_root: Root directory of the project (for adding to Python path)
        
    Returns:
        Tuple of (successful_count, total_count)
    """
    experiments_dir = Path(experiments_dir)
    config_files = list(experiments_dir.rglob("*.yaml"))
    
    print(f"\nTesting {len(config_files)} experiment configs...")
    
    successful = 0
    failed = []
    
    for config_file in config_files:
        if test_experiment_build(config_file, project_root):
            successful += 1
        else:
            failed.append(config_file)
    
    print(f"\n{'='*60}")
    print(f"Build test results: {successful}/{len(config_files)} successful")
    if failed:
        print(f"\nFailed configs:")
        for f in failed:
            print(f"  - {f}")
    
    return successful, len(config_files)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate VAE checkpoint and test experiments")
    parser.add_argument(
        "--old-checkpoint",
        type=str,
        default="checkpoints/backups/vae_clip_checkpoint_best.pt",
        help="Path to old VAE checkpoint with CLIP projections"
    )
    parser.add_argument(
        "--new-vae-checkpoint",
        type=str,
        default="checkpoints/vae_checkpoint_best.pt",
        help="Path to save new VAE checkpoint (without CLIP projections)"
    )
    parser.add_argument(
        "--clip-projection-checkpoint",
        type=str,
        default="checkpoints/clip_projection_checkpoint_best.pt",
        help="Path to save CLIP projection checkpoint"
    )
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default="experiments/diffusion",
        help="Directory containing diffusion experiment configs"
    )
    parser.add_argument(
        "--update-configs",
        action="store_true",
        help="Update experiment configs to use new checkpoint path"
    )
    parser.add_argument(
        "--old-checkpoint-patterns",
        type=str,
        nargs="+",
        default=None,
        help="Additional old checkpoint path patterns to match (for updating configs)"
    )
    parser.add_argument(
        "--test-experiments",
        action="store_true",
        help="Test all experiments can be built"
    )
    parser.add_argument(
        "--update-embedding-projections",
        action="store_true",
        help="Update embedding_projection configs to include clip_projections path"
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    old_checkpoint = Path(args.old_checkpoint).resolve()
    new_vae_checkpoint = Path(args.new_vae_checkpoint).resolve()
    clip_projection_checkpoint = Path(args.clip_projection_checkpoint).resolve()
    experiments_dir = Path(args.experiments_dir).resolve()
    
    if not old_checkpoint.exists():
        print(f"Error: Old checkpoint not found: {old_checkpoint}")
        sys.exit(1)
    
    print("="*60)
    print("VAE Checkpoint Migration")
    print("="*60)
    
    # Migrate checkpoint
    vae_keys, clip_keys = migrate_vae_checkpoint(
        str(old_checkpoint),
        str(new_vae_checkpoint),
        str(clip_projection_checkpoint)
    )
    
    # Update experiment configs if requested
    if args.update_configs:
        print("\n" + "="*60)
        print("Updating Experiment Configs")
        print("="*60)
        old_paths = [str(old_checkpoint)]
        if args.old_checkpoint_patterns:
            old_paths.extend(args.old_checkpoint_patterns)
        # Also add common old paths found in experiments
        old_paths.extend([
            "/work3/s233249/ImgiNav/experiments/clip/vae_clip/checkpoints/vae_clip_checkpoint_best.pt",
            "/work3/s233249/ImgiNav/experiments/clip/vae_clip_spatial/checkpoints/vae_clip_spatial_checkpoint_best.pt",
            "vae_clip_checkpoint_best.pt",
            "vae_clip_spatial_checkpoint_best.pt"
        ])
        update_experiment_configs(
            str(experiments_dir),
            old_paths,
            str(new_vae_checkpoint)
        )
    
    # Update embedding_projection configs if requested
    if args.update_embedding_projections:
        print("\n" + "="*60)
        print("Updating Embedding Projection Configs")
        print("="*60)
        update_embedding_projection_configs(
            str(experiments_dir),
            str(clip_projection_checkpoint)
        )
    
    # Test experiments if requested
    if args.test_experiments:
        print("\n" + "="*60)
        print("Testing Experiment Builds")
        print("="*60)
        # Get project root (parent of experiments_dir or current working directory)
        project_root = Path.cwd()
        test_all_experiments(str(experiments_dir), str(project_root))
    
    print("\n" + "="*60)
    print("Migration complete!")
    print("="*60)
    print(f"New VAE checkpoint: {new_vae_checkpoint}")
    print(f"CLIP projection checkpoint: {clip_projection_checkpoint}")

