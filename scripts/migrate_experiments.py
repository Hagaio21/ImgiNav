"""
Migration script to copy old experiments and generate YAML configs for the new architecture.

This script:
1. Copies experiment directories from old location to experiments/migrated
2. Generates YAML config files based on experiment names
3. Updates paths to use the new architecture (checkpoint registry, relative paths, etc.)
"""

import yaml
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import re


def parse_experiment_name(name: str) -> Dict[str, Any]:
    """
    Parse experiment name to extract configuration.
    
    Pattern: diff_clip_{type}_{dataset}_{size}_{attention}_{modality}
    
    Examples:
    - diff_clip_regular_rooms_small_down_bottleneck_text_only
    - diff_clip_spatial_scenes_medium_all
    - diff_clip_regular_both_large_down
    """
    parts = name.replace("diff_clip_", "").split("_")
    
    # Determine embedding type (regular or spatial)
    embedding_type = "regular"
    if parts[0] == "spatial":
        embedding_type = "spatial"
        parts = parts[1:]
    elif parts[0] == "regular":
        parts = parts[1:]
    
    # Determine dataset type (rooms, scenes, or both)
    dataset_type = "rooms"
    if parts[0] in ["rooms", "scenes", "both"]:
        dataset_type = parts[0]
        parts = parts[1:]
    
    # Determine size (small, medium, large)
    size = "small"
    if parts[0] in ["small", "medium", "large"]:
        size = parts[0]
        parts = parts[1:]
    
    # Determine attention config and modality
    attention_config = "all"
    modality = None
    
    # Check for modality suffixes
    if "text_only" in name:
        modality = "text_only"
        name = name.replace("_text_only", "")
    elif "pov_only" in name:
        modality = "pov_only"
        name = name.replace("_pov_only", "")
    
    # Extract attention config from remaining parts
    remaining = "_".join(parts)
    if "down_bottleneck" in remaining:
        attention_config = "down_bottleneck"
    elif "down" in remaining:
        attention_config = "down"
    elif "up" in remaining:
        attention_config = "up"
    elif "bottleneck" in remaining:
        attention_config = "bottleneck"
    elif "all" in remaining or remaining == "":
        attention_config = "all"
    
    return {
        "embedding_type": embedding_type,
        "dataset_type": dataset_type,
        "size": size,
        "attention_config": attention_config,
        "modality": modality,
        "original_name": name
    }


def get_unet_config(size: str, attention_config: str) -> Dict[str, Any]:
    """Get UNet configuration based on size and attention config."""
    # Base channels based on size
    base_channels_map = {
        "small": 48,
        "medium": 64,
        "large": 96
    }
    base_channels = base_channels_map.get(size, 48)
    
    # Attention locations based on config
    attention_at_map = {
        "all": ["downs", "bottleneck", "ups"],
        "down_bottleneck": ["downs", "bottleneck"],
        "down": ["downs"],
        "up": ["ups"],
        "bottleneck": ["bottleneck"]
    }
    attention_at = attention_at_map.get(attention_config, ["downs", "bottleneck", "ups"])
    
    return {
        "type": "UnetWithAttention",
        "in_channels": 4,
        "out_channels": 4,
        "base_channels": base_channels,
        "depth": 3,
        "num_res_blocks": 2,
        "time_dim": 256,
        "norm_groups": 8,
        "dropout": 0.1,
        "use_attention": True,
        "attention_heads": 2,
        "attention_at": attention_at,
        "enable_cross_attention": True,
        "use_ema": False
    }


def get_dataset_config(dataset_type: str, modality: Optional[str] = None) -> Dict[str, Any]:
    """Get dataset configuration based on dataset type and modality."""
    config = {
        "manifest": "data/manifest.csv",  # Will be updated to use relative path
        "outputs": {
            "latent": "latent_path_vae_clip",
        },
        "filters": {
            "is_empty": [False]
        },
        "return_path": False
    }
    
    # Add dataset type filter
    if dataset_type == "rooms":
        config["filters"]["type"] = ["room"]
    elif dataset_type == "scenes":
        config["filters"]["type"] = ["scene"]
    # "both" doesn't need a type filter
    
    # Add embedding outputs based on modality
    if modality == "text_only":
        config["outputs"]["text_emb"] = "graph_embedding_path"
    elif modality == "pov_only":
        config["outputs"]["pov_emb"] = "pov_embedding_path"
    else:
        # Both modalities
        config["outputs"]["text_emb"] = "graph_embedding_path"
        config["outputs"]["pov_emb"] = "pov_embedding_path"
    
    return config


def get_embedding_projection_config(embedding_type: str, size: str) -> Dict[str, Any]:
    """Get embedding projection configuration."""
    # Spatial size based on model size
    spatial_size_map = {
        "small": [16, 16],
        "medium": [32, 32],
        "large": [32, 32]
    }
    spatial_size = spatial_size_map.get(size, [16, 16])
    
    # Output channels based on size
    output_channels_map = {
        "small": 48,
        "medium": 64,
        "large": 96
    }
    output_channels = output_channels_map.get(size, 48)
    
    return {
        "type": "CLIPEmbeddingToSpatial",
        "output_channels": output_channels,
        "spatial_size": spatial_size,
        "combine_method": "average",
        "_clip_projections_embedded": True
    }


def get_training_config(size: str) -> Dict[str, Any]:
    """Get training configuration based on model size."""
    # Batch size based on model size
    batch_size_map = {
        "small": 64,
        "medium": 32,
        "large": 16
    }
    batch_size = batch_size_map.get(size, 64)
    
    return {
        "seed": 2024,
        "train_split": 0.8,
        "batch_size": batch_size,
        "num_workers": 4,
        "shuffle": True,
        "epochs": 1000,
        "learning_rate": 0.0001,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "save_interval": 20,
        "eval_interval": 10,
        "sample_interval": 10,
        "use_amp": True,
        "max_grad_norm": 0.5,
        "gradient_accumulation_steps": 1,
        "scheduler": {
            "type": "cosine"
        },
        "cfg_dropout_rate": 0.1,
        "guidance_scale": 5.0,
        "loss": {
            "type": "MSELoss",
            "key": "pred_noise",
            "target": "noise",
            "weight": 1.0
        }
    }


def generate_config(experiment_name: str, base_dir: Path) -> Dict[str, Any]:
    """Generate YAML config for an experiment."""
    parsed = parse_experiment_name(experiment_name)
    
    # Determine config file location based on structure
    config_subdir = f"{parsed['embedding_type']}_{parsed['dataset_type']}"
    if parsed['dataset_type'] == "both":
        config_subdir = parsed['embedding_type']
    
    config_filename = f"{parsed['size']}_{parsed['attention_config']}"
    if parsed['modality']:
        config_filename += f"_{parsed['modality']}"
    config_filename += ".yaml"
    
    config_path = base_dir / "experiments" / "diffusion" / "clip" / config_subdir / config_filename
    
    # Generate config
    config = {
        "experiment": {
            "name": experiment_name,
            "phase": "diffusion_training",
            "save_path": f"outputs/{experiment_name}",
            "epochs_target": 1000
        },
        "dataset": get_dataset_config(parsed['dataset_type'], parsed['modality']),
        "autoencoder": {
            "checkpoint": "@vae_best",  # Use checkpoint registry
            "frozen": True
        },
        "latent_clamp_min": -6.0,
        "latent_clamp_max": 6.0,
        "scale_factor": 1.0,
        "embedding_projection": get_embedding_projection_config(parsed['embedding_type'], parsed['size']),
        "unet": get_unet_config(parsed['size'], parsed['attention_config']),
        "scheduler": {
            "type": "LinearScheduler",
            "num_steps": 1000
        },
        "training": get_training_config(parsed['size'])
    }
    
    return config, config_path


def copy_experiment_directory(source: Path, dest: Path) -> bool:
    """Copy an experiment directory."""
    try:
        if dest.exists():
            print(f"  ⚠ Destination already exists: {dest}")
            return False
        
        shutil.copytree(source, dest)
        print(f"  ✓ Copied: {source.name}")
        return True
    except Exception as e:
        print(f"  ✗ Error copying {source.name}: {e}")
        return False


def migrate_experiments(
    source_dir: str,
    target_dir: str,
    project_root: Optional[str] = None
):
    """
    Migrate experiments from old location to new architecture.
    
    Args:
        source_dir: Source directory containing old experiments
        target_dir: Target directory for migrated experiments
        project_root: Project root directory (for config paths)
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    project_root = Path(project_root) if project_root else Path.cwd()
    
    if not source_path.exists():
        print(f"Error: Source directory does not exist: {source_dir}")
        return
    
    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Get all experiment directories (exclude non-experiment dirs)
    exclude_dirs = {"comparison_summary", "vae_clip", "vae_clip_spatial"}
    experiment_dirs = [
        d for d in source_path.iterdir()
        if d.is_dir() and d.name.startswith("diff_clip_") and d.name not in exclude_dirs
    ]
    
    print(f"Found {len(experiment_dirs)} experiment directories to migrate")
    print("=" * 60)
    
    # Copy directories and generate configs
    copied_count = 0
    config_count = 0
    
    for exp_dir in sorted(experiment_dirs):
        exp_name = exp_dir.name
        print(f"\nProcessing: {exp_name}")
        
        # Copy directory
        dest_dir = target_path / exp_name
        if copy_experiment_directory(exp_dir, dest_dir):
            copied_count += 1
        
        # Generate config
        try:
            config, config_path = generate_config(exp_name, project_root)
            
            # Create config directory if needed
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if config already exists - don't overwrite existing configs
            if config_path.exists():
                print(f"  ⚠ Config already exists, skipping: {config_path.relative_to(project_root)}")
            else:
                # Write config only if it doesn't exist
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
                
                print(f"  ✓ Generated config: {config_path.relative_to(project_root)}")
                config_count += 1
        except Exception as e:
            print(f"  ✗ Error generating config for {exp_name}: {e}")
    
    print("\n" + "=" * 60)
    print(f"Migration complete!")
    print(f"  - Copied {copied_count}/{len(experiment_dirs)} experiment directories")
    print(f"  - Generated {config_count} config files")
    print(f"  - Target directory: {target_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate experiments to new architecture")
    parser.add_argument(
        "--source-dir",
        type=str,
        default="D:/experiments/clip",
        help="Source directory containing old experiments"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="D:/migrated_experiments",
        help="Target directory for migrated experiments"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root directory (default: current directory)"
    )
    
    args = parser.parse_args()
    
    migrate_experiments(
        args.source_dir,
        args.target_dir,
        args.project_root
    )

