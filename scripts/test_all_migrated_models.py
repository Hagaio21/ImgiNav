#!/usr/bin/env python3
"""
Test inference on all migrated models.

This script:
1. Finds all experiment directories in migrated experiments directory
2. For each experiment, finds the best checkpoint
3. Loads the model using the corresponding config file
4. Runs inference with provided condition embeddings
5. Saves samples to test_samples directory
"""

import sys
import yaml
import torch
from pathlib import Path
from torchvision.utils import save_image
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.diffusion import DiffusionModel
from common.utils import load_config_with_profile


def find_experiment_config(experiment_name: str, project_root: Path) -> Path:
    """Find the existing config file for an experiment based on its name."""
    # Parse experiment name to determine config location
    # Pattern: diff_clip_{type}_{dataset}_{size}_{attention}_{modality}
    parts = experiment_name.replace("diff_clip_", "").split("_")
    
    # Determine embedding type
    embedding_type = "regular"
    if parts[0] == "spatial":
        embedding_type = "spatial"
        parts = parts[1:]
    elif parts[0] == "regular":
        parts = parts[1:]
    
    # Determine dataset type
    dataset_type = "rooms"
    if parts[0] in ["rooms", "scenes", "both"]:
        dataset_type = parts[0]
        parts = parts[1:]
    
    # Determine size
    size = "small"
    if parts[0] in ["small", "medium", "large"]:
        size = parts[0]
        parts = parts[1:]
    
    # Determine attention config and modality
    attention_config = "all"
    modality = None
    
    if "text_only" in experiment_name:
        modality = "text_only"
    elif "pov_only" in experiment_name:
        modality = "pov_only"
    
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
    
    # Build config path - use existing config files
    if dataset_type == "both":
        config_subdir = embedding_type
    else:
        config_subdir = f"{embedding_type}_{dataset_type}"
    
    config_filename = f"{size}_{attention_config}"
    if modality:
        config_filename += f"_{modality}"
    config_filename += ".yaml"
    
    config_path = project_root / "experiments" / "diffusion" / "clip" / config_subdir / config_filename
    
    return config_path


def find_best_checkpoint(experiment_dir: Path) -> Path:
    """Find the best checkpoint in an experiment directory."""
    checkpoints_dir = experiment_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return None
    
    # Look for checkpoint_best.pt first
    best_checkpoint = checkpoints_dir / f"{experiment_dir.name}_checkpoint_best.pt"
    if best_checkpoint.exists():
        return best_checkpoint
    
    # Fallback to checkpoint_latest.pt
    latest_checkpoint = checkpoints_dir / f"{experiment_dir.name}_checkpoint_latest.pt"
    if latest_checkpoint.exists():
        return latest_checkpoint
    
    # Try to find any checkpoint file
    checkpoint_files = list(checkpoints_dir.glob("*checkpoint*.pt"))
    if checkpoint_files:
        return checkpoint_files[0]
    
    return None


def run_inference(
    model: DiffusionModel,
    text_emb: torch.Tensor = None,
    pov_emb: torch.Tensor = None,
    batch_size: int = 1,
    num_steps: int = 50,
    guidance_scale: float = 5.0,
    device: str = "cuda"
):
    """Run inference with the model."""
    model.eval()
    
    with torch.no_grad():
        samples = model.sample(
            batch_size=batch_size,
            num_steps=num_steps,
            method="ddim",
            eta=0.0,
            device=device,
            guidance_scale=guidance_scale,
            text_emb=text_emb,
            pov_emb=pov_emb,
            verbose=False
        )
        
        # Extract RGB from samples dict
        if isinstance(samples, dict) and "rgb" in samples:
            samples = samples["rgb"]
        elif isinstance(samples, dict) and "latent" in samples:
            # Decode latents
            decoded = model.decoder({"latent": samples["latent"]})
            if "rgb" in decoded:
                samples = decoded["rgb"]
            else:
                raise ValueError("Decoder did not produce RGB output")
    
    return samples


def test_all_models(
    migrated_experiments_dir: str,
    output_dir: str,
    text_emb_path: str,
    pov_emb_path: str,
    project_root: str = None,
    batch_size: int = 1,
    num_steps: int = 50,
    guidance_scale: float = 5.0,
    device: str = None
):
    """
    Test inference on all migrated models.
    
    Args:
        migrated_experiments_dir: Directory containing migrated experiments
        output_dir: Directory to save test samples
        text_emb_path: Path to text embedding file
        pov_emb_path: Path to POV embedding file
        project_root: Project root directory (default: current directory)
        batch_size: Batch size for inference
        num_steps: Number of sampling steps
        guidance_scale: Guidance scale for classifier-free guidance
        device: Device to use (default: auto-detect)
    """
    migrated_dir = Path(migrated_experiments_dir)
    output_path = Path(output_dir)
    project_root = Path(project_root) if project_root else Path.cwd()
    
    if not migrated_dir.exists():
        print(f"Error: Migrated experiments directory does not exist: {migrated_dir}")
        return
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load condition embeddings
    print(f"\nLoading condition embeddings...")
    text_emb = None
    pov_emb = None
    
    if text_emb_path and Path(text_emb_path).exists():
        text_emb = torch.load(text_emb_path, map_location=device)
        # Ensure proper shape: [batch, features]
        if text_emb.dim() == 1:
            # 1D tensor: add batch dimension
            text_emb = text_emb.unsqueeze(0)
        elif text_emb.dim() > 2:
            # Multi-dimensional: flatten spatial dims but keep batch
            text_emb = text_emb.flatten(start_dim=1)
        # Expand to batch size if needed
        if text_emb.shape[0] == 1 and batch_size > 1:
            text_emb = text_emb.repeat(batch_size, 1)
        print(f"  ✓ Text embedding: {text_emb.shape}")
    else:
        print(f"  ⚠ Text embedding not found: {text_emb_path}")
    
    if pov_emb_path and Path(pov_emb_path).exists():
        pov_emb = torch.load(pov_emb_path, map_location=device)
        # Ensure proper shape: [batch, features]
        if pov_emb.dim() == 1:
            # 1D tensor: add batch dimension
            pov_emb = pov_emb.unsqueeze(0)
        elif pov_emb.dim() > 2:
            # Multi-dimensional: flatten spatial dims but keep batch
            pov_emb = pov_emb.flatten(start_dim=1)
        # Expand to batch size if needed
        if pov_emb.shape[0] == 1 and batch_size > 1:
            pov_emb = pov_emb.repeat(batch_size, 1)
        print(f"  ✓ POV embedding: {pov_emb.shape}")
    else:
        print(f"  ⚠ POV embedding not found: {pov_emb_path}")
    
    # Find all experiment directories
    experiment_dirs = [
        d for d in migrated_dir.iterdir()
        if d.is_dir() and d.name.startswith("diff_clip_")
    ]
    
    print(f"\nFound {len(experiment_dirs)} experiments to test")
    print("=" * 80)
    
    successful = 0
    failed = 0
    
    for exp_dir in sorted(experiment_dirs):
        exp_name = exp_dir.name
        print(f"\n[{successful + failed + 1}/{len(experiment_dirs)}] Testing: {exp_name}")
        
        try:
            # Find checkpoint
            checkpoint_path = find_best_checkpoint(exp_dir)
            if checkpoint_path is None:
                print(f"  ✗ No checkpoint found")
                failed += 1
                continue
            
            print(f"  Checkpoint: {checkpoint_path.name}")
            
            # Find config file
            config_path = find_experiment_config(exp_name, project_root)
            if not config_path.exists():
                print(f"  ✗ Config file not found: {config_path}")
                failed += 1
                continue
            
            print(f"  Config: {config_path.relative_to(project_root)}")
            
            # Load config
            config = load_config_with_profile(str(config_path), resolve_checkpoints=True)
            
            # Determine which embeddings to use based on experiment name
            use_text_emb = text_emb is not None
            use_pov_emb = pov_emb is not None
            
            # Check if experiment is modality-specific
            if "text_only" in exp_name:
                use_pov_emb = False
            elif "pov_only" in exp_name:
                use_text_emb = False
            
            # Prepare embeddings
            current_text_emb = text_emb if use_text_emb else None
            current_pov_emb = pov_emb if use_pov_emb else None
            
            # Load model
            print(f"  Loading model...")
            model = DiffusionModel.load_checkpoint(
                str(checkpoint_path),
                map_location=device,
                config=config
            )
            model = model.to(device)
            print(f"  ✓ Model loaded")
            
            # Run inference
            print(f"  Running inference...")
            samples = run_inference(
                model,
                text_emb=current_text_emb,
                pov_emb=current_pov_emb,
                batch_size=batch_size,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                device=device
            )
            
            # Save samples
            output_file = output_path / f"{exp_name}_samples.png"
            save_image(samples, output_file, nrow=min(4, batch_size), normalize=False)
            print(f"  ✓ Saved: {output_file.name}")
            
            successful += 1
            
            # Clean up model to free memory
            del model
            torch.cuda.empty_cache() if device == "cuda" else None
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue
    
    print("\n" + "=" * 80)
    print(f"Testing complete!")
    print(f"  Successful: {successful}/{len(experiment_dirs)}")
    print(f"  Failed: {failed}/{len(experiment_dirs)}")
    print(f"  Output directory: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test inference on all migrated models")
    parser.add_argument(
        "--migrated-dir",
        type=str,
        default="D:/migrated_experiments",
        help="Directory containing migrated experiments"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="D:/migrated_experiments/test_samples",
        help="Directory to save test samples"
    )
    parser.add_argument(
        "--text-emb",
        type=str,
        default="D:/0a8d471a-2587-458a-9214-586e003e9cf9_3019_room_graph.pt",
        help="Path to text embedding file"
    )
    parser.add_argument(
        "--pov-emb",
        type=str,
        default="D:/0a8d471a-2587-458a-9214-586e003e9cf9_3019_v02_pov_seg.pt",
        help="Path to POV embedding file"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Number of sampling steps"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=5.0,
        help="Guidance scale for classifier-free guidance"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    test_all_models(
        args.migrated_dir,
        args.output_dir,
        args.text_emb,
        args.pov_emb,
        args.project_root,
        args.batch_size,
        args.num_steps,
        args.guidance_scale,
        args.device
    )

