#!/usr/bin/env python3
"""
Job script to run debug_diffusion.py on all architectures (attention ablations) for different data types.

This script can be submitted as a single HPC job and will run all configurations internally.

Runs debug script on:
- All attention ablations: down, bottleneck, up, all
- All data types: 
  * Regular CLIP VAE: regular (both), regular_rooms, regular_scenes
    (uses vae_clip checkpoint: /work3/s233249/ImgiNav/experiments/clip/vae_clip)
  * Spatial CLIP VAE: spatial (both), spatial_rooms, spatial_scenes
    (uses vae_clip_spatial checkpoint: /work3/s233249/ImgiNav/experiments/clip/vae_clip_spatial)
- Small model size only (for efficiency)
"""

import subprocess
import json
import sys
from pathlib import Path
from collections import defaultdict
import argparse

# Base directory for experiments
EXPERIMENTS_DIR = Path("experiments/diffusion/clip")

# Attention ablations to test
ATTENTION_TYPES = ["down", "bottleneck", "up", "all"]

# Data types to test (both regular and spatial CLIP VAE)
# Regular CLIP VAE: uses vae_clip checkpoint (/work3/s233249/ImgiNav/experiments/clip/vae_clip)
# Spatial CLIP VAE: uses vae_clip_spatial checkpoint (/work3/s233249/ImgiNav/experiments/clip/vae_clip_spatial)
DATA_TYPES = {
    "regular": "regular",  # Regular CLIP VAE (vae_clip), both rooms and scenes
    "regular_rooms": "regular_rooms",  # Regular CLIP VAE (vae_clip), rooms only
    "regular_scenes": "regular_scenes",  # Regular CLIP VAE (vae_clip), scenes only
    "spatial": "spatial",  # Spatial CLIP VAE (vae_clip_spatial), both rooms and scenes
    "spatial_rooms": "spatial_rooms",  # Spatial CLIP VAE (vae_clip_spatial), rooms only
    "spatial_scenes": "spatial_scenes"  # Spatial CLIP VAE (vae_clip_spatial), scenes only
}

# Model size (we'll use small for debugging)
MODEL_SIZE = "small"


def find_config_files():
    """Find all relevant config files."""
    configs = {}
    
    for data_type_name, data_type_dir in DATA_TYPES.items():
        configs[data_type_name] = {}
        for attn_type in ATTENTION_TYPES:
            config_path = EXPERIMENTS_DIR / data_type_dir / f"{MODEL_SIZE}_{attn_type}.yaml"
            if config_path.exists():
                configs[data_type_name][attn_type] = config_path
            else:
                print(f"Warning: Config not found: {config_path}")
    
    return configs


def run_debug(config_path, output_dir):
    """Run debug_diffusion.py on a single config."""
    print(f"\n{'='*80}")
    print(f"Running debug on: {config_path.name}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        result = subprocess.run(
            [sys.executable, "debug_diffusion.py", str(config_path), "--output-dir", str(output_dir)],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr, file=sys.stderr)
        
        # Load metrics
        metrics_path = output_dir / "debug_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return metrics
        else:
            print(f"Warning: Metrics file not found: {metrics_path}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error running debug script:")
        print(e.stdout)
        print(e.stderr, file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="Run debug script on all architectures (job version)")
    parser.add_argument(
        "--output-base-dir",
        type=Path,
        default=Path("debug_results"),
        help="Base directory for all debug results"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip configs that already have results"
    )
    args = parser.parse_args()
    
    output_base_dir = args.output_base_dir
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all config files
    configs = find_config_files()
    
    # Collect all results
    all_results = {}
    
    # Run debug on each config
    for data_type_name, attn_configs in configs.items():
        all_results[data_type_name] = {}
        
        for attn_type, config_path in attn_configs.items():
            # Create output directory: debug_results/regular_rooms/small_down/
            output_dir = output_base_dir / data_type_name / f"{MODEL_SIZE}_{attn_type}"
            
            # Skip if already exists and --skip-existing is set
            if args.skip_existing and (output_dir / "debug_metrics.json").exists():
                print(f"Skipping {data_type_name}/{attn_type} (already exists)")
                with open(output_dir / "debug_metrics.json", 'r') as f:
                    all_results[data_type_name][attn_type] = json.load(f)
                continue
            
            metrics = run_debug(config_path, output_dir)
            all_results[data_type_name][attn_type] = metrics
    
    # Save summary of all results
    summary_path = output_base_dir / "summary_all_results.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*80}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*80}")
    
    # Print summary table
    print("\nSummary of Results:")
    print("-" * 110)
    print(f"{'VAE Type':<15} {'Data Type':<15} {'Attention':<15} {'VAE MSE':<12} {'Init Loss':<12} {'Final Loss':<12} {'Reduction %':<12} {'Opt BS':<8} {'Peak Mem':<10}")
    print("-" * 110)
    
    for data_type_name, attn_results in all_results.items():
        # Determine VAE type and data filter from data_type_name
        if data_type_name.startswith("spatial"):
            vae_type = "spatial"
            filter_type = data_type_name.replace("spatial", "").replace("_", "") or "all"
        else:
            vae_type = "regular"
            filter_type = data_type_name.replace("regular", "").replace("_", "") or "all"
        
        for attn_type, metrics in attn_results.items():
            if metrics is None:
                print(f"{vae_type:<15} {filter_type:<15} {attn_type:<15} {'FAILED':<12} {'-':<12} {'-':<12} {'-':<12} {'-':<8} {'-':<10}")
                continue
            
            vae_mse = "-"
            if metrics.get("vae_round_trip"):
                vae_mse = f"{metrics['vae_round_trip'].get('vae_reconstruction_mse', 0):.6f}"
            
            init_loss = "-"
            final_loss = "-"
            reduction = "-"
            peak_mem = "-"
            if metrics.get("overfit_test"):
                ot = metrics["overfit_test"]
                init_loss = f"{ot.get('initial_loss', 0):.6f}"
                final_loss = f"{ot.get('final_loss', 0):.6f}"
                reduction = f"{ot.get('loss_reduction_percent', 0):.2f}"
                peak_mem = f"{ot.get('peak_memory_gb', 0):.2f}GB"
            
            opt_bs = "-"
            if metrics.get("batch_size_test") and metrics["batch_size_test"].get("optimal_batch_size"):
                opt_bs = str(metrics["batch_size_test"]["optimal_batch_size"])
            
            print(f"{vae_type:<15} {filter_type:<15} {attn_type:<15} {vae_mse:<12} {init_loss:<12} {final_loss:<12} {reduction:<12} {opt_bs:<8} {peak_mem:<10}")
    
    print("-" * 110)
    print(f"\nResults saved to: {output_base_dir}")
    print(f"Run plot_debug_summary.py to generate visualization plots.")


if __name__ == "__main__":
    main()

