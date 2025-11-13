#!/usr/bin/env python3
"""
Script to analyze model sizes (number of trainable parameters) from experiment configs.

Usage:
    python scripts/analyze_model_sizes.py [--output OUTPUT_FILE] [--config-dir EXPERIMENTS_DIR]
    
Examples:
    # Analyze all configs in experiments/ directory
    python scripts/analyze_model_sizes.py
    
    # Save to specific file
    python scripts/analyze_model_sizes.py --output model_sizes.csv
    
    # Analyze specific directory
    python scripts/analyze_model_sizes.py --config-dir experiments/diffusion
"""

import argparse
import yaml
import torch
from pathlib import Path
import sys
import json
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.autoencoder import Autoencoder
from models.diffusion import DiffusionModel


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters in a model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "frozen": total - trainable
    }


def determine_model_type(config: dict) -> str:
    """Determine model type from config structure."""
    if "unet" in config:
        return "Diffusion"
    elif "autoencoder" in config:
        ae_cfg = config.get("autoencoder", {})
        # Check if it's a full autoencoder config (has encoder/decoder)
        if "encoder" in ae_cfg and "decoder" in ae_cfg:
            return "Autoencoder"
        else:
            # Has checkpoint but no encoder/decoder - likely diffusion using autoencoder
            return "Diffusion"
    else:
        return "Unknown"


def build_model_from_config(config: dict, config_path: Path) -> Optional[torch.nn.Module]:
    """Build model from config, handling different model types. Returns the relevant component for parameter counting."""
    model_type = determine_model_type(config)
    
    try:
        if model_type == "Autoencoder":
            ae_cfg = config.get("autoencoder", {})
            # Build from config (ignore checkpoint)
            if "encoder" in ae_cfg and "decoder" in ae_cfg:
                model = Autoencoder.from_config(ae_cfg)
                return model
            else:
                print(f"  Warning: Autoencoder config missing encoder/decoder")
                return None
            
        elif model_type == "Diffusion":
            # We only care about UNet trainable parameters, not decoder
            # Build UNet directly from config
            unet_cfg = config.get("unet", {})
            if not unet_cfg:
                print(f"  Warning: Diffusion config missing 'unet' section")
                return None
            
            from models.components.unet import Unet, DualUNet, UnetWithAttention
            
            # Handle different UNet types
            unet_type = unet_cfg.get("type", "").lower()
            if unet_type in ("dualunet", "dual_unet"):
                unet = DualUNet.from_config(unet_cfg)
            elif unet_type in ("unetwithattention", "unet_with_attention"):
                unet = UnetWithAttention.from_config(unet_cfg)
            else:
                unet = Unet.from_config(unet_cfg)
            
            # Apply freezing if specified
            if unet_cfg.get("frozen", False):
                unet.freeze()
            
            return unet
            
        else:
            print(f"  Warning: Unknown model type: {model_type}")
            return None
            
    except Exception as e:
        print(f"  Error building model: {e}")
        import traceback
        traceback.print_exc()
        return None


def format_number(num: int) -> str:
    """Format number with commas and appropriate units."""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)


def analyze_config(config_path: Path) -> Optional[Dict]:
    """Analyze a single config file and return model size info."""
    print(f"\nAnalyzing: {config_path}")
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        if not config:
            print(f"  Warning: Empty config file")
            return None
        
        # Get experiment name
        exp_name = config.get("experiment", {}).get("name", config_path.stem)
        
        # Determine model type
        model_type = determine_model_type(config)
        
        # Build model
        model = build_model_from_config(config, config_path)
        
        if model is None:
            return {
                "config_path": str(config_path),
                "experiment_name": exp_name,
                "model_type": model_type,
                "trainable_params": None,
                "total_params": None,
                "frozen_params": None,
                "status": "failed_to_build"
            }
        
        # Count parameters (only trainable matters for our use case)
        param_counts = count_parameters(model)
        
        print(f"  Model type: {model_type}")
        print(f"  Trainable params: {format_number(param_counts['trainable'])} ({param_counts['trainable']:,})")
        
        # For diffusion, clarify what we're counting
        if model_type == "Diffusion":
            print(f"    (UNet trainable parameters only)")
        
        return {
            "config_path": str(config_path),
            "experiment_name": exp_name,
            "model_type": model_type,
            "trainable_params": param_counts["trainable"],
            "total_params": param_counts["total"],
            "frozen_params": param_counts["frozen"],
            "status": "success"
        }
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "config_path": str(config_path),
            "experiment_name": config_path.stem,
            "model_type": "Unknown",
            "trainable_params": None,
            "total_params": None,
            "frozen_params": None,
            "status": f"error: {str(e)}"
        }


def find_config_files(config_dir: Path) -> List[Path]:
    """Find all YAML config files in directory, excluding base_config.yaml."""
    config_files = []
    for yaml_file in config_dir.rglob("*.yaml"):
        if yaml_file.name != "base_config.yaml":
            config_files.append(yaml_file)
    for yaml_file in config_dir.rglob("*.yml"):
        if yaml_file.name != "base_config.yml":
            config_files.append(yaml_file)
    return sorted(config_files)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze model sizes from experiment configs"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model_sizes.csv"),
        help="Output file path (CSV or JSON). Default: model_sizes.csv"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("experiments"),
        help="Directory containing experiment configs. Default: experiments/"
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "both"],
        default="csv",
        help="Output format. Default: csv"
    )
    
    args = parser.parse_args()
    
    # Find all config files
    config_dir = args.config_dir
    if not config_dir.exists():
        print(f"Error: Config directory not found: {config_dir}")
        return 1
    
    config_files = find_config_files(config_dir)
    print(f"Found {len(config_files)} config files in {config_dir}")
    
    if not config_files:
        print("No config files found!")
        return 1
    
    # Analyze each config
    results = []
    for config_file in config_files:
        result = analyze_config(config_file)
        if result:
            results.append(result)
    
    # Write results
    output_path = args.output
    output_format = args.format
    
    if output_format in ("csv", "both"):
        import pandas as pd
        df = pd.DataFrame(results)
        csv_path = output_path if output_path.suffix == ".csv" else output_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to {csv_path}")
    
    if output_format in ("json", "both"):
        json_path = output_path if output_path.suffix == ".json" else output_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to {json_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]
    
    print(f"Total configs analyzed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print("\nModel sizes (trainable parameters):")
        for result in sorted(successful, key=lambda x: x.get("trainable_params", 0) or 0, reverse=True):
            trainable = result.get("trainable_params")
            if trainable is not None:
                print(f"  {result['experiment_name']:50s} {format_number(trainable):>10s} ({trainable:,})")
    
    if failed:
        print("\nFailed configs:")
        for result in failed:
            print(f"  {result['experiment_name']:50s} {result['status']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

