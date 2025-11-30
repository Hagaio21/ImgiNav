#!/usr/bin/env python3
"""
Generate diffusion experiment configs from template.

This script generates all diffusion experiment configurations based on:
- Variant: seg/tex (2 options)
- Scope: rooms/scenes/both (3 options)
- Conditioning: graph/povs/both (3 options)
- Architecture: small/medium with bottleneck+down (2 options)

Total: 2 × 3 × 3 × 2 = 36 experiments

All seg experiments use the same VAE manifest (manifest_seg.csv)
All tex experiments use the same VAE manifest (manifest_tex.csv)
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional

# Base paths (can be overridden via environment variables)
BASE_DIR = Path(__file__).parent
TEMPLATE_PATH = BASE_DIR / "template_diffusion.yaml"
# Output directory for generated configs (v2 structure)
OUTPUT_BASE_DIR = BASE_DIR.parent / "v2"

# Default paths (will be replaced based on environment)
DEFAULT_PATHS = {
    "local": {
        "manifest_seg": "dataset_v2/manifests/manifest_seg.csv",
        "manifest_tex": "dataset_v2/manifests/manifest_tex.csv",
        "vae_seg": "checkpoints/vae_seg_checkpoint_best.pt",
        "vae_tex": "checkpoints/vae_tex_checkpoint_best.pt",
        "experiments_base": "outputs/diffusion",
    },
    "hpc": {
        "manifest_seg": "/work3/s233249/ImgiNav/dataset_v2/manifests/manifest_seg.csv",
        "manifest_tex": "/work3/s233249/ImgiNav/dataset_v2/manifests/manifest_tex.csv",
        "vae_seg": "/work3/s233249/ImgiNav/checkpoints/vae_seg_checkpoint_best.pt",
        "vae_tex": "/work3/s233249/ImgiNav/checkpoints/vae_tex_checkpoint_best.pt",
        "experiments_base": "/work3/s233249/ImgiNav/experiments/diffusion",
    }
}

# Architecture configurations
ARCHITECTURES = {
    "small": {
        "base_channels": 48,
        "depth": 3,
        "attention_heads": 2,
        "batch_size": 48,
    },
    "medium": {
        "base_channels": 64,
        "depth": 4,
        "attention_heads": 4,
        "batch_size": 48,
    }
}

# Experiment structure
VARIANTS = ["seg", "tex"]
SCOPES = ["rooms", "scenes", "both"]
CONDITIONINGS = ["graph", "povs", "both"]
ARCH_SIZES = ["small", "medium"]


def detect_environment() -> str:
    """Detect if running on HPC or local."""
    if os.getenv("IMGINAV_ENV") == "hpc":
        return "hpc"
    if os.getenv("HOSTNAME") and "compute" in os.getenv("HOSTNAME", "").lower():
        return "hpc"
    return "local"


def get_paths(env: str) -> Dict[str, str]:
    """Get paths for the given environment."""
    return DEFAULT_PATHS.get(env, DEFAULT_PATHS["local"])


def get_manifest_path(variant: str, env: str) -> str:
    """Get manifest path for variant."""
    paths = get_paths(env)
    key = f"manifest_{variant}"
    return paths.get(key, paths["manifest_seg"] if variant == "seg" else paths["manifest_tex"])


def get_vae_checkpoint(variant: str, env: str) -> str:
    """Get VAE checkpoint path for variant."""
    paths = get_paths(env)
    key = f"vae_{variant}"
    return paths.get(key, paths["vae_seg"] if variant == "seg" else paths["vae_tex"])


def get_type_filter(scope: str) -> str:
    """Get type filter YAML string for scope."""
    if scope == "rooms":
        return "    type:\n    - room"
    elif scope == "scenes":
        return "    type:\n    - scene"
    else:  # both
        return "    type: []"


def get_embedding_outputs(conditioning: str) -> str:
    """Get embedding outputs YAML string for conditioning."""
    if conditioning == "graph":
        return "    text_emb: graph_embedding_path"
    elif conditioning == "povs":
        return "    pov_emb: pov_embedding_path"
    else:  # both
        return "    text_emb: graph_embedding_path\n    pov_emb: pov_embedding_path"


def generate_experiment_name(variant: str, scope: str, conditioning: str, arch_size: str) -> str:
    """Generate experiment name."""
    parts = ["diff_clip", variant, scope, arch_size, "down_bottleneck"]
    if conditioning != "both":
        parts.append(conditioning)
    return "_".join(parts)


def generate_save_path(variant: str, scope: str, conditioning: str, arch_size: str, env: str) -> str:
    """Generate save path for experiment."""
    paths = get_paths(env)
    base = paths["experiments_base"]
    exp_name = generate_experiment_name(variant, scope, conditioning, arch_size)
    return f"{base}/{exp_name}"


def load_template() -> str:
    """Load template file."""
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Template not found: {TEMPLATE_PATH}")
    return TEMPLATE_PATH.read_text()


def generate_config(
    variant: str,
    scope: str,
    conditioning: str,
    arch_size: str,
    env: str = "local"
) -> Dict:
    """Generate a single config dictionary."""
    template = load_template()
    
    # Get architecture config
    arch_config = ARCHITECTURES[arch_size]
    
    # Generate experiment name and paths
    exp_name = generate_experiment_name(variant, scope, conditioning, arch_size)
    save_path = generate_save_path(variant, scope, conditioning, arch_size, env)
    manifest_path = get_manifest_path(variant, env)
    vae_checkpoint = get_vae_checkpoint(variant, env)
    
    # Get filters and embeddings
    type_filter = get_type_filter(scope)
    embedding_outputs = get_embedding_outputs(conditioning)
    
    # Replace placeholders
    config_str = template.replace("{EXPERIMENT_NAME}", exp_name)
    config_str = config_str.replace("{SAVE_PATH}", save_path)
    config_str = config_str.replace("{MANIFEST_PATH}", manifest_path)
    config_str = config_str.replace("{VAE_CHECKPOINT}", vae_checkpoint)
    config_str = config_str.replace("{EMBEDDING_OUTPUTS}", embedding_outputs)
    config_str = config_str.replace("{TYPE_FILTER}", type_filter)
    config_str = config_str.replace("{BASE_CHANNELS}", str(arch_config["base_channels"]))
    config_str = config_str.replace("{DEPTH}", str(arch_config["depth"]))
    config_str = config_str.replace("{ATTENTION_HEADS}", str(arch_config["attention_heads"]))
    config_str = config_str.replace("{BATCH_SIZE}", str(arch_config["batch_size"]))
    
    # Parse as YAML
    return yaml.safe_load(config_str)


def save_config(config: Dict, output_path: Path):
    """Save config to YAML file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def generate_all_configs(output_dir: Optional[Path] = None, env: str = "hpc", dry_run: bool = False):
    """Generate all experiment configs.
    
    Note: Configs always use HPC paths since experiments are run on HPC.
    The env parameter is kept for backward compatibility but defaults to 'hpc'.
    """
    if output_dir is None:
        output_dir = OUTPUT_BASE_DIR
    
    # Always use HPC paths for generated configs
    config_env = "hpc"
    print(f"Generating configs with HPC paths (env parameter '{env}' ignored for config generation)")
    print(f"Output directory: {output_dir}")
    print()
    
    total = 0
    generated = []
    
    for variant in VARIANTS:
        for scope in SCOPES:
            for conditioning in CONDITIONINGS:
                for arch_size in ARCH_SIZES:
                    total += 1
                    
                    # Generate config (always use HPC paths)
                    config = generate_config(variant, scope, conditioning, arch_size, config_env)
                    
                    # Determine output path
                    exp_name = generate_experiment_name(variant, scope, conditioning, arch_size)
                    output_path = output_dir / variant / scope / conditioning / f"{arch_size}_down_bottleneck.yaml"
                    
                    if not dry_run:
                        save_config(config, output_path)
                    
                    # Calculate relative path from workspace root or output_dir
                    try:
                        rel_path = output_path.relative_to(Path.cwd())
                    except ValueError:
                        rel_path = output_path.relative_to(output_dir)
                    
                    generated.append({
                        "variant": variant,
                        "scope": scope,
                        "conditioning": conditioning,
                        "arch_size": arch_size,
                        "name": exp_name,
                        "path": str(rel_path),
                    })
    
    # Print summary
    print(f"Generated {total} experiment configs:")
    print()
    for exp in generated:
        print(f"  {exp['path']}")
        print(f"    Name: {exp['name']}")
        print()
    
    print(f"\nTotal experiments: {total}")
    print(f"\nBreakdown:")
    print(f"  Variants (seg/tex): {len(VARIANTS)}")
    print(f"  Scopes (rooms/scenes/both): {len(SCOPES)}")
    print(f"  Conditionings (graph/povs/both): {len(CONDITIONINGS)}")
    print(f"  Architectures (small/medium): {len(ARCH_SIZES)}")
    print(f"  Total: {len(VARIANTS)} × {len(SCOPES)} × {len(CONDITIONINGS)} × {len(ARCH_SIZES)} = {total}")
    
    return generated


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate diffusion experiment configs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as script directory)"
    )
    parser.add_argument(
        "--env",
        choices=["local", "hpc", "auto"],
        default="hpc",
        help="Environment (ignored - configs always use HPC paths). Kept for backward compatibility."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without creating files"
    )
    
    args = parser.parse_args()
    
    generate_all_configs(
        output_dir=args.output_dir,
        env=args.env,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()

