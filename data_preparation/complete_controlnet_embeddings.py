#!/usr/bin/env python3
"""
Complete missing embeddings in ControlNet manifest.

This script:
1. Reads an existing ControlNet manifest
2. Identifies missing embeddings (POV, graph, layout)
3. Creates embeddings for missing items
4. Updates the manifest with embedding paths

Usage:
    python complete_controlnet_embeddings.py \
        --manifest path/to/controlnet_manifest.csv \
        --output path/to/controlnet_manifest_complete.csv \
        --taxonomy config/taxonomy.json \
        --autoencoder-config config/autoencoder.yaml \
        --autoencoder-checkpoint checkpoints/autoencoder.pt \
        [--overwrite] [--device cuda] [--batch-size 32] [--num-workers 8]
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict

import torch
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent))

from common.file_io import read_manifest, create_manifest
from common.taxonomy import Taxonomy
from data_preparation.create_embeddings import (
    load_autoencoder_model,
    create_pov_embeddings,
    create_graph_embeddings,
    create_layout_embeddings_from_manifest
)


def check_embedding_exists(embedding_path: str) -> bool:
    """Check if embedding file exists and is valid."""
    if not embedding_path or embedding_path == "":
        return False
    
    path = Path(embedding_path)
    if not path.exists():
        return False
    
    # Try to load the embedding to verify it's valid
    try:
        if path.suffix == ".pt":
            torch.load(path, map_location="cpu")
        elif path.suffix == ".npy":
            import numpy as np
            np.load(path)
        return True
    except Exception as e:
        print(f"Warning: Invalid embedding file {path}: {e}")
        return False


def identify_missing_embeddings(manifest_rows: List[Dict]) -> Dict[str, Set[str]]:
    """
    Identify which embeddings are missing.
    Returns dict with keys: 'pov', 'graph', 'layout' and sets of paths that need embedding.
    """
    missing = {
        'pov': set(),
        'graph': set(),
        'layout': set()
    }
    
    for row in manifest_rows:
        # Check POV embeddings
        pov_path = row.get("pov_path", "")
        pov_emb_path = row.get("pov_embedding_path", "")
        if pov_path and pov_path != "0" and Path(pov_path).exists():
            if not check_embedding_exists(pov_emb_path):
                missing['pov'].add(pov_path)
        
        # Check graph embeddings
        graph_json_path = row.get("graph_json_path", "")
        graph_emb_path = row.get("graph_embedding_path", "")
        if graph_json_path and Path(graph_json_path).exists():
            if not check_embedding_exists(graph_emb_path):
                missing['graph'].add(graph_json_path)
        
        # Check layout embeddings
        layout_path = row.get("layout_path", "")
        layout_emb_path = row.get("layout_embedding_path", "")
        if layout_path and Path(layout_path).exists():
            if not check_embedding_exists(layout_emb_path):
                missing['layout'].add(layout_path)
    
    return missing


def create_missing_pov_embeddings(
    pov_paths: Set[str],
    output_dir: Path,
    batch_size: int = 32,
    overwrite: bool = False
) -> Dict[str, str]:
    """
    Create embeddings for missing POVs.
    Returns mapping: pov_path -> embedding_path
    """
    if not pov_paths:
        return {}
    
    print(f"\n[1/3] Creating POV embeddings for {len(pov_paths)} missing items...")
    
    # Create temporary manifest
    temp_dir = output_dir / "temp_embeddings"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    pov_rows = [
        {"pov_path": path, "is_empty": 0}
        for path in pov_paths
        if Path(path).exists()
    ]
    
    if not pov_rows:
        print("  No valid POV paths found")
        return {}
    
    pov_manifest_temp = temp_dir / "povs_temp.csv"
    create_manifest(pov_rows, pov_manifest_temp, ["pov_path", "is_empty"])
    
    pov_manifest_out = temp_dir / "povs_with_embeddings.csv"
    create_pov_embeddings(
        manifest_path=pov_manifest_temp,
        output_manifest=pov_manifest_out,
        save_format="pt",
        batch_size=batch_size
    )
    
    # Read mapping
    pov_emb_df = pd.read_csv(pov_manifest_out)
    pov_emb_mapping = dict(zip(pov_emb_df["pov_path"], pov_emb_df["embedding_path"]))
    
    print(f"  ✓ Created {len(pov_emb_mapping)} POV embeddings")
    return pov_emb_mapping


def create_missing_graph_embeddings(
    graph_paths: Set[str],
    taxonomy_path: Path,
    output_dir: Path,
    batch_size: int = 32,
    overwrite: bool = False
) -> Dict[str, str]:
    """
    Create embeddings for missing graphs.
    Returns mapping: graph_json_path -> embedding_path
    """
    if not graph_paths:
        return {}
    
    print(f"\n[2/3] Creating graph embeddings for {len(graph_paths)} missing items...")
    
    # Create temporary manifest
    temp_dir = output_dir / "temp_embeddings"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    graph_rows = [
        {"graph_path": path}
        for path in graph_paths
        if Path(path).exists()
    ]
    
    if not graph_rows:
        print("  No valid graph paths found")
        return {}
    
    graph_manifest_temp = temp_dir / "graphs_temp.csv"
    create_manifest(graph_rows, graph_manifest_temp, ["graph_path"])
    
    graph_manifest_out = temp_dir / "graphs_with_embeddings.csv"
    create_graph_embeddings(
        manifest_path=graph_manifest_temp,
        taxonomy_path=taxonomy_path,
        output_manifest=graph_manifest_out,
        model_name="all-MiniLM-L6-v2",
        save_format="pt"
    )
    
    # Read mapping
    graph_emb_df = pd.read_csv(graph_manifest_out)
    graph_emb_mapping = dict(zip(graph_emb_df["graph_path"], graph_emb_df["embedding_path"]))
    
    print(f"  ✓ Created {len(graph_emb_mapping)} graph embeddings")
    return graph_emb_mapping


def create_missing_layout_embeddings(
    layout_paths: Set[str],
    autoencoder_config: Optional[Path],
    autoencoder_checkpoint: Path,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 32,
    num_workers: int = 8,
    overwrite: bool = False
) -> Dict[str, str]:
    """
    Create embeddings for missing layouts.
    Returns mapping: layout_path -> latent_path
    """
    if not layout_paths:
        return {}
    
    print(f"\n[3/3] Creating layout embeddings for {len(layout_paths)} missing items...")
    
    # Create temporary manifest
    temp_dir = output_dir / "temp_embeddings"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    layout_rows = [
        {"layout_path": path, "is_empty": 0}
        for path in layout_paths
        if Path(path).exists()
    ]
    
    if not layout_rows:
        print("  No valid layout paths found")
        return {}
    
    layout_manifest_temp = temp_dir / "layouts_temp.csv"
    create_manifest(layout_rows, layout_manifest_temp, ["layout_path", "is_empty"])
    
    layout_manifest_out = temp_dir / "layouts_with_embeddings.csv"
    
    # Load autoencoder
    if autoencoder_config and autoencoder_config.exists():
        model = load_autoencoder_model(
            str(autoencoder_config),
            str(autoencoder_checkpoint),
            device=device
        )
        config_path = str(autoencoder_config)
    else:
        # Load from checkpoint only (config is embedded in checkpoint)
        from models.autoencoder import Autoencoder
        print(f"[INFO] Loading autoencoder from checkpoint (config embedded): {autoencoder_checkpoint}")
        model = Autoencoder.load_checkpoint(str(autoencoder_checkpoint), map_location=device)
        model = model.to(device)
        model.eval()
        # Try to find config from checkpoint path
        checkpoint_path = Path(autoencoder_checkpoint)
        # Remove _checkpoint_best or _checkpoint_latest suffix to get base name
        base_name = checkpoint_path.stem
        for suffix in ["_checkpoint_best", "_checkpoint_latest", "_checkpoint"]:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
        
        possible_configs = [
            # Same directory as checkpoint
            checkpoint_path.parent / f"{base_name}.yaml",
            checkpoint_path.parent / "experiment_config.yaml",
            checkpoint_path.parent / "autoencoder_config.yaml",
            # Parent directory
            checkpoint_path.parent.parent / "experiment_config.yaml",
            # Check in experiments/autoencoders structure
            checkpoint_path.parent.parent.parent / "autoencoders" / checkpoint_path.parent.name / f"{base_name}.yaml",
        ]
        
        config_path = None
        for pc in possible_configs:
            if pc.exists():
                config_path = str(pc)
                print(f"[INFO] Found config file: {config_path}")
                break
        
        if not config_path:
            print("[WARNING] Could not find config file, using defaults for transform")
            print(f"[INFO] Checked locations:")
            for pc in possible_configs:
                print(f"  - {pc} {'(exists)' if pc.exists() else '(not found)'}")
            config_path = None
    
    create_layout_embeddings_from_manifest(
        encoder=model.encoder,
        manifest_path=layout_manifest_temp,
        output_manifest_path=layout_manifest_out,
        batch_size=batch_size,
        num_workers=num_workers,
        overwrite=overwrite,
        device=device,
        autoencoder_config_path=config_path,
        output_latent_dir=None,
        diffusion_config_path=None
    )
    
    # Read mapping
    layout_emb_df = pd.read_csv(layout_manifest_out)
    layout_emb_mapping = dict(zip(layout_emb_df["layout_path"], layout_emb_df["latent_path"]))
    
    print(f"  ✓ Created {len(layout_emb_mapping)} layout embeddings")
    return layout_emb_mapping


def update_manifest_with_embeddings(
    manifest_rows: List[Dict],
    pov_mapping: Dict[str, str],
    graph_mapping: Dict[str, str],
    layout_mapping: Dict[str, str]
) -> List[Dict]:
    """Update manifest rows with embedding paths."""
    updated_rows = []
    
    for row in manifest_rows:
        updated_row = row.copy()
        
        # Update POV embedding
        pov_path = row.get("pov_path", "")
        if pov_path and pov_path != "0":
            if pov_path in pov_mapping:
                updated_row["pov_embedding_path"] = pov_mapping[pov_path]
            elif not check_embedding_exists(row.get("pov_embedding_path", "")):
                updated_row["pov_embedding_path"] = ""
        else:
            updated_row["pov_embedding_path"] = ""
        
        # Update graph embedding
        graph_json_path = row.get("graph_json_path", "")
        if graph_json_path:
            if graph_json_path in graph_mapping:
                updated_row["graph_embedding_path"] = graph_mapping[graph_json_path]
            elif not check_embedding_exists(row.get("graph_embedding_path", "")):
                updated_row["graph_embedding_path"] = ""
        else:
            updated_row["graph_embedding_path"] = ""
        
        # Update layout embedding
        layout_path = row.get("layout_path", "")
        if layout_path:
            if layout_path in layout_mapping:
                updated_row["layout_embedding_path"] = layout_mapping[layout_path]
            elif not check_embedding_exists(row.get("layout_embedding_path", "")):
                updated_row["layout_embedding_path"] = ""
        else:
            updated_row["layout_embedding_path"] = ""
        
        updated_rows.append(updated_row)
    
    return updated_rows


def main():
    parser = argparse.ArgumentParser(
        description="Complete missing embeddings in ControlNet manifest"
    )
    
    # Input/output
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to input ControlNet manifest CSV"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to output manifest CSV (with completed embeddings)"
    )
    
    # Required for embeddings
    parser.add_argument(
        "--taxonomy",
        required=True,
        type=Path,
        help="Path to taxonomy.json (required for graph embeddings)"
    )
    parser.add_argument(
        "--autoencoder-checkpoint",
        required=True,
        type=Path,
        help="Path to autoencoder checkpoint (required for layout embeddings)"
    )
    parser.add_argument(
        "--autoencoder-config",
        type=Path,
        help="Path to autoencoder config YAML (optional, will try to infer from checkpoint)"
    )
    
    # Optional parameters
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding creation"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing embedding files"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.manifest.exists():
        parser.error(f"Manifest not found: {args.manifest}")
    
    if not args.taxonomy.exists():
        parser.error(f"Taxonomy file not found: {args.taxonomy}")
    
    if not args.autoencoder_checkpoint.exists():
        parser.error(f"Autoencoder checkpoint not found: {args.autoencoder_checkpoint}")
    
    # Read manifest
    print(f"Reading manifest: {args.manifest}")
    manifest_rows = read_manifest(args.manifest)
    print(f"  Found {len(manifest_rows)} data points")
    
    # Identify missing embeddings
    print("\nIdentifying missing embeddings...")
    missing = identify_missing_embeddings(manifest_rows)
    
    total_missing = sum(len(paths) for paths in missing.values())
    print(f"  Missing POV embeddings: {len(missing['pov'])}")
    print(f"  Missing graph embeddings: {len(missing['graph'])}")
    print(f"  Missing layout embeddings: {len(missing['layout'])}")
    print(f"  Total missing: {total_missing}")
    
    if total_missing == 0:
        print("\n✓ All embeddings are complete!")
        # Still write output manifest (may have path updates)
        output_dir = args.output.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fieldnames = list(manifest_rows[0].keys()) if manifest_rows else []
        create_manifest(manifest_rows, args.output, fieldnames)
        print(f"  Output manifest: {args.output}")
        return
    
    # Setup output directory
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create missing embeddings
    pov_mapping = {}
    graph_mapping = {}
    layout_mapping = {}
    
    if missing['pov']:
        pov_mapping = create_missing_pov_embeddings(
            missing['pov'],
            output_dir,
            batch_size=args.batch_size,
            overwrite=args.overwrite
        )
    
    if missing['graph']:
        graph_mapping = create_missing_graph_embeddings(
            missing['graph'],
            args.taxonomy,
            output_dir,
            batch_size=args.batch_size,
            overwrite=args.overwrite
        )
    
    if missing['layout']:
        layout_mapping = create_missing_layout_embeddings(
            missing['layout'],
            args.autoencoder_config,
            args.autoencoder_checkpoint,
            output_dir,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            overwrite=args.overwrite
        )
    
    # Update manifest
    print("\nUpdating manifest with embedding paths...")
    updated_rows = update_manifest_with_embeddings(
        manifest_rows,
        pov_mapping,
        graph_mapping,
        layout_mapping
    )
    
    # Write output manifest
    fieldnames = list(updated_rows[0].keys()) if updated_rows else []
    create_manifest(updated_rows, args.output, fieldnames)
    
    # Clean up temp directory
    temp_dir = output_dir / "temp_embeddings"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    # Summary
    print(f"\n✓ Embedding completion finished!")
    print(f"  Created POV embeddings: {len(pov_mapping)}")
    print(f"  Created graph embeddings: {len(graph_mapping)}")
    print(f"  Created layout embeddings: {len(layout_mapping)}")
    print(f"  Output manifest: {args.output}")


if __name__ == "__main__":
    main()

