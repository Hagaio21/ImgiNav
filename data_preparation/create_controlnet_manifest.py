#!/usr/bin/env python3
"""
Create ControlNet training manifest by aligning:
- Layout embeddings (target for diffusion)
- POV embeddings (condition 1)
- Graph embeddings (condition 2)

Handles the 1:many relationship: each layout has multiple POVs.
Creates one training sample per (layout, POV) pair, with the same graph embedding.

Usage:
    # Collect existing embeddings
    python data_preparation/create_controlnet_manifest.py \
        --layouts-manifest datasets/layouts.csv \
        --pov-embeddings-manifest datasets/povs_with_embeddings.csv \
        --graph-embeddings-manifest datasets/graphs_with_embeddings.csv \
        --output datasets/controlnet_training_manifest.csv
    
    # Create layout embeddings if missing
    python data_preparation/create_controlnet_manifest.py \
        --layouts-manifest datasets/layouts.csv \
        --pov-embeddings-manifest datasets/povs_with_embeddings.csv \
        --graph-embeddings-manifest datasets/graphs_with_embeddings.csv \
        --output datasets/controlnet_training_manifest.csv \
        --create-layout-embeddings \
        --autoencoder-config experiments/autoencoders/phase1/config.yaml \
        --autoencoder-checkpoint checkpoints/autoencoder.pt
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import subprocess

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def check_layout_embeddings_exist(layouts_manifest: Path) -> Tuple[bool, Optional[Path]]:
    """
    Check if layout embeddings already exist.
    
    Returns:
        (exists, manifest_path): Whether embeddings exist and path to manifest with latent_path
    """
    # Check if manifest has latent_path column
    try:
        df = pd.read_csv(layouts_manifest)
        if "latent_path" in df.columns:
            # Check if paths actually exist
            valid_paths = df["latent_path"].dropna()
            if len(valid_paths) > 0:
                # Check first few paths
                sample_paths = valid_paths.head(5)
                existing = sum(1 for p in sample_paths if Path(p).exists())
                if existing == len(sample_paths):
                    return True, layouts_manifest
    except Exception as e:
        print(f"Warning: Could not check layout embeddings: {e}")
    
    # Check for common naming patterns
    possible_names = [
        layouts_manifest.parent / "layouts_with_latents.csv",
        layouts_manifest.parent / "layouts_latents.csv",
        layouts_manifest.with_name(layouts_manifest.stem + "_latents.csv"),
    ]
    
    for path in possible_names:
        if path.exists():
            try:
                df = pd.read_csv(path)
                if "latent_path" in df.columns:
                    valid_paths = df["latent_path"].dropna()
                    if len(valid_paths) > 0:
                        sample_paths = valid_paths.head(5)
                        existing = sum(1 for p in sample_paths if Path(p).exists())
                        if existing == len(sample_paths):
                            return True, path
            except Exception:
                continue
    
    return False, None


def create_layout_embeddings(
    layouts_manifest: Path,
    autoencoder_config: Path,
    autoencoder_checkpoint: Path,
    output_manifest: Optional[Path] = None,
    batch_size: int = 32,
    num_workers: int = 8,
    device: str = "cuda"
) -> Path:
    """
    Create layout embeddings using create_embeddings.py script.
    
    Returns:
        Path to output manifest with latent_path column
    """
    if output_manifest is None:
        output_manifest = layouts_manifest.parent / f"{layouts_manifest.stem}_latents.csv"
    
    print(f"\nCreating layout embeddings...")
    print(f"  Input manifest: {layouts_manifest}")
    print(f"  Output manifest: {output_manifest}")
    print(f"  Autoencoder config: {autoencoder_config}")
    print(f"  Autoencoder checkpoint: {autoencoder_checkpoint}")
    
    # Call create_embeddings.py
    script_path = Path(__file__).parent / "create_embeddings.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--type", "layout",
        "--manifest", str(layouts_manifest),
        "--output-manifest", str(output_manifest),
        "--autoencoder-config", str(autoencoder_config),
        "--autoencoder-checkpoint", str(autoencoder_checkpoint),
        "--batch-size", str(batch_size),
        "--num-workers", str(num_workers),
        "--device", device
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create layout embeddings. Exit code: {result.returncode}")
    
    print(f"✓ Layout embeddings created: {output_manifest}")
    return output_manifest


def create_controlnet_manifest(
    layouts_manifest: Path,
    pov_embeddings_manifest: Path,
    graph_embeddings_manifest: Path,
    output_manifest: Path,
    layout_latent_col: str = "latent_path",
    pov_emb_col: str = "embedding_path",
    graph_emb_col: str = "embedding_path"
) -> None:
    """
    Create ControlNet training manifest by aligning layouts, POVs, and graphs.
    
    For each layout:
      - Get layout embedding (1 per layout)
      - Get graph embedding (1 per layout, 1:1)
      - For each POV associated with that layout:
          - Create training sample: (layout_emb, pov_emb, graph_emb)
    
    Args:
        layouts_manifest: Manifest with layout_path and latent_path columns
        pov_embeddings_manifest: Manifest with POV embeddings (embedding_path column)
        graph_embeddings_manifest: Manifest with graph embeddings (embedding_path column)
        output_manifest: Output path for training manifest
        layout_latent_col: Column name for layout embeddings in layouts manifest
        pov_emb_col: Column name for POV embeddings in POV manifest
        graph_emb_col: Column name for graph embeddings in graph manifest
    """
    print(f"\nLoading manifests...")
    
    # Load layouts manifest
    layouts_df = pd.read_csv(layouts_manifest)
    print(f"  Layouts: {len(layouts_df)} rows")
    
    # Load POV embeddings manifest
    pov_df = pd.read_csv(pov_embeddings_manifest)
    print(f"  POV embeddings: {len(pov_df)} rows")
    
    # Load graph embeddings manifest
    graph_df = pd.read_csv(graph_embeddings_manifest)
    print(f"  Graph embeddings: {len(graph_df)} rows")
    
    # Filter out rows with missing embeddings
    layouts_df = layouts_df.dropna(subset=[layout_latent_col])
    layouts_df = layouts_df[layouts_df[layout_latent_col] != ""]
    
    pov_df = pov_df.dropna(subset=[pov_emb_col])
    pov_df = pov_df[pov_df[pov_emb_col] != ""]
    
    graph_df = graph_df.dropna(subset=[graph_emb_col])
    graph_df = graph_df[graph_df[graph_emb_col] != ""]
    
    print(f"\nAfter filtering missing embeddings:")
    print(f"  Layouts: {len(layouts_df)} rows")
    print(f"  POV embeddings: {len(pov_df)} rows")
    print(f"  Graph embeddings: {len(graph_df)} rows")
    
    # Create training samples
    training_samples = []
    
    # Process room-level layouts
    room_layouts = layouts_df[layouts_df.get("type", "") == "room"].copy()
    if "type" not in room_layouts.columns or len(room_layouts) == 0:
        # Try to infer from room_id
        room_layouts = layouts_df[layouts_df.get("room_id", "") != "0000"].copy()
    
    print(f"\nProcessing room-level layouts ({len(room_layouts)} layouts)...")
    
    for _, layout_row in room_layouts.iterrows():
        scene_id = str(layout_row.get("scene_id", ""))
        room_id = str(layout_row.get("room_id", ""))
        layout_emb_path = layout_row[layout_latent_col]
        
        if not scene_id or not room_id or pd.isna(layout_emb_path) or layout_emb_path == "":
            continue
        
        # Find matching graph embedding (1:1 with layout)
        graph_match = graph_df[
            (graph_df.get("scene_id", "").astype(str) == scene_id) &
            (graph_df.get("room_id", "").astype(str) == room_id) &
            (graph_df.get("type", "").astype(str) == "room")
        ]
        
        if len(graph_match) == 0:
            # Try without type filter
            graph_match = graph_df[
                (graph_df.get("scene_id", "").astype(str) == scene_id) &
                (graph_df.get("room_id", "").astype(str) == room_id)
            ]
        
        if len(graph_match) == 0:
            continue
        
        graph_emb_path = graph_match.iloc[0][graph_emb_col]
        
        # Find all POVs for this room
        pov_matches = pov_df[
            (pov_df.get("scene_id", "").astype(str) == scene_id) &
            (pov_df.get("room_id", "").astype(str) == room_id)
        ]
        
        # Create one training sample per POV
        for _, pov_row in pov_matches.iterrows():
            pov_emb_path = pov_row[pov_emb_col]
            pov_type = str(pov_row.get("type", pov_row.get("pov_type", "")))
            viewpoint = str(pov_row.get("viewpoint", pov_row.get("view_id", "")))
            
            # Resolve paths to absolute
            try:
                layout_emb_abs = str(Path(layout_emb_path).resolve())
                pov_emb_abs = str(Path(pov_emb_path).resolve())
                graph_emb_abs = str(Path(graph_emb_path).resolve())
            except Exception as e:
                print(f"Warning: Could not resolve paths for {scene_id}_{room_id}: {e}")
                continue
            
            training_samples.append({
                "scene_id": scene_id,
                "room_id": room_id,
                "sample_type": "room",
                "pov_type": pov_type,
                "viewpoint": viewpoint,
                "layout_embedding": layout_emb_abs,
                "pov_embedding": pov_emb_abs,
                "graph_embedding": graph_emb_abs,
            })
    
    # Process scene-level layouts
    scene_layouts = layouts_df[layouts_df.get("type", "").astype(str) == "scene"].copy()
    if "type" not in scene_layouts.columns or len(scene_layouts) == 0:
        scene_layouts = layouts_df[layouts_df.get("room_id", "").astype(str).isin(["0000", "scene", ""])].copy()
    
    print(f"\nProcessing scene-level layouts ({len(scene_layouts)} layouts)...")
    
    for _, layout_row in scene_layouts.iterrows():
        scene_id = str(layout_row.get("scene_id", ""))
        layout_emb_path = layout_row[layout_latent_col]
        
        if not scene_id or pd.isna(layout_emb_path) or layout_emb_path == "":
            continue
        
        # Find matching graph embedding (1:1 with layout)
        graph_match = graph_df[
            (graph_df.get("scene_id", "").astype(str) == scene_id) &
            (graph_df.get("type", "").astype(str) == "scene")
        ]
        
        if len(graph_match) == 0:
            # Try with room_id == "0000" or "scene"
            graph_match = graph_df[
                (graph_df.get("scene_id", "").astype(str) == scene_id) &
                (graph_df.get("room_id", "").astype(str).isin(["0000", "scene", ""]))
            ]
        
        if len(graph_match) == 0:
            continue
        
        graph_emb_path = graph_match.iloc[0][graph_emb_col]
        
        # For scene layouts, we still need POVs - but they might be from any room in the scene
        # Or we might not have POVs for scene layouts
        # For now, create one sample per scene layout with its graph embedding
        # (POV embedding can be empty or we can use a placeholder)
        
        try:
            layout_emb_abs = str(Path(layout_emb_path).resolve())
            graph_emb_abs = str(Path(graph_emb_path).resolve())
        except Exception as e:
            print(f"Warning: Could not resolve paths for scene {scene_id}: {e}")
            continue
        
        training_samples.append({
            "scene_id": scene_id,
            "room_id": "0000",
            "sample_type": "scene",
            "pov_type": "",
            "viewpoint": "",
            "layout_embedding": layout_emb_abs,
            "pov_embedding": "",  # Scene layouts might not have POVs
            "graph_embedding": graph_emb_abs,
        })
    
    # Create output DataFrame
    training_df = pd.DataFrame(training_samples)
    
    if len(training_df) == 0:
        print("\n⚠️  No training samples created! Check your manifests.")
        return
    
    # Filter out samples with missing embeddings
    training_df = training_df[
        (training_df["layout_embedding"] != "") &
        (training_df["graph_embedding"] != "")
    ]
    
    # For room samples, require POV embedding
    room_samples = training_df[training_df["sample_type"] == "room"].copy()
    room_samples = room_samples[room_samples["pov_embedding"] != ""]
    
    # Combine room and scene samples
    scene_samples = training_df[training_df["sample_type"] == "scene"].copy()
    training_df = pd.concat([room_samples, scene_samples], ignore_index=True)
    
    print(f"\nCreated {len(training_df)} training samples")
    print(f"  Room samples: {len(room_samples)}")
    print(f"  Scene samples: {len(scene_samples)}")
    
    # Save manifest
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(output_manifest, index=False)
    
    print(f"\n✓ Training manifest saved: {output_manifest}")
    
    # Print statistics
    if len(room_samples) > 0:
        unique_room_layouts = room_samples[["scene_id", "room_id"]].drop_duplicates()
        print(f"\nStatistics:")
        print(f"  Unique room layouts: {len(unique_room_layouts)}")
        print(f"  Unique POVs: {len(room_samples)}")
        print(f"  Average POVs per room layout: {len(room_samples) / max(1, len(unique_room_layouts)):.2f}")
    
    if len(scene_samples) > 0:
        print(f"  Scene layouts: {len(scene_samples)}")


def main():
    parser = argparse.ArgumentParser(
        description="Create ControlNet training manifest by aligning layout, POV, and graph embeddings"
    )
    
    # Required arguments
    parser.add_argument(
        "--layouts-manifest",
        type=str,
        required=True,
        help="Path to layouts manifest CSV (with layout_path column)"
    )
    parser.add_argument(
        "--pov-embeddings-manifest",
        type=str,
        required=True,
        help="Path to POV embeddings manifest CSV (with embedding_path column)"
    )
    parser.add_argument(
        "--graph-embeddings-manifest",
        type=str,
        required=True,
        help="Path to graph embeddings manifest CSV (with embedding_path column)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for ControlNet training manifest CSV"
    )
    
    # Optional: Create layout embeddings if missing
    parser.add_argument(
        "--create-layout-embeddings",
        action="store_true",
        help="Create layout embeddings if they don't exist"
    )
    parser.add_argument(
        "--autoencoder-config",
        type=str,
        help="Path to autoencoder config YAML (required if --create-layout-embeddings)"
    )
    parser.add_argument(
        "--autoencoder-checkpoint",
        type=str,
        help="Path to autoencoder checkpoint (required if --create-layout-embeddings)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for creating embeddings (default: 32)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of workers for creating embeddings (default: 8)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for creating embeddings (default: cuda)"
    )
    
    args = parser.parse_args()
    
    layouts_manifest = Path(args.layouts_manifest)
    pov_embeddings_manifest = Path(args.pov_embeddings_manifest)
    graph_embeddings_manifest = Path(args.graph_embeddings_manifest)
    output_manifest = Path(args.output)
    
    # Check if layout embeddings exist
    embeddings_exist, layouts_with_embeddings = check_layout_embeddings_exist(layouts_manifest)
    
    if not embeddings_exist:
        if args.create_layout_embeddings:
            if not args.autoencoder_config or not args.autoencoder_checkpoint:
                parser.error("--autoencoder-config and --autoencoder-checkpoint are required when --create-layout-embeddings is set")
            
            layouts_with_embeddings = create_layout_embeddings(
                layouts_manifest=layouts_manifest,
                autoencoder_config=Path(args.autoencoder_config),
                autoencoder_checkpoint=Path(args.autoencoder_checkpoint),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device
            )
        else:
            print("\n⚠️  Layout embeddings not found!")
            print(f"   Layouts manifest: {layouts_manifest}")
            print("\n   Options:")
            print("   1. Create layout embeddings first using:")
            print(f"      python data_preparation/create_embeddings.py --type layout \\")
            print(f"        --manifest {layouts_manifest} \\")
            print(f"        --output-manifest {layouts_manifest.parent}/{layouts_manifest.stem}_latents.csv \\")
            print(f"        --autoencoder-config <config.yaml> \\")
            print(f"        --autoencoder-checkpoint <checkpoint.pt>")
            print("\n   2. Or use --create-layout-embeddings flag with --autoencoder-config and --autoencoder-checkpoint")
            sys.exit(1)
    else:
        print(f"\n✓ Found existing layout embeddings: {layouts_with_embeddings}")
    
    # Create training manifest
    create_controlnet_manifest(
        layouts_manifest=layouts_with_embeddings,
        pov_embeddings_manifest=pov_embeddings_manifest,
        graph_embeddings_manifest=graph_embeddings_manifest,
        output_manifest=output_manifest
    )
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()

