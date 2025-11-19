#!/usr/bin/env python3
"""
Create ControlNet dataset from layouts, POVs, and graphs manifests.

This script:
1. Copies files to controlnet dataset directories
2. Converts graphs to text using graph2text
3. Creates data points: (pov, graph, layout) for rooms, (0, graph, layout) for scenes
4. Creates embeddings for each data point
5. Updates manifest with embedding paths
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent))

from common.file_io import read_manifest, create_manifest
from common.taxonomy import Taxonomy
from data_preparation.utils.text_utils import graph2text
from data_preparation.create_embeddings import (
    load_autoencoder_model,
    load_resnet_model,
    load_sentence_transformer_model,
    create_pov_embeddings,
    create_graph_embeddings,
    create_layout_embeddings_from_manifest
)


def copy_file(src: Path, dst: Path, overwrite: bool = False) -> bool:
    """Copy a file, creating parent directories if needed."""
    if dst.exists() and not overwrite:
        return True
    
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        print(f"Error copying {src} to {dst}: {e}")
        return False


def setup_directories(base_dir: Path) -> Dict[str, Path]:
    """Create and return dictionary of dataset directories."""
    dirs = {
        "graphs_json": base_dir / "controlnet" / "graphs" / "json",
        "graphs_text": base_dir / "controlnet" / "graphs" / "text",
        "pov_tex": base_dir / "controlnet" / "pov" / "tex",
        "pov_seg": base_dir / "controlnet" / "pov" / "seg",
        "layouts": base_dir / "layouts",
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def copy_graphs_to_dataset(graphs_manifest: Path, output_dirs: Dict[str, Path], 
                           taxonomy: Taxonomy, overwrite: bool = False) -> Dict[str, str]:
    """
    Copy graph JSON files and convert to text.
    Returns mapping: original_graph_path -> (json_path, text_path)
    """
    print("Processing graphs...")
    rows = read_manifest(graphs_manifest)
    
    graph_mapping = {}
    skipped = 0
    
    for row in tqdm(rows, desc="Copying and converting graphs"):
        graph_path_str = row.get("graph_path", "")
        if not graph_path_str:
            skipped += 1
            continue
        
        graph_path = Path(graph_path_str)
        if not graph_path.exists():
            print(f"Warning: Graph file not found: {graph_path}")
            skipped += 1
            continue
        
        # Determine output filename
        # Use scene_id and room_id to create unique filename
        scene_id = row.get("scene_id", "")
        room_id = row.get("room_id", "")
        graph_type = row.get("type", "room")
        
        if graph_type == "scene":
            filename = f"{scene_id}_scene_graph.json"
        else:
            filename = f"{scene_id}_{room_id}_room_graph.json"
        
        # Copy JSON
        json_dst = output_dirs["graphs_json"] / filename
        if copy_file(graph_path, json_dst, overwrite):
            # Convert to text using the copied JSON (for consistency)
            try:
                text = graph2text(json_dst, taxonomy)
                if text:
                    text_dst = output_dirs["graphs_text"] / filename.replace(".json", ".txt")
                    text_dst.write_text(text, encoding="utf-8")
                    graph_mapping[graph_path_str] = (str(json_dst), str(text_dst))
                else:
                    print(f"Warning: Empty text for {graph_path}")
                    skipped += 1
            except Exception as e:
                print(f"Error converting graph {graph_path} to text: {e}")
                skipped += 1
    
    print(f"✓ Processed {len(graph_mapping)} graphs, skipped {skipped}")
    return graph_mapping


def copy_povs_to_dataset(povs_manifest: Path, output_dirs: Dict[str, Path], 
                         overwrite: bool = False) -> Dict[Tuple[str, str], str]:
    """
    Copy POV files to appropriate directories (tex/seg).
    Returns mapping: (scene_id, room_id, type) -> pov_path
    """
    print("Processing POVs...")
    rows = read_manifest(povs_manifest)
    
    pov_mapping = {}
    skipped = 0
    
    for row in tqdm(rows, desc="Copying POVs"):
        pov_path_str = row.get("pov_path", "")
        if not pov_path_str:
            skipped += 1
            continue
        
        pov_path = Path(pov_path_str)
        if not pov_path.exists():
            print(f"Warning: POV file not found: {pov_path}")
            skipped += 1
            continue
        
        # Check if empty
        is_empty = int(row.get("is_empty", 0))
        if is_empty:
            skipped += 1
            continue
        
        scene_id = row.get("scene_id", "")
        room_id = row.get("room_id", "")
        pov_type = row.get("type", "")
        
        # Determine output directory
        if pov_type == "tex":
            output_dir = output_dirs["pov_tex"]
        elif pov_type == "seg":
            output_dir = output_dirs["pov_seg"]
        else:
            print(f"Warning: Unknown POV type '{pov_type}', skipping")
            skipped += 1
            continue
        
        # Create unique filename
        filename = pov_path.name
        dst = output_dir / filename
        
        if copy_file(pov_path, dst, overwrite):
            key = (scene_id, room_id, pov_type)
            pov_mapping[key] = str(dst)
    
    print(f"✓ Processed {len(pov_mapping)} POVs, skipped {skipped}")
    return pov_mapping


def copy_layouts_to_dataset(layouts_manifest: Path, output_dirs: Dict[str, Path],
                           overwrite: bool = False) -> Dict[Tuple[str, str], str]:
    """
    Copy layout files to dataset directory.
    Returns mapping: (scene_id, room_id, type) -> layout_path
    """
    print("Processing layouts...")
    rows = read_manifest(layouts_manifest)
    
    layout_mapping = {}
    skipped = 0
    
    for row in tqdm(rows, desc="Copying layouts"):
        layout_path_str = row.get("layout_path", "")
        if not layout_path_str:
            skipped += 1
            continue
        
        layout_path = Path(layout_path_str)
        if not layout_path.exists():
            print(f"Warning: Layout file not found: {layout_path}")
            skipped += 1
            continue
        
        # Check if empty
        is_empty = int(row.get("is_empty", 0))
        if is_empty:
            skipped += 1
            continue
        
        scene_id = row.get("scene_id", "")
        room_id = row.get("room_id", "")
        layout_type = row.get("type", "")
        
        # Create unique filename
        if layout_type == "scene":
            filename = f"{scene_id}_scene_layout.png"
        else:
            filename = f"{scene_id}_{room_id}_room_layout.png"
        
        dst = output_dirs["layouts"] / filename
        
        if copy_file(layout_path, dst, overwrite):
            key = (scene_id, room_id, layout_type)
            layout_mapping[key] = str(dst)
    
    print(f"✓ Processed {len(layout_mapping)} layouts, skipped {skipped}")
    return layout_mapping


def create_data_points(layouts_manifest: Path, graphs_manifest: Path, povs_manifest: Path,
                      graph_mapping: Dict[str, Tuple[str, str]],
                      pov_mapping: Dict[Tuple[str, str], str],
                      layout_mapping: Dict[Tuple[str, str], str]) -> List[Dict]:
    """
    Create data points: one per POV (pov, graph, layout) and one per scene (0, graph, layout).
    Number of data points = number of POVs + number of scenes.
    """
    print("Creating data points...")
    
    layouts_rows = read_manifest(layouts_manifest)
    graphs_rows = read_manifest(graphs_manifest)
    povs_rows = read_manifest(povs_manifest)
    
    # Build lookup dictionaries for graphs and layouts
    graph_lookup = {}
    for row in graphs_rows:
        scene_id = row.get("scene_id", "")
        room_id = row.get("room_id", "")
        graph_type = row.get("type", "")
        graph_path = row.get("graph_path", "")
        
        # Normalize room_id for scenes
        if graph_type == "scene":
            room_id = "scene"
        
        if graph_path in graph_mapping:
            key = (scene_id, room_id, graph_type)
            graph_lookup[key] = graph_mapping[graph_path]
    
    data_points = []
    
    # Step 1: Create one data point for each POV
    print("Creating data points from POVs...")
    for pov_row in tqdm(povs_rows, desc="Processing POVs"):
        scene_id = pov_row.get("scene_id", "")
        room_id = pov_row.get("room_id", "")
        pov_type = pov_row.get("type", "")
        pov_path_str = pov_row.get("pov_path", "")
        
        if not pov_path_str:
            continue
        
        # Check if empty
        is_empty = int(pov_row.get("is_empty", 0))
        if is_empty:
            continue
        
        # Get POV path from mapping
        pov_key = (scene_id, room_id, pov_type)
        if pov_key not in pov_mapping:
            continue
        
        pov_path = pov_mapping[pov_key]
        
        # Get matching room's graph and layout
        graph_key = (scene_id, room_id, "room")
        layout_key = (scene_id, room_id, "room")
        
        if graph_key not in graph_lookup or layout_key not in layout_mapping:
            continue
        
        graph_json, graph_text = graph_lookup[graph_key]
        layout_path = layout_mapping[layout_key]
        
        # Create data point: (pov, graph, layout)
        data_point = {
            "scene_id": scene_id,
            "room_id": room_id,
            "type": "room",
            "pov_type": pov_type,
            "pov_path": pov_path,
            "graph_json_path": graph_json,
            "graph_text_path": graph_text,
            "layout_path": layout_path,
        }
        data_points.append(data_point)
    
    # Step 2: Create one data point for each scene
    print("Creating data points from scenes...")
    for layout_row in tqdm(layouts_rows, desc="Processing scenes"):
        scene_id = layout_row.get("scene_id", "")
        layout_type = layout_row.get("type", "")
        
        # Only process scenes
        if layout_type != "scene":
            continue
        
        room_id = "scene"
        layout_key = (scene_id, room_id, "scene")
        
        if layout_key not in layout_mapping:
            continue
        
        layout_path = layout_mapping[layout_key]
        
        # Get scene graph
        graph_key = (scene_id, room_id, "scene")
        if graph_key not in graph_lookup:
            continue
        
        graph_json, graph_text = graph_lookup[graph_key]
        
        # Create data point: (0, graph, layout)
        data_point = {
            "scene_id": scene_id,
            "room_id": "scene",
            "type": "scene",
            "pov_type": "",
            "pov_path": "0",
            "graph_json_path": graph_json,
            "graph_text_path": graph_text,
            "layout_path": layout_path,
        }
        data_points.append(data_point)
    
    # Step 3: Remove data points with missing layouts
    print("Filtering data points with missing layouts...")
    initial_count = len(data_points)
    data_points = [
        dp for dp in data_points
        if dp.get("layout_path") and Path(dp["layout_path"]).exists()
    ]
    removed_count = initial_count - len(data_points)
    
    if removed_count > 0:
        print(f"  Removed {removed_count} data points with missing layouts")
    
    print(f"✓ Created {len(data_points)} data points ({len([dp for dp in data_points if dp['type'] == 'room'])} from POVs, {len([dp for dp in data_points if dp['type'] == 'scene'])} from scenes)")
    return data_points


def create_embeddings_for_manifest(data_points: List[Dict], output_manifest: Path,
                                   taxonomy_path: Path, autoencoder_config: Optional[Path],
                                   autoencoder_checkpoint: Optional[Path],
                                   device: str = "cuda", batch_size: int = 32,
                                   num_workers: int = 8, overwrite: bool = False):
    """
    Create embeddings for POVs, graphs, and layouts, and update manifest.
    """
    print("Creating embeddings...")
    
    # Create temporary manifests for each type
    temp_dir = output_manifest.parent / "temp_embeddings"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create POV embeddings
    print("\n[1/3] Creating POV embeddings...")
    pov_rows = []
    for dp in data_points:
        if dp["pov_path"] != "0" and Path(dp["pov_path"]).exists():
            pov_rows.append({
                "pov_path": dp["pov_path"],
                "is_empty": 0
            })
    
    if pov_rows:
        pov_manifest_temp = temp_dir / "povs_temp.csv"
        create_manifest(pov_rows, pov_manifest_temp, ["pov_path", "is_empty"])
        
        pov_manifest_out = temp_dir / "povs_with_embeddings.csv"
        create_pov_embeddings(
            manifest_path=pov_manifest_temp,
            output_manifest=pov_manifest_out,
            save_format="pt",
            batch_size=batch_size
        )
        
        # Read POV embeddings mapping
        pov_emb_df = pd.read_csv(pov_manifest_out)
        pov_emb_mapping = dict(zip(pov_emb_df["pov_path"], pov_emb_df["embedding_path"]))
    else:
        pov_emb_mapping = {}
    
    # 2. Create graph embeddings
    print("\n[2/3] Creating graph embeddings...")
    graph_rows = []
    for dp in data_points:
        if dp["graph_json_path"]:
            graph_rows.append({
                "graph_path": dp["graph_json_path"]
            })
    
    if graph_rows:
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
        
        # Read graph embeddings mapping
        graph_emb_df = pd.read_csv(graph_manifest_out)
        graph_emb_mapping = dict(zip(graph_emb_df["graph_path"], graph_emb_df["embedding_path"]))
    else:
        graph_emb_mapping = {}
    
    # 3. Create layout embeddings
    print("\n[3/3] Creating layout embeddings...")
    layout_rows = []
    for dp in data_points:
        if dp["layout_path"]:
            layout_rows.append({
                "layout_path": dp["layout_path"],
                "is_empty": 0
            })
    
    if layout_rows and autoencoder_checkpoint:
        layout_manifest_temp = temp_dir / "layouts_temp.csv"
        create_manifest(layout_rows, layout_manifest_temp, ["layout_path", "is_empty"])
        
        layout_manifest_out = temp_dir / "layouts_with_embeddings.csv"
        
        # Load autoencoder - config is optional, checkpoint contains it
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
            # Look for config in same directory or parent
            possible_configs = [
                checkpoint_path.parent / f"{checkpoint_path.stem.replace('_checkpoint_best', '')}.yaml",
                checkpoint_path.parent.parent / "experiment_config.yaml",
            ]
            config_path = None
            for pc in possible_configs:
                if pc.exists():
                    config_path = str(pc)
                    break
            if not config_path:
                print("[WARNING] Could not find config file, using defaults for transform")
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
        
        # Read layout embeddings mapping
        layout_emb_df = pd.read_csv(layout_manifest_out)
        layout_emb_mapping = dict(zip(layout_emb_df["layout_path"], layout_emb_df["latent_path"]))
    else:
        layout_emb_mapping = {}
    
    # Update data points with embedding paths
    print("\nUpdating manifest with embedding paths...")
    for dp in data_points:
        pov_path = dp["pov_path"]
        if pov_path != "0":
            dp["pov_embedding_path"] = pov_emb_mapping.get(pov_path, "")
        else:
            dp["pov_embedding_path"] = ""
        
        dp["graph_embedding_path"] = graph_emb_mapping.get(dp["graph_json_path"], "")
        dp["layout_embedding_path"] = layout_emb_mapping.get(dp["layout_path"], "")
    
    # Clean up temp directory
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    print("✓ Embeddings created and manifest updated")


def main():
    parser = argparse.ArgumentParser(
        description="Create ControlNet dataset from manifests"
    )
    
    # Input manifests
    parser.add_argument(
        "--layouts-manifest",
        required=True,
        type=Path,
        help="Path to layouts manifest CSV"
    )
    parser.add_argument(
        "--pov-manifest",
        required=True,
        type=Path,
        help="Path to POVs manifest CSV"
    )
    parser.add_argument(
        "--graph-manifest",
        required=True,
        type=Path,
        help="Path to graphs manifest CSV"
    )
    
    # Output
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to output manifest CSV"
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="Base directory for dataset (will create subdirectories)"
    )
    
    # Required for embeddings
    parser.add_argument(
        "--taxonomy",
        required=True,
        type=Path,
        help="Path to taxonomy.json"
    )
    parser.add_argument(
        "--autoencoder-config",
        type=Path,
        help="Path to autoencoder config YAML (required for layout embeddings)"
    )
    parser.add_argument(
        "--autoencoder-checkpoint",
        type=Path,
        help="Path to autoencoder checkpoint (required for layout embeddings)"
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
        help="Overwrite existing files"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    for manifest_path, name in [
        (args.layouts_manifest, "layouts"),
        (args.pov_manifest, "POVs"),
        (args.graph_manifest, "graphs"),
    ]:
        if not manifest_path.exists():
            parser.error(f"{name} manifest not found: {manifest_path}")
    
    if not args.taxonomy.exists():
        parser.error(f"Taxonomy file not found: {args.taxonomy}")
    
    # Load taxonomy
    print("Loading taxonomy...")
    taxonomy = Taxonomy(args.taxonomy)
    
    # Setup directories
    print("Setting up directories...")
    output_dirs = setup_directories(args.dataset_dir)
    
    # Step 1: Copy and convert graphs
    graph_mapping = copy_graphs_to_dataset(
        args.graph_manifest,
        output_dirs,
        taxonomy,
        overwrite=args.overwrite
    )
    
    # Step 2: Copy POVs
    pov_mapping = copy_povs_to_dataset(
        args.pov_manifest,
        output_dirs,
        overwrite=args.overwrite
    )
    
    # Step 3: Copy layouts
    layout_mapping = copy_layouts_to_dataset(
        args.layouts_manifest,
        output_dirs,
        overwrite=args.overwrite
    )
    
    # Step 4: Create data points
    data_points = create_data_points(
        args.layouts_manifest,
        args.graph_manifest,
        args.pov_manifest,
        graph_mapping,
        pov_mapping,
        layout_mapping
    )
    
    # Step 5: Create embeddings
    if args.autoencoder_config and args.autoencoder_checkpoint:
        create_embeddings_for_manifest(
            data_points,
            args.output,
            args.taxonomy,
            args.autoencoder_config,
            args.autoencoder_checkpoint,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            overwrite=args.overwrite
        )
    else:
        print("Warning: Autoencoder config/checkpoint not provided, skipping layout embeddings")
        for dp in data_points:
            dp["pov_embedding_path"] = ""
            dp["graph_embedding_path"] = ""
            dp["layout_embedding_path"] = ""
    
    # Step 6: Write final manifest
    print(f"\nWriting final manifest to {args.output}...")
    fieldnames = [
        "scene_id", "room_id", "type", "pov_type",
        "pov_path", "graph_json_path", "graph_text_path", "layout_path",
        "pov_embedding_path", "graph_embedding_path", "layout_embedding_path"
    ]
    create_manifest(data_points, args.output, fieldnames)
    
    print(f"\n✓ ControlNet dataset creation complete!")
    print(f"  Output manifest: {args.output}")
    print(f"  Dataset directory: {args.dataset_dir}")


if __name__ == "__main__":
    main()

