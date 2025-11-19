#!/usr/bin/env python3
"""
Create ControlNet training dataset in one unified script.

This script:
1. Embeds POVs (using ResNet18)
2. Embeds graphs (using SentenceTransformer)
3. Embeds layouts (using autoencoder)
4. Creates ControlNet training manifest

Usage:
    python create_controlnet_dataset.py \
        --layouts-manifest datasets/layouts_cleaned.csv \
        --autoencoder-checkpoint checkpoints/autoencoder.pt \
        --taxonomy config/taxonomy.json \
        --output datasets/controlnet_training_manifest.csv
"""

import argparse
import sys
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import json
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_preparation.create_embeddings import (
    create_pov_embeddings,
    create_graph_embeddings
)
from data_preparation.utils.text_utils import graph2text
from common.taxonomy import Taxonomy
from models.autoencoder import Autoencoder
from data_preparation.create_controlnet_manifest_from_joint import (
    create_controlnet_manifest_from_joint
)
import shutil


def embed_layouts_from_manifest(
    layouts_df: pd.DataFrame,
    autoencoder_checkpoint: Path,
    batch_size: int = 32,
    num_workers: int = 8,
    device: str = "cuda"
) -> pd.DataFrame:
    """Embed layouts using autoencoder and add latent_path column."""
    print(f"\n{'='*60}")
    print("Embedding Layouts")
    print(f"{'='*60}")
    
    # Create temporary manifest for embedding
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    temp_manifest = temp_dir / "layouts_for_embedding.csv"
    
    # Ensure required columns exist
    required_cols = ["layout_path", "scene_id", "type", "room_id"]
    if "is_empty" not in layouts_df.columns:
        layouts_df["is_empty"] = False
    
    layouts_df[required_cols + ["is_empty"]].to_csv(temp_manifest, index=False)
    
    # Use create_embeddings.py functionality
    from data_preparation.create_embeddings import create_layout_embeddings_from_manifest
    
    output_manifest = temp_dir / "layouts_with_latents.csv"
    output_latent_dir = temp_dir / "latents"
    
    # Load autoencoder from checkpoint (config is stored in checkpoint)
    print(f"Loading autoencoder from {autoencoder_checkpoint}...")
    device_obj = torch.device(device)
    model = Autoencoder.load_checkpoint(str(autoencoder_checkpoint), map_location=device)
    model = model.to(device_obj)
    model.eval()
    encoder = model.encoder
    
    # Extract config from checkpoint for filters/transforms if needed
    checkpoint_data = torch.load(autoencoder_checkpoint, map_location=device)
    autoencoder_config_dict = checkpoint_data.get("config", {})
    
    # Save config to temp file if it exists (for create_layout_embeddings_from_manifest)
    temp_config_path = None
    if autoencoder_config_dict:
        import yaml
        temp_config_file = temp_dir / "autoencoder_config.yaml"
        with open(temp_config_file, 'w') as f:
            yaml.dump(autoencoder_config_dict, f)
        temp_config_path = str(temp_config_file)
    
    create_layout_embeddings_from_manifest(
        encoder=encoder,
        manifest_path=temp_manifest,
        output_manifest_path=output_manifest,
        batch_size=batch_size,
        num_workers=num_workers,
        overwrite=True,
        device=device,
        autoencoder_config_path=temp_config_path,  # Use extracted config if available
        output_latent_dir=output_latent_dir
    )
    
    # Load the output manifest with latents
    layouts_with_latents = pd.read_csv(output_manifest)
    
    # Merge back with original dataframe to preserve all columns
    merged = layouts_df.merge(
        layouts_with_latents[["scene_id", "type", "room_id", "latent_path"]],
        on=["scene_id", "type", "room_id"],
        how="left"
    )
    
    return merged


def copy_and_organize_files(layouts_df, output_base_dir: Path, taxonomy_path: Path):
    """
    Copy and organize all files for standalone ControlNet dataset.
    
    Creates structure:
    datasets/controlnet/
      graphs/     - graph JSONs and texts
      povs/       - POV images
      layouts/    - layout images
      latents/    - layout embeddings
      embeddings/ - POV and graph embeddings
    """
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    graphs_dir = output_base_dir / "graphs"
    povs_dir = output_base_dir / "povs"
    layouts_dir = output_base_dir / "layouts"
    latents_dir = output_base_dir / "latents"
    embeddings_dir = output_base_dir / "embeddings"
    
    for dir_path in [graphs_dir, povs_dir, layouts_dir, latents_dir, embeddings_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Organizing Files for Standalone Dataset")
    print(f"{'='*60}")
    print(f"Output directory: {output_base_dir}")
    
    # Load taxonomy for graph text conversion
    taxonomy = Taxonomy(taxonomy_path)
    
    # Track copied files to update paths
    copied_graphs = {}
    copied_povs = {}
    copied_layouts = {}
    copied_latents = {}
    
    print(f"\n  Copying and processing files...")
    
    for idx, row in tqdm(layouts_df.iterrows(), total=len(layouts_df), desc="Organizing files"):
        scene_id = row["scene_id"]
        room_id = row.get("room_id", "")
        layout_type = row["type"]
        
        # Copy graph JSON and create text
        graph_path = row.get("graph_path", "")
        if graph_path and Path(graph_path).exists():
            graph_path_obj = Path(graph_path)
            
            # Determine output filename
            if layout_type == "scene":
                graph_filename = f"{scene_id}_scene_graph.json"
                graph_text_filename = f"{scene_id}_scene_graph.txt"
            else:
                graph_filename = f"{scene_id}_{room_id}_graph.json"
                graph_text_filename = f"{scene_id}_{room_id}_graph.txt"
            
            dest_graph_json = graphs_dir / graph_filename
            dest_graph_txt = graphs_dir / graph_text_filename
            
            # Copy graph JSON if not already copied
            if graph_path not in copied_graphs:
                shutil.copy2(graph_path_obj, dest_graph_json)
                copied_graphs[graph_path] = str(dest_graph_json.resolve())
                
                # Convert to text
                try:
                    text = graph2text(graph_path_obj, taxonomy)
                    if text:
                        dest_graph_txt.write_text(text, encoding='utf-8')
                except Exception as e:
                    print(f"    Warning: Failed to convert graph to text for {graph_path}: {e}")
            
            # Update path in dataframe
            layouts_df.at[idx, "graph_path"] = copied_graphs[graph_path]
            layouts_df.at[idx, "graph_text_path"] = str(dest_graph_txt.resolve())
        
        # Copy POV image
        pov_path = row.get("pov_path", "")
        if pov_path and Path(pov_path).exists():
            pov_path_obj = Path(pov_path)
            
            # Extract viewpoint number from filename (e.g., v01, v02)
            match = re.search(r'_v(\d+)_', pov_path_obj.name)
            view_num = match.group(1) if match else "01"
            
            pov_filename = f"{scene_id}_{room_id}_v{view_num}_pov_tex.png"
            dest_pov = povs_dir / pov_filename
            
            # Copy POV if not already copied
            if pov_path not in copied_povs:
                shutil.copy2(pov_path_obj, dest_pov)
                copied_povs[pov_path] = str(dest_pov.resolve())
            
            # Update path in dataframe
            layouts_df.at[idx, "pov_path"] = copied_povs[pov_path]
        
        # Copy layout image
        layout_path = row.get("layout_path", "")
        if layout_path and Path(layout_path).exists():
            layout_path_obj = Path(layout_path)
            
            if layout_type == "scene":
                layout_filename = f"{scene_id}_scene_layout.png"
            else:
                layout_filename = f"{scene_id}_{room_id}_room_seg_layout.png"
            
            dest_layout = layouts_dir / layout_filename
            
            # Copy layout if not already copied
            if layout_path not in copied_layouts:
                shutil.copy2(layout_path_obj, dest_layout)
                copied_layouts[layout_path] = str(dest_layout.resolve())
            
            # Update path in dataframe
            layouts_df.at[idx, "layout_path"] = copied_layouts[layout_path]
        
        # Copy layout latent
        latent_path = row.get("latent_path", "")
        if latent_path and Path(latent_path).exists():
            latent_path_obj = Path(latent_path)
            
            if layout_type == "scene":
                latent_filename = f"{scene_id}_scene_layout.pt"
            else:
                latent_filename = f"{scene_id}_{room_id}_room_seg_layout.pt"
            
            dest_latent = latents_dir / latent_filename
            
            # Copy latent if not already copied
            if latent_path not in copied_latents:
                shutil.copy2(latent_path_obj, dest_latent)
                copied_latents[latent_path] = str(dest_latent.resolve())
            
            # Update path in dataframe
            layouts_df.at[idx, "latent_path"] = copied_latents[latent_path]
    
    print(f"\n  Copied:")
    print(f"    Graphs: {len(copied_graphs)} JSONs + texts")
    print(f"    POVs: {len(copied_povs)} images")
    print(f"    Layouts: {len(copied_layouts)} images")
    print(f"    Latents: {len(copied_latents)} embeddings")
    
    return layouts_df


def create_controlnet_dataset(
    layouts_manifest: Path,
    autoencoder_checkpoint: Path,
    taxonomy_path: Path,
    output_manifest: Path,
    pov_manifest: Path = None,
    graph_manifest: Path = None,
    pov_batch_size: int = 32,
    graph_model: str = "all-MiniLM-L6-v2",
    layout_batch_size: int = 32,
    num_workers: int = 8,
    device: str = "cuda",
    handle_scenes_without_pov: str = "zero",
    create_standalone: bool = True
):
    """
    Create complete ControlNet training dataset.
    
    Args:
        layouts_manifest: Path to layouts manifest (must have layout_path, scene_id, type, room_id)
        autoencoder_checkpoint: Path to autoencoder checkpoint (config is stored in checkpoint)
        taxonomy_path: Path to taxonomy.json for graph embedding
        output_manifest: Output path for ControlNet training manifest
        pov_batch_size: Batch size for POV embedding
        graph_model: SentenceTransformer model name
        layout_batch_size: Batch size for layout embedding
        num_workers: Number of workers for data loading
        device: Device to use (cuda/cpu)
        handle_scenes_without_pov: How to handle scenes without POVs (zero/empty)
    """
    print(f"\n{'='*60}")
    print("Creating ControlNet Training Dataset")
    print(f"{'='*60}")
    print(f"Layouts manifest: {layouts_manifest}")
    print(f"Autoencoder checkpoint: {autoencoder_checkpoint}")
    print(f"Taxonomy: {taxonomy_path}")
    print(f"Output: {output_manifest}")
    print(f"Create standalone dataset: {create_standalone}")
    print(f"{'='*60}\n")
    
    # Determine output base directory for standalone dataset
    if create_standalone:
        output_base_dir = output_manifest.parent / "controlnet"
    else:
        output_base_dir = None
    
    # Load layouts manifest
    print("Loading layouts manifest...")
    layouts_df = pd.read_csv(layouts_manifest)
    print(f"  Found {len(layouts_df)} layout entries")
    
    # Filter empty layouts
    if "is_empty" in layouts_df.columns:
        layouts_df = layouts_df[layouts_df["is_empty"] == False].copy()
        print(f"  After filtering empty: {len(layouts_df)} entries")
    
    # Required columns
    required_cols = ["layout_path", "scene_id", "type", "room_id"]
    missing_cols = [col for col in required_cols if col not in layouts_df.columns]
    if missing_cols:
        raise ValueError(f"Layouts manifest missing required columns: {missing_cols}")
    
    # Step 1: Embed layouts
    print(f"\n{'='*60}")
    print("Step 1/4: Embedding Layouts")
    print(f"{'='*60}")
    layouts_df = embed_layouts_from_manifest(
        layouts_df,
        autoencoder_checkpoint,
        batch_size=layout_batch_size,
        num_workers=num_workers,
        device=device
    )
    
    # Filter out rows without latents
    layouts_df = layouts_df[layouts_df["latent_path"].notna() & (layouts_df["latent_path"] != "")].copy()
    print(f"  Successfully embedded {len(layouts_df)} layouts")
    
    # Step 2: Find POV and graph paths (load from manifests if available, otherwise infer)
    print(f"\n{'='*60}")
    print("Step 2/4: Finding POV and Graph Paths")
    print(f"{'='*60}")
    
    # Check if paths already exist in manifest
    if "pov_path" not in layouts_df.columns:
        layouts_df["pov_path"] = ""
    if "graph_path" not in layouts_df.columns:
        layouts_df["graph_path"] = ""
    
    # Load POV manifest if provided
    pov_df = None
    if pov_manifest and pov_manifest.exists():
        print(f"  Loading POV manifest: {pov_manifest}")
        pov_df = pd.read_csv(pov_manifest)
        print(f"  Loaded {len(pov_df)} POV entries")
        # Filter out empty POVs
        if "is_empty" in pov_df.columns:
            pov_df = pov_df[pov_df["is_empty"] == 0].copy()
            print(f"  After filtering empty: {len(pov_df)} POV entries")
    
    # Load graph manifest if provided
    graph_df = None
    if graph_manifest and graph_manifest.exists():
        print(f"  Loading graph manifest: {graph_manifest}")
        graph_df = pd.read_csv(graph_manifest)
        print(f"  Loaded {len(graph_df)} graph entries")
    
    # Only infer for rows that don't have paths
    needs_pov = layouts_df["pov_path"].isna() | (layouts_df["pov_path"] == "")
    needs_graph = layouts_df["graph_path"].isna() | (layouts_df["graph_path"] == "")
    
    # Try to infer POV and graph paths from layout paths
    # Cache for directory existence to avoid repeated checks
    _dir_cache = {}
    
    def find_all_pov_paths(layout_path_str, scene_id, room_id):
        """Find ALL POV paths for a room layout - returns list of paths. OPTIMIZED."""
        if pd.isna(layout_path_str) or not room_id or pd.isna(room_id) or room_id == "":
            return []
        
        pov_paths = []
        
        # Try relative to layout path (structure: scenes/{scene_id}/rooms/{room_id}/layouts/...)
        if not pd.isna(layout_path_str):
            try:
                layout_path = Path(layout_path_str)
                layout_dir = layout_path.parent  # layouts/
                room_dir = layout_dir.parent     # {room_id}/
                
                # POVs are in rooms/{room_id}/povs/tex/
                pov_dir = room_dir / "povs" / "tex"
                
                # Cache directory existence check
                pov_dir_str = str(pov_dir)
                if pov_dir_str not in _dir_cache:
                    _dir_cache[pov_dir_str] = pov_dir.exists() and pov_dir.is_dir()
                
                if _dir_cache[pov_dir_str]:
                    # Find all POV files for this room - glob is still needed but we cache dir existence
                    try:
                        pov_files = list(pov_dir.glob(f"{scene_id}_{room_id}_v*_pov_tex.png"))
                        pov_paths.extend([str(p.resolve()) for p in sorted(pov_files)])
                    except Exception:
                        pass
            except Exception:
                pass
        
        # Skip collected directory check - assume POVs are in scenes structure
        # This saves ~15k file system operations
        
        return pov_paths
    
    def infer_graph_path(layout_path_str, scene_id, room_id, layout_type):
        """Infer graph path based on actual directory structure.
        
        For scenes: graphs are in scene root: datasets/scenes/{scene_id}/{scene_id}_scene_graph.json
        For rooms: graphs are in layouts/: datasets/scenes/{scene_id}/rooms/{room_id}/layouts/{scene_id}_{room_id}_graph.json
        """
        if pd.isna(layout_path_str):
            return ""
        
        layout_path = Path(layout_path_str)
        layout_dir = layout_path.parent  # layouts/ directory
        
        if layout_type == "room" and room_id and not pd.isna(room_id) and room_id != "":
            # Room graphs: in the layouts/ directory with the layout
            # Structure: datasets/scenes/{scene_id}/rooms/{room_id}/layouts/{scene_id}_{room_id}_graph.json
            graph_filenames = [
                f"{scene_id}_{room_id}_graph.json",
                f"{scene_id}_{room_id}_room_graph.json"
            ]
            
            # Try in layouts directory (same as layout) - this should work
            for graph_filename in graph_filenames:
                graph_file = layout_dir / graph_filename
                if graph_file.exists():
                    return str(graph_file.resolve())
            
            # Try collected directory
            for graph_filename in graph_filenames:
                collected_graph = Path(f"/work3/s233249/ImgiNav/datasets/collected/graphs/{graph_filename}")
                if collected_graph.exists():
                    return str(collected_graph.resolve())
            
        else:
            # Scene graphs: in scene root, NOT in layouts/
            # Structure: datasets/scenes/{scene_id}/{scene_id}_scene_graph.json
            graph_filename = f"{scene_id}_scene_graph.json"
            
            # Scene root is parent of layouts/ directory
            if layout_dir.name == "layouts":
                scene_root = layout_dir.parent
            else:
                # If layout is directly in scene root, use that
                scene_root = layout_dir
            
            graph_file = scene_root / graph_filename
            if graph_file.exists():
                return str(graph_file.resolve())
            
            # Try collected directory
            collected_graph = Path(f"/work3/s233249/ImgiNav/datasets/collected/graphs/{graph_filename}")
            if collected_graph.exists():
                return str(collected_graph.resolve())
        
        return ""
    
    # Expand layouts to include all POVs (one row per POV)
    print(f"  Finding all POVs for {len(layouts_df)} layouts...")
    
    # Build a cache: (scene_id, room_id) -> list of POV paths
    pov_cache = {}
    
    if pov_df is not None:
        print(f"  Using POV manifest to match POVs to layouts...")
        for (scene_id, room_id), group in pov_df.groupby(["scene_id", "room_id"]):
            pov_paths = group["pov_path"].tolist()
            pov_cache[(scene_id, room_id)] = pov_paths
        print(f"  Found POVs for {len(pov_cache)} unique rooms")
    else:
        # Fallback: infer from file system (slow)
        print(f"  No POV manifest provided, inferring from file system (this will be slow)...")
        room_layouts = layouts_df[
            (layouts_df["type"] == "room") & 
            (layouts_df["room_id"].notna()) & 
            (layouts_df["room_id"] != "")
        ].copy()
        
        if len(room_layouts) > 0:
            unique_rooms = room_layouts.groupby(["scene_id", "room_id"]).first().reset_index()
            print(f"  Checking POV directories for {len(unique_rooms)} unique rooms...")
            
            for idx, row in tqdm(unique_rooms.iterrows(), total=len(unique_rooms), desc="Scanning POV dirs"):
                scene_id = row["scene_id"]
                room_id = row["room_id"]
                layout_path_str = row["layout_path"]
                pov_paths = find_all_pov_paths(layout_path_str, scene_id, room_id)
                pov_cache[(scene_id, room_id)] = pov_paths
    
    print(f"  Expanding layouts with found POVs...")
    
    expanded_rows = []
    for idx, row in layouts_df.iterrows():
        layout_type = row["type"]
        
        if layout_type == "room" and row.get("room_id") and not pd.isna(row.get("room_id")):
            cache_key = (row["scene_id"], row["room_id"])
            pov_paths = pov_cache.get(cache_key, [])
            
            if pov_paths:
                for pov_path in pov_paths:
                    new_row = row.copy()
                    new_row["pov_path"] = str(pov_path)
                    if pov_df is not None:
                        pov_row = pov_df[pov_df["pov_path"] == pov_path]
                        if len(pov_row) > 0:
                            new_row["pov_type"] = pov_row.iloc[0].get("type", "")
                    expanded_rows.append(new_row)
            else:
                new_row = row.copy()
                new_row["pov_path"] = ""
                expanded_rows.append(new_row)
        else:
            new_row = row.copy()
            new_row["pov_path"] = ""
            expanded_rows.append(new_row)
    
    layouts_df = pd.DataFrame(expanded_rows).reset_index(drop=True)
    print(f"  Expanded to {len(layouts_df)} rows")
    
    # Now find graph paths
    if graph_df is not None:
        # Use graph manifest - MUCH faster!
        print(f"  Using graph manifest to match graphs to layouts...")
        # Create lookup: (scene_id, type, room_id) -> graph_path
        graph_lookup = {}
        for _, row in graph_df.iterrows():
            scene_id = row["scene_id"]
            graph_type = row.get("type", "room")
            room_id = str(row.get("room_id", "")) if pd.notna(row.get("room_id")) else ""
            graph_path = row.get("graph_path", "")
            
            if graph_type == "scene" or room_id == "scene" or room_id == "":
                key = (scene_id, "scene")
            else:
                key = (scene_id, room_id)
            graph_lookup[key] = graph_path
        
        # Match graphs to layouts
        def get_graph_path(row):
            scene_id = row["scene_id"]
            layout_type = row["type"]
            room_id = str(row.get("room_id", "")) if pd.notna(row.get("room_id")) else ""
            
            if layout_type == "scene" or room_id == "":
                key = (scene_id, "scene")
            else:
                key = (scene_id, room_id)
            return graph_lookup.get(key, "")
        
        layouts_df["graph_path"] = layouts_df.apply(get_graph_path, axis=1)
        matched_count = (layouts_df["graph_path"] != "").sum()
        print(f"  Matched graphs for {matched_count}/{len(layouts_df)} layouts")
    elif needs_graph.any() or True:  # Fallback: infer from file system
        print(f"  No graph manifest provided, inferring graph paths (this may be slow)...")
        layouts_df["graph_path"] = layouts_df.apply(
            lambda row: infer_graph_path(row["layout_path"], row["scene_id"], row.get("room_id", ""), row["type"]),
            axis=1
        )
    else:
        print("  Using existing graph paths from manifest")
    
    pov_count = (layouts_df["pov_path"] != "").sum()
    graph_count = (layouts_df["graph_path"] != "").sum()
    print(f"  Found {pov_count} POV paths")
    print(f"  Found {graph_count} graph paths")
    
    # Debug: show a few examples of missing graphs if none found
    if graph_count == 0:
        print(f"\n  [DEBUG] Checking why graphs weren't found...")
        # Check unique layouts (not all expanded POV rows)
        unique_layouts = layouts_df.drop_duplicates(subset=["layout_path", "scene_id", "type", "room_id"])
        sample_rows = unique_layouts.head(3)
        for idx, row in sample_rows.iterrows():
            layout_path = Path(row["layout_path"])
            layout_dir = layout_path.parent
            scene_id = row["scene_id"]
            room_id = row.get("room_id", "")
            layout_type = row["type"]
            
            print(f"\n  Example {idx}:")
            print(f"    Layout: {layout_path}")
            print(f"    Layout dir: {layout_dir}")
            print(f"    Type: {layout_type}, Scene: {scene_id}, Room: {room_id}")
            
            if layout_type == "room" and room_id:
                expected_graph = layout_dir / f"{scene_id}_{room_id}_graph.json"
                print(f"    Expected graph: {expected_graph}")
                print(f"    Exists: {expected_graph.exists()}")
                if layout_dir.exists():
                    json_files = list(layout_dir.glob('*.json'))
                    print(f"    Files in layout dir: {[f.name for f in json_files[:5]]}")
            else:
                scene_root = layout_dir.parent if layout_dir.name == "layouts" else layout_dir
                expected_graph = scene_root / f"{scene_id}_scene_graph.json"
                print(f"    Scene root: {scene_root}")
                print(f"    Expected graph: {expected_graph}")
                print(f"    Exists: {expected_graph.exists()}")
                if scene_root.exists():
                    json_files = list(scene_root.glob('*_scene_graph.json'))
                    print(f"    Scene graph files: {[f.name for f in json_files[:5]]}")
    
    # Step 3: Copy and organize files for standalone dataset (before embedding)
    if create_standalone and output_base_dir:
        print(f"\n{'='*60}")
        print("Step 5/6: Creating Standalone Dataset")
        print(f"{'='*60}")
        layouts_df = copy_and_organize_files(
            layouts_df, output_base_dir, taxonomy_path
        )
        
        # Update embedding paths to point to embeddings directory
        embeddings_dir = output_base_dir / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy POV embeddings
        pov_embeddings_copied = {}
        for idx, row in layouts_df.iterrows():
            pov_emb_path = row.get("pov_embedding_path", "")
            if pov_emb_path and Path(pov_emb_path).exists():
                if pov_emb_path not in pov_embeddings_copied:
                    pov_path_obj = Path(pov_emb_path)
                    scene_id = row["scene_id"]
                    room_id = row.get("room_id", "")
                    pov_path_str = row.get("pov_path", "")
                    match = re.search(r'_v(\d+)_', pov_path_str) if pov_path_str else None
                    view_num = match.group(1) if match else "01"
                    
                    dest_emb = embeddings_dir / f"{scene_id}_{room_id}_v{view_num}_pov_tex.pt"
                    shutil.copy2(pov_path_obj, dest_emb)
                    pov_embeddings_copied[pov_emb_path] = str(dest_emb.resolve())
                
                layouts_df.at[idx, "pov_embedding_path"] = pov_embeddings_copied[pov_emb_path]
        
    
    # Step 4: Embed POVs
    print(f"\n{'='*60}")
    print("Step 4/6: Embedding POVs")
    print(f"{'='*60}")
    
    pov_rows = layouts_df[layouts_df["pov_path"] != ""].copy()
    if len(pov_rows) > 0:
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        temp_pov_manifest = temp_dir / "povs_for_embedding.csv"
        output_pov_manifest = temp_dir / "povs_with_embeddings.csv"
        
        if "is_empty" not in pov_rows.columns:
            pov_rows["is_empty"] = False
        
        pov_rows[["pov_path", "is_empty"]].to_csv(temp_pov_manifest, index=False)
        
        create_pov_embeddings(
            manifest_path=temp_pov_manifest,
            output_manifest=output_pov_manifest,
            batch_size=pov_batch_size,
            save_format="pt"
        )
        
        pov_emb_df = pd.read_csv(output_pov_manifest)
        pov_emb_dict = dict(zip(pov_emb_df["pov_path"], pov_emb_df["embedding_path"]))
        layouts_df["pov_embedding_path"] = layouts_df["pov_path"].map(pov_emb_dict).fillna("")
        print(f"  Embedded {len([x for x in layouts_df['pov_embedding_path'] if x != ''])} POVs")
        
        # Copy POV embeddings to embeddings directory
        if create_standalone and output_base_dir:
            embeddings_dir = output_base_dir / "embeddings"
            pov_embeddings_copied = {}
            for idx, row in layouts_df.iterrows():
                pov_emb_path = row.get("pov_embedding_path", "")
                if pov_emb_path and Path(pov_emb_path).exists():
                    if pov_emb_path not in pov_embeddings_copied:
                        pov_path_obj = Path(pov_emb_path)
                        scene_id = row["scene_id"]
                        room_id = row.get("room_id", "")
                        pov_path_str = row.get("pov_path", "")
                        match = re.search(r'_v(\d+)_', pov_path_str) if pov_path_str else None
                        view_num = match.group(1) if match else "01"
                        
                        dest_emb = embeddings_dir / f"{scene_id}_{room_id}_v{view_num}_pov_tex.pt"
                        shutil.copy2(pov_path_obj, dest_emb)
                        pov_embeddings_copied[pov_emb_path] = str(dest_emb.resolve())
                    
                    layouts_df.at[idx, "pov_embedding_path"] = pov_embeddings_copied[pov_emb_path]
    else:
        layouts_df["pov_embedding_path"] = ""
        print("  No POVs to embed")
    
    # Step 5: Embed graphs (after copying, so text files exist and we use copied graph JSONs)
    print(f"\n{'='*60}")
    print("Step 5/6: Embedding Graphs")
    print(f"{'='*60}")
    
    graph_rows = layouts_df[layouts_df["graph_path"] != ""].copy()
    if len(graph_rows) > 0:
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        temp_graph_manifest = temp_dir / "graphs_for_embedding.csv"
        output_graph_manifest = temp_dir / "graphs_with_embeddings.csv"
        
        graph_rows[["graph_path", "scene_id", "type", "room_id"]].to_csv(
            temp_graph_manifest, index=False
        )
        
        # Embed graphs: reads graph JSON -> converts to text -> embeds text
        create_graph_embeddings(
            manifest_path=temp_graph_manifest,
            taxonomy_path=taxonomy_path,
            output_manifest=output_graph_manifest,
            model_name=graph_model,
            save_format="pt"
        )
        
        graph_emb_df = pd.read_csv(output_graph_manifest)
        graph_emb_dict = dict(zip(graph_emb_df["graph_path"], graph_emb_df["embedding_path"]))
        layouts_df["graph_embedding_path"] = layouts_df["graph_path"].map(graph_emb_dict).fillna("")
        print(f"  Embedded {len([x for x in layouts_df['graph_embedding_path'] if x != ''])} graphs")
        
        # Copy graph embeddings to embeddings directory
        if create_standalone and output_base_dir:
            embeddings_dir = output_base_dir / "embeddings"
            graph_embeddings_copied = {}
            for idx, row in layouts_df.iterrows():
                graph_emb_path = row.get("graph_embedding_path", "")
                if graph_emb_path and Path(graph_emb_path).exists():
                    if graph_emb_path not in graph_embeddings_copied:
                        graph_emb_path_obj = Path(graph_emb_path)
                        scene_id = row["scene_id"]
                        room_id = row.get("room_id", "")
                        layout_type = row["type"]
                        
                        if layout_type == "scene":
                            dest_emb = embeddings_dir / f"{scene_id}_scene_graph.pt"
                        else:
                            dest_emb = embeddings_dir / f"{scene_id}_{room_id}_graph.pt"
                        
                        shutil.copy2(graph_emb_path_obj, dest_emb)
                        graph_embeddings_copied[graph_emb_path] = str(dest_emb.resolve())
                    
                    layouts_df.at[idx, "graph_embedding_path"] = graph_embeddings_copied[graph_emb_path]
    else:
        layouts_df["graph_embedding_path"] = ""
        print("  No graphs to embed")
    
    # Step 6: Create ControlNet training manifest
    print(f"\n{'='*60}")
    print("Step 6/6: Creating ControlNet Training Manifest")
    print(f"{'='*60}")
    
    # Create a "joint manifest" structure in memory
    joint_df = layouts_df.copy()
    
    # Add graph_text_path if we can infer it
    def infer_graph_text_path(graph_path_str):
        if pd.isna(graph_path_str) or graph_path_str == "":
            return ""
        graph_path = Path(graph_path_str)
        text_path = graph_path.with_suffix('.txt')
        if text_path.exists():
            return str(text_path.resolve())
        return ""
    
    joint_df["graph_text_path"] = joint_df["graph_path"].apply(infer_graph_text_path)
    
    # Save temporary joint manifest
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    temp_joint_manifest = temp_dir / "joint_manifest_temp.csv"
    joint_df.to_csv(temp_joint_manifest, index=False)
    
    # Use the existing manifest creation function
    create_controlnet_manifest_from_joint(
        joint_manifest=temp_joint_manifest,
        output_manifest=output_manifest,
        layouts_latent_manifest=None,  # Already has latent_path
        handle_scenes_without_pov=handle_scenes_without_pov
    )
    
    print(f"\n{'='*60}")
    print("✓ ControlNet Dataset Creation Complete!")
    print(f"{'='*60}")
    print(f"Output manifest: {output_manifest}")
    print(f"Total training samples: {len(pd.read_csv(output_manifest))}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create ControlNet training dataset (embeds everything and creates manifest)"
    )
    parser.add_argument(
        "--layouts-manifest",
        type=Path,
        required=True,
        help="Path to layouts manifest (must have layout_path, scene_id, type, room_id)"
    )
    parser.add_argument(
        "--autoencoder-checkpoint",
        type=Path,
        required=True,
        help="Path to autoencoder checkpoint (config is stored in checkpoint)"
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        required=True,
        help="Path to taxonomy.json for graph embedding"
    )
    parser.add_argument(
        "--pov-manifest",
        type=Path,
        default=None,
        help="Path to POV manifest CSV (optional, speeds up POV finding significantly)"
    )
    parser.add_argument(
        "--graph-manifest",
        type=Path,
        default=None,
        help="Path to graph manifest CSV (optional, speeds up graph finding significantly)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for ControlNet training manifest"
    )
    parser.add_argument(
        "--pov-batch-size",
        type=int,
        default=32,
        help="Batch size for POV embedding"
    )
    parser.add_argument(
        "--graph-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model for graph embedding"
    )
    parser.add_argument(
        "--layout-batch-size",
        type=int,
        default=32,
        help="Batch size for layout embedding"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of workers for data loading"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--handle-scenes-without-pov",
        type=str,
        default="zero",
        choices=["zero", "empty"],
        help="How to handle scenes without POVs"
    )
    parser.add_argument(
        "--create-standalone",
        action="store_true",
        default=True,
        help="Create standalone dataset with all files copied to datasets/controlnet/"
    )
    parser.add_argument(
        "--no-standalone",
        dest="create_standalone",
        action="store_false",
        help="Don't create standalone dataset (use original file paths)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.layouts_manifest.exists():
        print(f"[ERROR] Layouts manifest not found: {args.layouts_manifest}")
        sys.exit(1)
    
    if not args.autoencoder_checkpoint.exists():
        print(f"[ERROR] Autoencoder checkpoint not found: {args.autoencoder_checkpoint}")
        sys.exit(1)
    
    if not args.taxonomy.exists():
        print(f"[ERROR] Taxonomy file not found: {args.taxonomy}")
        sys.exit(1)
    
    create_controlnet_dataset(
        layouts_manifest=args.layouts_manifest,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        taxonomy_path=args.taxonomy,
        output_manifest=args.output,
        pov_manifest=args.pov_manifest,
        graph_manifest=args.graph_manifest,
        pov_batch_size=args.pov_batch_size,
        graph_model=args.graph_model,
        layout_batch_size=args.layout_batch_size,
        num_workers=args.num_workers,
        device=args.device,
        handle_scenes_without_pov=args.handle_scenes_without_pov,
        create_standalone=args.create_standalone
    )


if __name__ == "__main__":
    main()

