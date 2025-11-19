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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_preparation.create_embeddings import (
    create_pov_embeddings,
    create_graph_embeddings
)
from models.autoencoder import Autoencoder
from data_preparation.create_controlnet_manifest_from_joint import (
    create_controlnet_manifest_from_joint
)


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


def create_controlnet_dataset(
    layouts_manifest: Path,
    autoencoder_checkpoint: Path,
    taxonomy_path: Path,
    output_manifest: Path,
    pov_batch_size: int = 32,
    graph_model: str = "all-MiniLM-L6-v2",
    layout_batch_size: int = 32,
    num_workers: int = 8,
    device: str = "cuda",
    handle_scenes_without_pov: str = "zero"
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
    print(f"{'='*60}\n")
    
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
    
    # Step 2: Find POV and graph paths (use existing or infer from layout paths)
    print(f"\n{'='*60}")
    print("Step 2/4: Finding POV and Graph Paths")
    print(f"{'='*60}")
    
    # Check if paths already exist in manifest
    if "pov_path" not in layouts_df.columns:
        layouts_df["pov_path"] = ""
    if "graph_path" not in layouts_df.columns:
        layouts_df["graph_path"] = ""
    
    # Only infer for rows that don't have paths
    needs_pov = layouts_df["pov_path"].isna() | (layouts_df["pov_path"] == "")
    needs_graph = layouts_df["graph_path"].isna() | (layouts_df["graph_path"] == "")
    
    # Try to infer POV and graph paths from layout paths
    def infer_pov_path(layout_path_str, scene_id, room_id):
        if pd.isna(layout_path_str) or not room_id:
            return ""
        
        # Try collected directory first (most common)
        collected_pov = Path(f"/work3/s233249/ImgiNav/datasets/collected/povs/tex/{scene_id}_{room_id}_v01_pov_tex.png")
        if collected_pov.exists():
            return str(collected_pov.resolve())
        
        # Try other view numbers
        for view_num in range(1, 10):
            pov_path = Path(f"/work3/s233249/ImgiNav/datasets/collected/povs/tex/{scene_id}_{room_id}_v{view_num:02d}_pov_tex.png")
            if pov_path.exists():
                return str(pov_path.resolve())
        
        # Try relative to layout path
        if not pd.isna(layout_path_str):
            layout_path = Path(layout_path_str)
            # Try same directory structure
            if "layout" in str(layout_path):
                base_dir = layout_path.parent.parent if "layouts" in str(layout_path.parent) else layout_path.parent
                pov_dir = base_dir / "povs" / "tex"
                if pov_dir.exists():
                    for pov_file in pov_dir.glob(f"{scene_id}_{room_id}_*_pov_tex.png"):
                        return str(pov_file.resolve())
        
        return ""
    
    def infer_graph_path(layout_path_str, scene_id, room_id, layout_type):
        if pd.isna(layout_path_str):
            return ""
        
        # Try collected directory first (most common)
        if layout_type == "room" and room_id:
            collected_graph = Path(f"/work3/s233249/ImgiNav/datasets/collected/graphs/{scene_id}_{room_id}_room_graph.json")
        else:
            collected_graph = Path(f"/work3/s233249/ImgiNav/datasets/collected/graphs/{scene_id}_scene_graph.json")
        
        if collected_graph.exists():
            return str(collected_graph.resolve())
        
        # Try relative to layout path
        layout_path = Path(layout_path_str)
        if "layout" in str(layout_path):
            base_dir = layout_path.parent.parent if "layouts" in str(layout_path.parent) else layout_path.parent
            graph_dir = base_dir / "graphs"
            if graph_dir.exists():
                if layout_type == "room" and room_id:
                    for graph_file in graph_dir.glob(f"{scene_id}_{room_id}_*.json"):
                        return str(graph_file.resolve())
                else:
                    for graph_file in graph_dir.glob(f"{scene_id}_scene_*.json"):
                        return str(graph_file.resolve())
        
        return ""
    
    if needs_pov.any():
        print(f"  Inferring POV paths for {needs_pov.sum()} entries...")
        layouts_df.loc[needs_pov, "pov_path"] = layouts_df[needs_pov].apply(
            lambda row: infer_pov_path(row["layout_path"], row["scene_id"], row.get("room_id", "")),
            axis=1
        )
    else:
        print("  Using existing POV paths from manifest")
    
    if needs_graph.any():
        print(f"  Inferring graph paths for {needs_graph.sum()} entries...")
        layouts_df.loc[needs_graph, "graph_path"] = layouts_df[needs_graph].apply(
            lambda row: infer_graph_path(row["layout_path"], row["scene_id"], row.get("room_id", ""), row["type"]),
            axis=1
        )
    else:
        print("  Using existing graph paths from manifest")
    
    pov_count = (layouts_df["pov_path"] != "").sum()
    graph_count = (layouts_df["graph_path"] != "").sum()
    print(f"  Found {pov_count} POV paths")
    print(f"  Found {graph_count} graph paths")
    
    # Step 3: Embed POVs
    print(f"\n{'='*60}")
    print("Step 3/4: Embedding POVs")
    print(f"{'='*60}")
    
    pov_rows = layouts_df[layouts_df["pov_path"] != ""].copy()
    if len(pov_rows) > 0:
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        temp_pov_manifest = temp_dir / "povs_for_embedding.csv"
        output_pov_manifest = temp_dir / "povs_with_embeddings.csv"
        
        # Ensure is_empty column exists
        if "is_empty" not in pov_rows.columns:
            pov_rows["is_empty"] = False
        
        pov_rows[["pov_path", "is_empty"]].to_csv(temp_pov_manifest, index=False)
        
        # Embed POVs - this creates embeddings next to the POV files
        create_pov_embeddings(
            manifest_path=temp_pov_manifest,
            output_manifest=output_pov_manifest,
            batch_size=pov_batch_size,
            save_format="pt"
        )
        
        # Load the output manifest to get embedding paths
        pov_emb_df = pd.read_csv(output_pov_manifest)
        pov_emb_dict = dict(zip(pov_emb_df["pov_path"], pov_emb_df["embedding_path"]))
        
        # Add embedding paths to dataframe
        layouts_df["pov_embedding_path"] = layouts_df["pov_path"].map(pov_emb_dict).fillna("")
        print(f"  Embedded {len([x for x in layouts_df['pov_embedding_path'] if x != ''])} POVs")
    else:
        layouts_df["pov_embedding_path"] = ""
        print("  No POVs to embed")
    
    # Step 4: Embed graphs
    print(f"\n{'='*60}")
    print("Step 4/4: Embedding Graphs")
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
        
        # Embed graphs - this creates embeddings next to the graph files
        create_graph_embeddings(
            manifest_path=temp_graph_manifest,
            taxonomy_path=taxonomy_path,
            output_manifest=output_graph_manifest,
            model_name=graph_model,
            save_format="pt"
        )
        
        # Load the output manifest to get embedding paths
        graph_emb_df = pd.read_csv(output_graph_manifest)
        graph_emb_dict = dict(zip(graph_emb_df["graph_path"], graph_emb_df["embedding_path"]))
        
        # Add embedding paths to dataframe
        layouts_df["graph_embedding_path"] = layouts_df["graph_path"].map(graph_emb_dict).fillna("")
        print(f"  Embedded {len([x for x in layouts_df['graph_embedding_path'] if x != ''])} graphs")
    else:
        layouts_df["graph_embedding_path"] = ""
        print("  No graphs to embed")
    
    # Step 5: Create ControlNet training manifest
    print(f"\n{'='*60}")
    print("Step 5/5: Creating ControlNet Training Manifest")
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
        pov_batch_size=args.pov_batch_size,
        graph_model=args.graph_model,
        layout_batch_size=args.layout_batch_size,
        num_workers=args.num_workers,
        device=args.device,
        handle_scenes_without_pov=args.handle_scenes_without_pov
    )


if __name__ == "__main__":
    main()

