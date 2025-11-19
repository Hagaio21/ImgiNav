#!/usr/bin/env python3
"""
Embed POVs and graphs from the joint manifest.

This script reads the joint manifest and creates embeddings for:
- POV images (using ResNet18)
- Graph texts (using SentenceTransformer)

Usage:
    python embed_from_joint_manifest.py \
        --joint-manifest /work3/s233249/ImgiNav/datasets/joint_manifest.csv \
        --taxonomy config/taxonomy.json \
        --output-manifest /work3/s233249/ImgiNav/datasets/joint_manifest_with_embeddings.csv
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from common.file_io import read_manifest, create_manifest
from data_preparation.create_embeddings import (
    create_pov_embeddings,
    create_graph_embeddings
)


def embed_from_joint_manifest(
    joint_manifest: Path,
    taxonomy_path: Path,
    output_manifest: Path,
    pov_batch_size: int = 32,
    graph_model: str = "all-MiniLM-L6-v2",
    save_format: str = "pt"
) -> None:
    """
    Create embeddings for POVs and graphs from joint manifest.
    
    Args:
        joint_manifest: Path to joint manifest CSV
        taxonomy_path: Path to taxonomy.json
        output_manifest: Path to output manifest with embeddings
        pov_batch_size: Batch size for POV embedding
        graph_model: SentenceTransformer model name
        save_format: Format to save embeddings ("pt" or "npy")
    """
    print(f"Reading joint manifest from {joint_manifest}...")
    df = pd.read_csv(joint_manifest)
    print(f"Found {len(df)} rows")
    
    # Separate POVs and graphs
    pov_rows = df[df["pov_path"] != ""].copy()
    graph_rows = df[df["graph_path"] != ""].copy()
    
    print(f"\nPOVs to embed: {len(pov_rows)}")
    print(f"Graphs to embed: {len(graph_rows)}")
    
    # Create temporary manifests for embedding scripts
    temp_dir = output_manifest.parent / "temp_embeddings"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    pov_manifest = temp_dir / "povs_for_embedding.csv"
    graph_manifest = temp_dir / "graphs_for_embedding.csv"
    
    # Prepare POV manifest (columns: pov_path or pov_image, is_empty)
    # The create_pov_embeddings function expects 'pov_path' or 'pov_image' column
    pov_df = pov_rows[["pov_path", "is_empty"]].copy()
    pov_df.to_csv(pov_manifest, index=False)
    
    # Prepare graph manifest (columns: graph_path, scene_id, type, room_id)
    graph_df = graph_rows[["graph_path", "scene_id", "type", "room_id"]].copy()
    graph_df.to_csv(graph_manifest, index=False)
    
    # Embed POVs
    if len(pov_df) > 0:
        print(f"\n{'='*60}")
        print("Embedding POV images...")
        print(f"{'='*60}")
        pov_output = temp_dir / "povs_with_embeddings.csv"
        create_pov_embeddings(
            manifest_path=pov_manifest,
            output_manifest=pov_output,
            save_format=save_format,
            batch_size=pov_batch_size
        )
        
        # Load POV embeddings and merge back
        pov_emb_df = pd.read_csv(pov_output)
        pov_emb_dict = {}
        for _, row in pov_emb_df.iterrows():
            pov_path = row.get("pov_path") or row.get("pov_image", "")
            if pov_path:
                pov_emb_dict[pov_path] = row.get("embedding_path", "")
        
        # Add POV embeddings to main dataframe
        df["pov_embedding_path"] = ""
        for idx, row in df.iterrows():
            if row["pov_path"]:
                pov_path = row["pov_path"]
                if pov_path in pov_emb_dict:
                    df.at[idx, "pov_embedding_path"] = pov_emb_dict[pov_path]
    else:
        df["pov_embedding_path"] = ""
        print("\nNo POVs to embed")
    
    # Embed graphs
    if len(graph_df) > 0:
        print(f"\n{'='*60}")
        print("Embedding graphs...")
        print(f"{'='*60}")
        graph_output = temp_dir / "graphs_with_embeddings.csv"
        create_graph_embeddings(
            manifest_path=graph_manifest,
            taxonomy_path=taxonomy_path,
            output_manifest=graph_output,
            model_name=graph_model,
            save_format=save_format
        )
        
        # Load graph embeddings and merge back
        graph_emb_df = pd.read_csv(graph_output)
        graph_emb_dict = {}
        for _, row in graph_emb_df.iterrows():
            graph_path = row.get("graph_path", "")
            if graph_path:
                graph_emb_dict[graph_path] = row.get("embedding_path", "")
        
        # Add graph embeddings to main dataframe
        df["graph_embedding_path"] = ""
        for idx, row in df.iterrows():
            if row["graph_path"]:
                graph_path = row["graph_path"]
                if graph_path in graph_emb_dict:
                    df.at[idx, "graph_embedding_path"] = graph_emb_dict[graph_path]
    else:
        df["graph_embedding_path"] = ""
        print("\nNo graphs to embed")
    
    # Save output manifest
    print(f"\n{'='*60}")
    print("Saving output manifest...")
    print(f"{'='*60}")
    df.to_csv(output_manifest, index=False)
    
    # Print statistics
    povs_with_emb = (df["pov_embedding_path"] != "").sum()
    graphs_with_emb = (df["graph_embedding_path"] != "").sum()
    
    print(f"\n✓ Embedding complete!")
    print(f"  Total rows: {len(df)}")
    print(f"  POVs with embeddings: {povs_with_emb}")
    print(f"  Graphs with embeddings: {graphs_with_emb}")
    print(f"  Output manifest: {output_manifest}")
    
    # Cleanup temp files
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print(f"  Cleaned up temporary files")


def main():
    parser = argparse.ArgumentParser(
        description="Create embeddings for POVs and graphs from joint manifest"
    )
    parser.add_argument(
        "--joint-manifest",
        type=Path,
        required=True,
        help="Path to joint manifest CSV"
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        required=True,
        help="Path to taxonomy.json"
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        required=True,
        help="Path to output manifest with embeddings"
    )
    parser.add_argument(
        "--pov-batch-size",
        type=int,
        default=32,
        help="Batch size for POV embedding (default: 32)"
    )
    parser.add_argument(
        "--graph-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model for graph embedding (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pt",
        choices=["pt", "npy"],
        help="Format to save embeddings (default: pt)"
    )
    
    args = parser.parse_args()
    
    if not args.joint_manifest.exists():
        print(f"[error] Joint manifest not found: {args.joint_manifest}")
        sys.exit(1)
    
    if not args.taxonomy.exists():
        print(f"[error] Taxonomy file not found: {args.taxonomy}")
        sys.exit(1)
    
    embed_from_joint_manifest(
        args.joint_manifest,
        args.taxonomy,
        args.output_manifest,
        args.pov_batch_size,
        args.graph_model,
        args.format
    )


if __name__ == "__main__":
    main()

