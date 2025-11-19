#!/usr/bin/env python3
"""
Create ControlNet training manifest from joint manifest.

This script works with the joint manifest that contains:
- layout_path, layout embeddings (latent_path)
- pov_path, pov_embedding_path
- graph_path, graph_embedding_path
- type (room/scene)

For rooms: Creates one training sample per POV (with POV embedding)
For scenes: Creates one training sample per scene (without POV embedding, or with zero/empty POV)

Usage:
    python create_controlnet_manifest_from_joint.py \
        --joint-manifest datasets/joint_manifest_with_embeddings.csv \
        --layouts-latent-manifest datasets/layouts_with_latents.csv \
        --output datasets/controlnet_training_manifest.csv \
        --handle-scenes-without-pov skip  # or "zero" or "empty"
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import Literal
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from common.file_io import read_manifest, create_manifest


def create_controlnet_manifest_from_joint(
    joint_manifest: Path,
    layouts_latent_manifest: Path,
    output_manifest: Path,
        handle_scenes_without_pov: Literal["skip", "zero", "empty"] = "zero"
) -> None:
    """
    Create ControlNet training manifest from joint manifest.
    
    Args:
        joint_manifest: Joint manifest with POV and graph embeddings
        layouts_latent_manifest: Layouts manifest with latent_path (layout embeddings)
        output_manifest: Output path for ControlNet training manifest
        handle_scenes_without_pov: How to handle scenes without POVs
            - "skip": Skip scene samples (only train on rooms)
            - "zero": Use zero POV embedding for scenes
            - "empty": Use empty string (dataset will need to handle this)
    """
    print(f"\nLoading manifests...")
    
    # Load joint manifest
    joint_df = pd.read_csv(joint_manifest)
    print(f"  Joint manifest: {len(joint_df)} rows")
    
    # Load layouts with latents
    layouts_df = pd.read_csv(layouts_latent_manifest)
    print(f"  Layouts with latents: {len(layouts_df)} rows")
    
    # Filter valid rows
    joint_df = joint_df[
        (joint_df["graph_embedding_path"] != "") & 
        (joint_df["graph_embedding_path"].notna())
    ].copy()
    
    # Merge with layout latents
    # Match on scene_id, type, room_id
    merged_df = joint_df.merge(
        layouts_df[["scene_id", "type", "room_id", "latent_path"]],
        on=["scene_id", "type", "room_id"],
        how="left",
        suffixes=("", "_layout")
    )
    
    # Filter rows with valid layout embeddings
    merged_df = merged_df[
        (merged_df["latent_path"] != "") & 
        (merged_df["latent_path"].notna())
    ].copy()
    
    print(f"\nAfter merging and filtering: {len(merged_df)} rows")
    
    # Create training samples
    training_samples = []
    
    # Process room layouts (with POVs)
    room_df = merged_df[merged_df["type"] == "room"].copy()
    room_df = room_df[
        (room_df["pov_embedding_path"] != "") & 
        (room_df["pov_embedding_path"].notna())
    ].copy()
    
    print(f"\nProcessing room layouts ({len(room_df)} rows)...")
    
    for _, row in room_df.iterrows():
        # Resolve paths to absolute
        try:
            layout_emb = str(Path(row["latent_path"]).resolve())
            pov_emb = str(Path(row["pov_embedding_path"]).resolve())
            graph_emb = str(Path(row["graph_embedding_path"]).resolve())
        except Exception as e:
            print(f"Warning: Could not resolve paths for {row['scene_id']}_{row['room_id']}: {e}")
            continue
        
        training_samples.append({
            "scene_id": row["scene_id"],
            "room_id": row["room_id"],
            "sample_type": "room",
            "pov_type": row.get("pov_type", ""),
            "viewpoint": "",  # Could extract from pov_path if needed
            "layout_embedding": layout_emb,
            "pov_embedding": pov_emb,
            "graph_embedding": graph_emb,
            "layout_path": row.get("layout_path", ""),
            "pov_path": row.get("pov_path", ""),
            "graph_text_path": row.get("graph_text_path", ""),
        })
    
    # Process scene layouts
    scene_df = merged_df[merged_df["type"] == "scene"].copy()
    
    print(f"\nProcessing scene layouts ({len(scene_df)} rows)...")
    print(f"  Handling scenes without POV: {handle_scenes_without_pov}")
    
    for _, row in scene_df.iterrows():
        # Resolve paths to absolute
        try:
            layout_emb = str(Path(row["latent_path"]).resolve())
            graph_emb = str(Path(row["graph_embedding_path"]).resolve())
        except Exception as e:
            print(f"Warning: Could not resolve paths for scene {row['scene_id']}: {e}")
            continue
        
        # Handle POV embedding based on strategy
        # IMPORTANT: We cannot skip scenes - they must be included
        if handle_scenes_without_pov == "zero":
            # Use placeholder path that will be loaded as zeros
            # The dataset loader must handle "ZERO_EMBEDDING" marker by creating
            # a zero tensor with shape matching POV embeddings (512-dim for ResNet18)
            pov_emb = "ZERO_EMBEDDING"  # Special marker - dataset must handle this
        elif handle_scenes_without_pov == "empty":
            pov_emb = ""
        elif handle_scenes_without_pov == "skip":
            # Even if skip is requested, we include scenes but mark them
            # This ensures all data is used
            pov_emb = "ZERO_EMBEDDING"  # Default to zero for scenes
            print(f"Warning: 'skip' mode requested but scenes are included with zero POV embeddings")
        else:
            # Try to use existing POV if available (shouldn't happen for scenes)
            pov_emb = str(Path(row["pov_embedding_path"]).resolve()) if row.get("pov_embedding_path") else "ZERO_EMBEDDING"
        
        training_samples.append({
            "scene_id": row["scene_id"],
            "room_id": row.get("room_id", "0000"),
            "sample_type": "scene",
            "pov_type": "",
            "viewpoint": "",
            "layout_embedding": layout_emb,
            "pov_embedding": pov_emb,
            "graph_embedding": graph_emb,
            "layout_path": row.get("layout_path", ""),
            "pov_path": row.get("pov_path", ""),
            "graph_text_path": row.get("graph_text_path", ""),
        })
    
    # Create output DataFrame
    training_df = pd.DataFrame(training_samples)
    
    if len(training_df) == 0:
        print("\n⚠️  No training samples created! Check your manifests.")
        return
    
    # Filter out samples with missing required embeddings
    training_df = training_df[
        (training_df["layout_embedding"] != "") &
        (training_df["graph_embedding"] != "")
    ].copy()
    
    # For room samples, require POV embedding
    room_samples = training_df[training_df["sample_type"] == "room"].copy()
    room_samples = room_samples[room_samples["pov_embedding"] != ""]
    
    # Scene samples (may or may not have POV depending on strategy)
    scene_samples = training_df[training_df["sample_type"] == "scene"].copy()
    
    # Combine
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
        print(f"  Room training samples: {len(room_samples)}")
        print(f"  Average samples per room layout: {len(room_samples) / max(1, len(unique_room_layouts)):.2f}")
    
    if len(scene_samples) > 0:
        print(f"  Scene training samples: {len(scene_samples)}")


def main():
    parser = argparse.ArgumentParser(
        description="Create ControlNet training manifest from joint manifest"
    )
    parser.add_argument(
        "--joint-manifest",
        type=Path,
        required=True,
        help="Path to joint manifest with POV and graph embeddings"
    )
    parser.add_argument(
        "--layouts-latent-manifest",
        type=Path,
        required=True,
        help="Path to layouts manifest with latent_path column"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for ControlNet training manifest"
    )
    parser.add_argument(
        "--handle-scenes-without-pov",
        type=str,
        default="zero",
        choices=["skip", "zero", "empty"],
        help="How to handle scenes without POVs: zero (default, use zero embedding), empty (empty string). Note: scenes are NEVER skipped."
    )
    
    args = parser.parse_args()
    
    if not args.joint_manifest.exists():
        print(f"[error] Joint manifest not found: {args.joint_manifest}")
        sys.exit(1)
    
    if not args.layouts_latent_manifest.exists():
        print(f"[error] Layouts latent manifest not found: {args.layouts_latent_manifest}")
        sys.exit(1)
    
    create_controlnet_manifest_from_joint(
        args.joint_manifest,
        args.layouts_latent_manifest,
        args.output,
        args.handle_scenes_without_pov
    )
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()

