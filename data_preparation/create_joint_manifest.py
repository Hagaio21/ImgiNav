#!/usr/bin/env python3
"""
Create a joint manifest that combines layouts, POVs, graphs, graph texts, and graph embeddings.

This script reads the cleaned layouts manifest and collects:
- For room layouts: POV images, graph JSON, graph text, and graph embeddings
- For scene layouts: graph JSON, graph text, and graph embeddings (no POVs)

Usage:
    python create_joint_manifest.py \
        --layouts-manifest /work3/s233249/ImgiNav/datasets/layouts_cleaned.csv \
        --data-root /work3/s233249/ImgiNav/datasets/scenes \
        --output /work3/s233249/ImgiNav/datasets/joint_manifest.csv \
        --output-dir /work3/s233249/ImgiNav/datasets/collected
"""

import argparse
import csv
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import sys

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent))

from common.file_io import read_manifest, create_manifest


def find_graph_files(layout_path: Path, scene_id: str, layout_type: str, room_id: str) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    Find graph JSON, text, and embedding files for a layout.
    
    Returns:
        (graph_json_path, graph_text_path, graph_embedding_path)
    """
    layout_dir = layout_path.parent
    
    if layout_type == "scene":
        # Scene graph files
        graph_json = layout_dir / f"{scene_id}_scene_graph.json"
        graph_txt = layout_dir / f"{scene_id}_scene_graph.txt"
        graph_emb_pt = layout_dir / f"{scene_id}_scene_graph.pt"
        graph_emb_npy = layout_dir / f"{scene_id}_scene_graph.npy"
        
        # Also check parent directory
        if not graph_json.exists():
            parent_dir = layout_dir.parent
            graph_json = parent_dir / f"{scene_id}_scene_graph.json"
            graph_txt = parent_dir / f"{scene_id}_scene_graph.txt"
            graph_emb_pt = parent_dir / f"{scene_id}_scene_graph.pt"
            graph_emb_npy = parent_dir / f"{scene_id}_scene_graph.npy"
    else:
        # Room graph files
        graph_json = layout_dir / f"{scene_id}_{room_id}_graph.json"
        graph_txt = layout_dir / f"{scene_id}_{room_id}_graph.txt"
        graph_emb_pt = layout_dir / f"{scene_id}_{room_id}_graph.pt"
        graph_emb_npy = layout_dir / f"{scene_id}_{room_id}_graph.npy"
        
        # Also check room directory (parent of layouts)
        if not graph_json.exists():
            room_dir = layout_dir.parent
            graph_json = room_dir / f"{scene_id}_{room_id}_graph.json"
            graph_txt = room_dir / f"{scene_id}_{room_id}_graph.txt"
            graph_emb_pt = room_dir / f"{scene_id}_{room_id}_graph.pt"
            graph_emb_npy = room_dir / f"{scene_id}_{room_id}_graph.npy"
    
    # Choose embedding file (prefer .pt over .npy)
    graph_emb = graph_emb_pt if graph_emb_pt.exists() else (graph_emb_npy if graph_emb_npy.exists() else None)
    
    graph_json = graph_json if graph_json.exists() else None
    graph_txt = graph_txt if graph_txt.exists() else None
    
    return graph_json, graph_txt, graph_emb


def find_pov_images(data_root: Path, scene_id: str, room_id: str) -> List[Dict[str, str]]:
    """
    Find all POV images (tex and seg) for a room.
    
    Returns:
        List of dicts with 'pov_path' and 'pov_type' keys
    """
    povs = []
    
    # Try to find room directory
    room_dir = data_root / scene_id / "rooms" / room_id
    if not room_dir.exists():
        # Try alternative structure
        room_dir = data_root / scene_id / "rooms" / f"room_id={room_id}"
    
    if not room_dir.exists():
        return povs
    
    povs_dir = room_dir / "povs"
    if not povs_dir.exists():
        return povs
    
    # Find texture POVs
    tex_dir = povs_dir / "tex"
    if tex_dir.exists():
        for pov_file in sorted(tex_dir.glob(f"{scene_id}_{room_id}_*_pov_tex.png")):
            povs.append({
                "pov_path": str(pov_file.resolve()),
                "pov_type": "tex"
            })
    
    # Find segmentation POVs
    seg_dir = povs_dir / "seg"
    if seg_dir.exists():
        for pov_file in sorted(seg_dir.glob(f"{scene_id}_{room_id}_*_pov_seg.png")):
            povs.append({
                "pov_path": str(pov_file.resolve()),
                "pov_type": "seg"
            })
    
    return povs


def copy_file_safe(source: Path, dest: Path) -> bool:
    """
    Copy a file to destination, creating parent directories if needed.
    Returns True if successful, False otherwise.
    """
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            shutil.copy2(source, dest)
            return True
        return False
    except Exception as e:
        print(f"[warn] Failed to copy {source} to {dest}: {e}")
        return False


def create_joint_manifest(
    layouts_manifest: Path,
    data_root: Path,
    output_path: Path,
    output_dir: Optional[Path] = None
) -> None:
    """
    Create a joint manifest combining layouts, POVs, graphs, graph texts, and embeddings.
    Optionally copies POVs and graphs to separate folders.
    
    Args:
        layouts_manifest: Path to cleaned layouts manifest CSV
        data_root: Root directory of the dataset (where scenes are stored)
        output_path: Path to output joint manifest CSV
        output_dir: Optional directory to copy graphs and povs to (creates graphs/ and povs/ subdirs)
    """
    print(f"Reading layouts manifest from {layouts_manifest}...")
    layout_rows = read_manifest(layouts_manifest)
    print(f"Found {len(layout_rows)} layout entries")
    
    data_root = Path(data_root)
    
    # Set up output directories for copying
    graphs_dir = None
    povs_tex_dir = None
    povs_seg_dir = None
    if output_dir:
        output_dir = Path(output_dir)
        graphs_dir = output_dir / "graphs"
        povs_tex_dir = output_dir / "povs" / "tex"
        povs_seg_dir = output_dir / "povs" / "seg"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        povs_tex_dir.mkdir(parents=True, exist_ok=True)
        povs_seg_dir.mkdir(parents=True, exist_ok=True)
        print(f"Will copy graphs to: {graphs_dir}")
        print(f"Will copy textured POVs to: {povs_tex_dir}")
        print(f"Will copy segmented POVs to: {povs_seg_dir}")
    
    # Track copied files to avoid duplicates
    copied_graphs = {}  # source_path -> dest_path
    copied_povs = {}  # source_path -> dest_path
    
    joint_rows = []
    
    for layout_row in tqdm(layout_rows, desc="Processing layouts"):
        scene_id = layout_row.get("scene_id", "")
        layout_type = layout_row.get("type", "")
        room_id = layout_row.get("room_id", "")
        layout_path_str = layout_row.get("layout_path", "")
        is_empty = layout_row.get("is_empty", "0")
        
        if not layout_path_str:
            print(f"[warn] Skipping row with no layout_path: {scene_id}/{room_id}")
            continue
        
        layout_path = Path(layout_path_str)
        # Try to resolve relative paths
        if not layout_path.is_absolute():
            # Try relative to data_root
            layout_path = data_root / layout_path
        if not layout_path.exists():
            print(f"[warn] Layout file does not exist: {layout_path}")
            continue
        
        # Find graph files
        graph_json, graph_txt, graph_emb = find_graph_files(
            layout_path, scene_id, layout_type, room_id
        )
        
        # Copy graph files if output_dir is specified
        graph_json_path = ""
        graph_txt_path = ""
        graph_emb_path = ""
        
        if graph_json:
            if graphs_dir:
                # Copy graph JSON - only use copied path
                if graph_json not in copied_graphs:
                    graph_filename = f"{scene_id}_{room_id}_graph.json" if layout_type == "room" else f"{scene_id}_scene_graph.json"
                    dest_path = graphs_dir / graph_filename
                    if copy_file_safe(graph_json, dest_path):
                        copied_graphs[graph_json] = dest_path
                        graph_json_path = str(dest_path.resolve())
                    # If copy failed, leave path empty (point to copy only)
                else:
                    graph_json_path = str(copied_graphs[graph_json].resolve())
            else:
                # No copying - use original path
                graph_json_path = str(graph_json.resolve())
        
        if graph_txt:
            if graphs_dir:
                # Copy graph text - only use copied path
                if graph_txt not in copied_graphs:
                    graph_filename = f"{scene_id}_{room_id}_graph.txt" if layout_type == "room" else f"{scene_id}_scene_graph.txt"
                    dest_path = graphs_dir / graph_filename
                    if copy_file_safe(graph_txt, dest_path):
                        copied_graphs[graph_txt] = dest_path
                        graph_txt_path = str(dest_path.resolve())
                    # If copy failed, leave path empty (point to copy only)
                else:
                    graph_txt_path = str(copied_graphs[graph_txt].resolve())
            else:
                # No copying - use original path
                graph_txt_path = str(graph_txt.resolve())
        
        if graph_emb:
            if graphs_dir:
                # Copy graph embedding - only use copied path
                if graph_emb not in copied_graphs:
                    ext = graph_emb.suffix
                    graph_filename = f"{scene_id}_{room_id}_graph{ext}" if layout_type == "room" else f"{scene_id}_scene_graph{ext}"
                    dest_path = graphs_dir / graph_filename
                    if copy_file_safe(graph_emb, dest_path):
                        copied_graphs[graph_emb] = dest_path
                        graph_emb_path = str(dest_path.resolve())
                    # If copy failed, leave path empty (point to copy only)
                else:
                    graph_emb_path = str(copied_graphs[graph_emb].resolve())
            else:
                # No copying - use original path
                graph_emb_path = str(graph_emb.resolve())
        
        if layout_type == "room":
            # Room layout: collect POVs
            povs = find_pov_images(data_root, scene_id, room_id)
            
            if povs:
                # Create one row per POV
                for pov in povs:
                    pov_source = Path(pov["pov_path"])
                    pov_dest_path = ""
                    
                    # Copy POV if output_dir is specified
                    # Determine which directory to use based on POV type
                    target_povs_dir = povs_tex_dir if pov["pov_type"] == "tex" else povs_seg_dir
                    
                    if target_povs_dir:
                        # Only use copied path when copying is enabled
                        if pov_source.exists():
                            if pov_source not in copied_povs:
                                # Use original filename, organized by type
                                dest_path = target_povs_dir / pov_source.name
                                if copy_file_safe(pov_source, dest_path):
                                    copied_povs[pov_source] = dest_path
                                    pov_dest_path = str(dest_path.resolve())
                                # If copy failed, leave path empty (point to copy only)
                            else:
                                pov_dest_path = str(copied_povs[pov_source].resolve())
                        # If source doesn't exist, leave path empty
                    else:
                        # No copying - use original path (ensure absolute)
                        pov_dest_path = str(pov_source.resolve()) if pov_source.exists() else ""
                    
                    joint_rows.append({
                        "scene_id": scene_id,
                        "type": layout_type,
                        "room_id": room_id,
                        "layout_path": str(layout_path.resolve()),
                        "is_empty": is_empty,
                        "pov_path": pov_dest_path,
                        "pov_type": pov["pov_type"],
                        "graph_path": graph_json_path,
                        "graph_text_path": graph_txt_path,
                        "graph_embedding_path": graph_emb_path,
                    })
            else:
                # No POVs found, create one row without POV info
                joint_rows.append({
                    "scene_id": scene_id,
                    "type": layout_type,
                    "room_id": room_id,
                    "layout_path": str(layout_path.resolve()),
                    "is_empty": is_empty,
                    "pov_path": "",
                    "pov_type": "",
                    "graph_path": graph_json_path,
                    "graph_text_path": graph_txt_path,
                    "graph_embedding_path": graph_emb_path,
                })
        else:
            # Scene layout: no POVs
            joint_rows.append({
                "scene_id": scene_id,
                "type": layout_type,
                "room_id": room_id,
                "layout_path": str(layout_path.resolve()),
                "is_empty": is_empty,
                "pov_path": "",
                "pov_type": "",
                "graph_path": graph_json_path,
                "graph_text_path": graph_txt_path,
                "graph_embedding_path": graph_emb_path,
            })
    
    # Write output manifest
    fieldnames = [
        "scene_id", "type", "room_id", "layout_path", "is_empty",
        "pov_path", "pov_type",
        "graph_path", "graph_text_path", "graph_embedding_path"
    ]
    
    print(f"\nWriting joint manifest to {output_path}...")
    create_manifest(joint_rows, output_path, fieldnames)
    
    # Print statistics
    total_rows = len(joint_rows)
    room_rows = sum(1 for r in joint_rows if r["type"] == "room")
    scene_rows = sum(1 for r in joint_rows if r["type"] == "scene")
    rows_with_povs = sum(1 for r in joint_rows if r["pov_path"])
    rows_with_graphs = sum(1 for r in joint_rows if r["graph_path"])
    rows_with_graph_texts = sum(1 for r in joint_rows if r["graph_text_path"])
    rows_with_graph_embeddings = sum(1 for r in joint_rows if r["graph_embedding_path"])
    
    print(f"\n✓ Joint manifest created successfully!")
    print(f"  Total rows: {total_rows}")
    print(f"  Room layouts: {room_rows}")
    print(f"  Scene layouts: {scene_rows}")
    print(f"  Rows with POVs: {rows_with_povs}")
    print(f"  Rows with graphs: {rows_with_graphs}")
    print(f"  Rows with graph texts: {rows_with_graph_texts}")
    print(f"  Rows with graph embeddings: {rows_with_graph_embeddings}")
    
    if graphs_dir:
        print(f"\n✓ Copied {len(copied_graphs)} graph files to {graphs_dir}")
    if povs_tex_dir or povs_seg_dir:
        tex_count = 0
        seg_count = 0
        for dest_path in copied_povs.values():
            if povs_tex_dir and str(povs_tex_dir) in str(dest_path):
                tex_count += 1
            elif povs_seg_dir and str(povs_seg_dir) in str(dest_path):
                seg_count += 1
        print(f"✓ Copied {len(copied_povs)} POV files:")
        if povs_tex_dir:
            print(f"  - {tex_count} textured POVs to {povs_tex_dir}")
        if povs_seg_dir:
            print(f"  - {seg_count} segmented POVs to {povs_seg_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a joint manifest combining layouts, POVs, graphs, graph texts, and embeddings"
    )
    parser.add_argument(
        "--layouts-manifest",
        type=Path,
        required=True,
        help="Path to cleaned layouts manifest CSV"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory of the dataset (where scenes are stored)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output joint manifest CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to copy graphs and POVs to (creates graphs/ and povs/ subdirectories)"
    )
    
    args = parser.parse_args()
    
    if not args.layouts_manifest.exists():
        print(f"[error] Layouts manifest not found: {args.layouts_manifest}")
        sys.exit(1)
    
    if not args.data_root.exists():
        print(f"[error] Data root directory not found: {args.data_root}")
        sys.exit(1)
    
    create_joint_manifest(
        args.layouts_manifest,
        args.data_root,
        args.output,
        args.output_dir
    )


if __name__ == "__main__":
    main()

