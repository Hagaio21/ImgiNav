#!/usr/bin/env python3
"""
Collect Manifest - Generate CSV manifest for VAE training

Scans the dataset directory and creates comprehensive CSV manifests (tex and seg variants)
with all paths and pre-computed sample weights for balanced training.

Usage:
    python collect_manifest.py --dataset-root /path/to/dataset_v2
    
    # Custom output names
    python collect_manifest.py --dataset-root /path/to/dataset_v2 \
        --output-tex manifest_tex.csv --output-seg manifest_seg.csv

Weighting Strategy:
    1. pov_weight = 1/pov_count (normalizes rooms with many POVs)
    2. empty_weight = inverse_frequency(is_empty)
    3. type_weight = inverse_frequency(type)
    4. sample_weight = pov_weight * empty_weight * type_weight

Output CSV columns:
    - scene_id: Scene identifier
    - type: "room" or "scene"
    - room_type: Room type (e.g., "Bedroom") or "scene" for scene-level
    - room_id: Full room identifier (e.g., "Bedroom_1") or scene_id for scene-level
    - pov_id: POV identifier (e.g., "door0", "window1") or empty for scenes
    - pov_count: Number of POVs for this room (0 for scenes)
    - layout_path: Relative path to layout image
    - pov_path: Relative path to POV image (empty for scenes)
    - pov_embedding_path: Relative path to POV embedding (empty for scenes)
    - graph_json_path: Relative path to graph JSON
    - graph_text_path: Relative path to graph text description
    - graph_embedding_path: Relative path to text embedding
    - is_empty: Whether the room is empty (no furniture)
    - furniture_count: Number of furniture items
    - door_count: Number of doors
    - window_count: Number of windows
    - pov_weight: 1/pov_count (1.0 for scenes)
    - empty_weight: Inverse frequency weight for is_empty
    - type_weight: Inverse frequency weight for type
    - sample_weight: Combined weight (pov_weight * empty_weight * type_weight)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import csv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_all_povs(scene_id: str, room_id: str, povs_dir: Path, variant: str) -> List[Tuple[str, str]]:
    """
    Find all POV images for a room.
    
    Returns list of (pov_id, relative_path) tuples.
    """
    variant_dir = povs_dir / variant
    if not variant_dir.exists():
        return []
    
    povs = []
    
    # Find all matching POV files
    pattern = f"{scene_id}_{room_id}_*_{variant}_pov.png"
    for pov_path in variant_dir.glob(pattern):
        # Extract pov_id from filename: {scene_id}_{room_id}_{pov_id}_{variant}_pov.png
        name = pov_path.stem  # Remove .png
        # Remove suffix _{variant}_pov
        prefix = name.replace(f"_{variant}_pov", "")
        # Remove scene_id and room_id prefix
        pov_id = prefix.replace(f"{scene_id}_{room_id}_", "")
        
        relative_path = f"povs/{variant}/{pov_path.name}"
        povs.append((pov_id, relative_path))
    
    # Sort by pov_id for consistency (door0, door1, window0, etc.)
    povs.sort(key=lambda x: (
        0 if x[0].startswith("door") else 1,  # Doors first
        int(''.join(filter(str.isdigit, x[0])) or 0)  # Then by number
    ))
    
    return povs


def find_layout(scene_id: str, room_id: str, layouts_dir: Path, variant: str, is_scene: bool = False) -> Optional[str]:
    """Find layout image for a room or scene."""
    variant_dir = layouts_dir / variant
    if not variant_dir.exists():
        return None
    
    if is_scene:
        # Scene layout: {scene_id}_{variant}_layout.png
        layout_path = variant_dir / f"{scene_id}_{variant}_layout.png"
    else:
        # Room layout: {scene_id}_{room_id}_{variant}_layout.png
        layout_path = variant_dir / f"{scene_id}_{room_id}_{variant}_layout.png"
    
    if layout_path.exists():
        return f"layouts/{variant}/{layout_path.name}"
    
    return None


def find_graph_files(scene_id: str, room_id: str, graphs_dir: Path, is_scene: bool = False) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Find graph JSON, text description, and text embedding for a room or scene.
    
    Returns (json_path, text_path, embedding_path)
    """
    json_path = None
    text_path = None
    embedding_path = None
    
    if is_scene:
        identifier = scene_id
    else:
        identifier = f"{scene_id}_{room_id}"
    
    # Graph JSON
    jsons_dir = graphs_dir / "jsons"
    if jsons_dir.exists():
        # Try different patterns
        for pattern in [f"{identifier}_scene_graph.json", f"{identifier}_room_graph.json", f"{identifier}_graph.json", f"{identifier}.json"]:
            path = jsons_dir / pattern
            if path.exists():
                json_path = f"graphs/jsons/{path.name}"
                break
    
    # Text description
    texts_dir = graphs_dir / "texts"
    if texts_dir.exists():
        for pattern in [f"{identifier}_description.txt", f"{identifier}_text.txt", f"{identifier}.txt"]:
            path = texts_dir / pattern
            if path.exists():
                text_path = f"graphs/texts/{path.name}"
                break
    
    # Text embedding
    embeddings_dir = graphs_dir / "embeddings"
    if embeddings_dir.exists():
        for pattern in [f"{identifier}_text.pt", f"{identifier}_description.pt", f"{identifier}.pt"]:
            path = embeddings_dir / pattern
            if path.exists():
                embedding_path = f"graphs/embeddings/{path.name}"
                break
    
    return json_path, text_path, embedding_path


def load_room_metadata(metadata_dir: Path, scene_id: str) -> Dict[str, Dict[str, Any]]:
    """Load all room metadata for a scene."""
    rooms = {}
    rooms_dir = metadata_dir / "rooms"
    
    if not rooms_dir.exists():
        return rooms
    
    for room_meta_path in rooms_dir.glob(f"{scene_id}_*.json"):
        try:
            with open(room_meta_path, "r") as f:
                room_data = json.load(f)
            
            room_id = room_data.get("room_id", room_meta_path.stem.replace(f"{scene_id}_", ""))
            rooms[room_id] = room_data
        except Exception as e:
            logger.warning(f"Failed to load room metadata {room_meta_path}: {e}")
    
    return rooms


def compute_inverse_frequency_weights(values: List[Any]) -> Dict[Any, float]:
    """
    Compute inverse frequency weights for a list of values.
    
    Weight = total_count / (num_classes * class_count)
    This makes each class contribute equally in expectation.
    """
    from collections import Counter
    counts = Counter(values)
    total = len(values)
    num_classes = len(counts)
    
    weights = {}
    for value, count in counts.items():
        weights[value] = total / (num_classes * count)
    
    return weights


def collect_manifest_data(dataset_root: Path, variant: str) -> List[Dict[str, Any]]:
    """
    Collect all manifest data for a variant (tex or seg).
    
    Returns list of row dictionaries.
    """
    rows = []
    
    # Directory paths
    metadata_dir = dataset_root / "metadata"
    layouts_dir = dataset_root / "layouts"
    povs_dir = dataset_root / "povs"
    graphs_dir = dataset_root / "graphs"
    
    # Find all scenes from metadata
    scenes_dir = metadata_dir / "scenes"
    if not scenes_dir.exists():
        logger.error(f"Scenes metadata directory not found: {scenes_dir}")
        return rows
    
    scene_files = list(scenes_dir.glob("*.json"))
    logger.info(f"Found {len(scene_files)} scenes")
    
    for scene_meta_path in scene_files:
        scene_id = scene_meta_path.stem
        
        try:
            with open(scene_meta_path, "r") as f:
                scene_meta = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load scene metadata {scene_meta_path}: {e}")
            continue
        
        # Load room metadata for this scene
        rooms = load_room_metadata(metadata_dir, scene_id)
        
        # --- Scene-level entry (no POV) ---
        scene_layout = find_layout(scene_id, "", layouts_dir, variant, is_scene=True)
        scene_graph_json, scene_graph_text, _ = find_graph_files(
            scene_id, "", graphs_dir, is_scene=True
        )
        
        if scene_layout:  # Only add if layout exists
            rows.append({
                "scene_id": scene_id,
                "type": "scene",
                "room_type": "scene",
                "room_id": scene_id,
                "pov_id": "",
                "pov_count": 0,
                "layout_path": scene_layout,
                "pov_path": "",
                "pov_embedding_path": "",
                "graph_json_path": scene_graph_json or "",
                "graph_text_path": scene_graph_text or "",
                "graph_embedding_path": "",
                "is_empty": False,  # Scenes are never "empty"
                "furniture_count": scene_meta.get("furniture_count", 0),
                "door_count": scene_meta.get("door_count", 0),
                "window_count": scene_meta.get("window_count", 0),
            })
        
        # --- Room-level entries (with POVs) ---
        for room_id, room_meta in rooms.items():
            room_type = room_meta.get("room_type", "Unknown")
            furniture_count = room_meta.get("furniture_count", len(room_meta.get("furniture", [])))
            is_empty = furniture_count == 0
            door_count = room_meta.get("door_count", len(room_meta.get("doors", [])))
            window_count = room_meta.get("window_count", len(room_meta.get("windows", [])))
            
            # Find layout
            room_layout = find_layout(scene_id, room_id, layouts_dir, variant, is_scene=False)
            if not room_layout:
                continue  # Skip rooms without layouts
            
            # Find graph files
            room_graph_json, room_graph_text, _ = find_graph_files(
                scene_id, room_id, graphs_dir, is_scene=False
            )
            
            # Find all POVs for this room
            povs = find_all_povs(scene_id, room_id, povs_dir, variant)
            pov_count = len(povs)
            
            if pov_count == 0:
                # Room has no POVs - add single entry without POV
                rows.append({
                    "scene_id": scene_id,
                    "type": "room",
                    "room_type": room_type,
                    "room_id": room_id,
                    "pov_id": "",
                    "pov_count": 0,
                    "layout_path": room_layout,
                    "pov_path": "",
                    "pov_embedding_path": "",
                    "graph_json_path": room_graph_json or "",
                    "graph_text_path": room_graph_text or "",
                    "graph_embedding_path": "",
                    "is_empty": is_empty,
                    "furniture_count": furniture_count,
                    "door_count": door_count,
                    "window_count": window_count,
                })
            else:
                # Add one row per POV
                for pov_id, pov_path in povs:
                    rows.append({
                        "scene_id": scene_id,
                        "type": "room",
                        "room_type": room_type,
                        "room_id": room_id,
                        "pov_id": pov_id,
                        "pov_count": pov_count,
                        "layout_path": room_layout,
                        "pov_path": pov_path,
                        "pov_embedding_path": "",
                        "graph_json_path": room_graph_json or "",
                        "graph_text_path": room_graph_text or "",
                        "graph_embedding_path": "",
                        "is_empty": is_empty,
                        "furniture_count": furniture_count,
                        "door_count": door_count,
                        "window_count": window_count,
                    })
    
    return rows


def compute_weights(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compute sample weights for all rows.
    
    Weighting strategy:
    1. pov_weight = 1/pov_count (1.0 for scenes or rooms without POVs)
    2. empty_weight = inverse_frequency(is_empty)
    3. type_weight = inverse_frequency(type)
    4. sample_weight = pov_weight * empty_weight * type_weight
    """
    if not rows:
        return rows
    
    # Compute inverse frequency weights
    types = [r["type"] for r in rows]
    is_empties = [r["is_empty"] for r in rows]
    
    type_weights = compute_inverse_frequency_weights(types)
    empty_weights = compute_inverse_frequency_weights(is_empties)
    
    logger.info(f"Type weights: {type_weights}")
    logger.info(f"Empty weights: {empty_weights}")
    
    # Apply weights to each row
    for row in rows:
        # POV weight: normalize by number of POVs
        pov_count = row["pov_count"]
        if pov_count > 0:
            pov_weight = 1.0 / pov_count
        else:
            pov_weight = 1.0  # Scenes or rooms without POVs
        
        # Type and empty weights
        type_weight = type_weights[row["type"]]
        empty_weight = empty_weights[row["is_empty"]]
        
        # Combined weight
        sample_weight = pov_weight * type_weight * empty_weight
        
        row["pov_weight"] = round(pov_weight, 6)
        row["type_weight"] = round(type_weight, 6)
        row["empty_weight"] = round(empty_weight, 6)
        row["sample_weight"] = round(sample_weight, 6)
    
    return rows


def write_manifest(rows: List[Dict[str, Any]], output_path: Path):
    """Write manifest to CSV file."""
    if not rows:
        logger.warning(f"No rows to write to {output_path}")
        return
    
    # Column order
    columns = [
        "scene_id", "type", "room_type", "room_id", "pov_id", "pov_count",
        "layout_path", "pov_path", "pov_embedding_path",
        "graph_json_path", "graph_text_path", "graph_embedding_path",
        "is_empty", "furniture_count", "door_count", "window_count",
        "pov_weight", "type_weight", "empty_weight", "sample_weight"
    ]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    
    logger.info(f"Wrote {len(rows)} rows to {output_path}")


def print_statistics(rows: List[Dict[str, Any]], variant: str):
    """Print manifest statistics."""
    if not rows:
        return
    
    total = len(rows)
    scenes = sum(1 for r in rows if r["type"] == "scene")
    rooms = sum(1 for r in rows if r["type"] == "room")
    empty = sum(1 for r in rows if r["is_empty"])
    with_pov = sum(1 for r in rows if r["pov_path"])
    
    # Unique counts
    unique_scenes = len(set(r["scene_id"] for r in rows))
    unique_rooms = len(set((r["scene_id"], r["room_id"]) for r in rows if r["type"] == "room"))
    
    # Room type distribution
    from collections import Counter
    room_types = Counter(r["room_type"] for r in rows if r["type"] == "room")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Manifest Statistics ({variant})")
    logger.info(f"{'='*50}")
    logger.info(f"Total rows: {total}")
    logger.info(f"  - Scene rows: {scenes}")
    logger.info(f"  - Room rows: {rooms}")
    logger.info(f"Unique scenes: {unique_scenes}")
    logger.info(f"Unique rooms: {unique_rooms}")
    logger.info(f"Empty rooms: {empty} ({100*empty/total:.1f}%)")
    logger.info(f"Rows with POV: {with_pov} ({100*with_pov/total:.1f}%)")
    logger.info(f"\nRoom type distribution:")
    for room_type, count in room_types.most_common(10):
        logger.info(f"  {room_type}: {count}")
    logger.info(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate manifest CSV for VAE training")
    parser.add_argument("--dataset-root", required=True, help="Root directory of dataset")
    parser.add_argument("--output-tex", default=None, help="Output path for tex manifest (default: dataset_root/manifests/manifest_tex.csv)")
    parser.add_argument("--output-seg", default=None, help="Output path for seg manifest (default: dataset_root/manifests/manifest_seg.csv)")
    parser.add_argument("--tex-only", action="store_true", help="Only generate tex manifest")
    parser.add_argument("--seg-only", action="store_true", help="Only generate seg manifest")
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        logger.error(f"Dataset root not found: {dataset_root}")
        return
    
    # Default output paths
    manifests_dir = dataset_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    
    output_tex = Path(args.output_tex) if args.output_tex else manifests_dir / "manifest_tex.csv"
    output_seg = Path(args.output_seg) if args.output_seg else manifests_dir / "manifest_seg.csv"
    
    # Generate manifests
    variants = []
    if not args.seg_only:
        variants.append(("tex", output_tex))
    if not args.tex_only:
        variants.append(("seg", output_seg))
    
    for variant, output_path in variants:
        logger.info(f"\nCollecting {variant} manifest data...")
        rows = collect_manifest_data(dataset_root, variant)
        
        logger.info(f"Computing sample weights...")
        rows = compute_weights(rows)
        
        print_statistics(rows, variant)
        
        write_manifest(rows, output_path)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()