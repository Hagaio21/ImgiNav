#!/usr/bin/env python3
"""
Collect Manifest - FAST VERSION

Optimizations:
1. Scan each directory ONCE and build indexes (instead of glob per room)
2. Use sets for O(1) lookups instead of repeated exists() calls
3. Process both variants in parallel if needed

This should reduce runtime from hours to minutes.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict, Counter
import csv
import time
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Directory Indexing - Scan once, lookup many
# ============================================================================

def build_file_index(directory: Path, pattern: str = "*") -> Dict[str, Path]:
    """
    Build an index of all files in a directory.
    Returns dict mapping filename (stem or full name) to full path.
    """
    index = {}
    if not directory.exists():
        return index
    
    for path in directory.glob(pattern):
        if path.is_file():
            index[path.name] = path
            index[path.stem] = path  # Also index without extension
    
    return index


def build_pov_index(povs_dir: Path, variant: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Build index of all POV files.
    Returns dict mapping (scene_id, room_id) -> list of (pov_id, relative_path)
    
    Filename pattern: {scene_id}_{room_id}_{pov_id}_{variant}_pov.png
    """
    variant_dir = povs_dir / variant
    if not variant_dir.exists():
        return {}
    
    index = defaultdict(list)
    suffix = f"_{variant}_pov.png"
    
    logger.info(f"  Indexing POV files in {variant_dir}...")
    start = time.time()
    
    # Single directory scan
    files = list(variant_dir.glob(f"*{suffix}"))
    logger.info(f"    Found {len(files)} POV files")
    
    # Parse filenames to extract scene_id, room_id, pov_id
    # Pattern: {scene_id}_{room_id}_{pov_id}_{variant}_pov.png
    # Example: abc123_Bedroom_door0_tex_pov.png
    #          abc123_Bedroom_1_window0_tex_pov.png
    
    for path in files:
        name = path.stem  # Remove .png
        # Remove _{variant}_pov suffix
        prefix = name.replace(f"_{variant}_pov", "")
        
        # Split into parts - tricky because room_id can contain underscores
        # We know pov_id is at the end and is like "door0", "window1"
        parts = prefix.rsplit("_", 1)
        if len(parts) != 2:
            continue
        
        scene_room = parts[0]
        pov_id = parts[1]
        
        # Now split scene_room into scene_id and room_id
        # Scene ID is typically a UUID-like string, room_id is like "Bedroom" or "Bedroom_1"
        # Try to find the split point
        
        # Strategy: room types are known, find the longest matching suffix
        room_types = ["Bedroom", "LivingRoom", "DiningRoom", "Kitchen", "Bathroom", 
                      "Balcony", "Storage", "Corridor", "OtherRoom", "Library",
                      "MasterBedroom", "SecondBedroom", "KidsRoom", "Study", "Entrance"]
        
        scene_id = None
        room_id = None
        
        for rt in room_types:
            # Check for room_type with index (e.g., "Bedroom_1")
            for pattern in [f"_{rt}_", f"_{rt}"]:
                idx = scene_room.find(pattern)
                if idx != -1:
                    scene_id = scene_room[:idx]
                    room_id = scene_room[idx+1:]
                    break
            if scene_id:
                break
        
        if not scene_id:
            # Fallback: assume first part before any room-like word is scene_id
            # Just split on first underscore that follows a long hex-like sequence
            match = re.match(r'^([a-f0-9-]{8,})_(.+)$', scene_room, re.IGNORECASE)
            if match:
                scene_id = match.group(1)
                room_id = match.group(2)
            else:
                # Last resort: find first underscore
                parts2 = scene_room.split("_", 1)
                if len(parts2) == 2:
                    scene_id, room_id = parts2
                else:
                    continue
        
        if scene_id and room_id:
            relative_path = f"povs/{variant}/{path.name}"
            index[(scene_id, room_id)].append((pov_id, relative_path))
    
    # Sort POVs for each room
    for key in index:
        index[key].sort(key=lambda x: (
            0 if x[0].startswith("door") else 1,
            int(''.join(filter(str.isdigit, x[0])) or 0)
        ))
    
    elapsed = time.time() - start
    logger.info(f"    Indexed {len(index)} room POV sets in {elapsed:.2f}s")
    
    return dict(index)


def build_layout_index(layouts_dir: Path, variant: str) -> Tuple[Set[str], Set[str]]:
    """
    Build index of layout files.
    Returns (scene_layouts, room_layouts) as sets of identifiers.
    
    Scene layout: {scene_id}_{variant}_layout.png
    Room layout: {scene_id}_{room_id}_{variant}_layout.png
    """
    variant_dir = layouts_dir / variant
    if not variant_dir.exists():
        return set(), set()
    
    logger.info(f"  Indexing layout files in {variant_dir}...")
    start = time.time()
    
    suffix = f"_{variant}_layout.png"
    files = list(variant_dir.glob(f"*{suffix}"))
    logger.info(f"    Found {len(files)} layout files")
    
    scene_layouts = set()
    room_layouts = set()
    
    for path in files:
        name = path.stem.replace(f"_{variant}_layout", "")
        
        # Check if it's a scene layout (no room type) or room layout
        # Scene layouts: just scene_id
        # Room layouts: scene_id_room_id
        
        # Simple heuristic: if the name contains a known room type, it's a room layout
        room_types = ["Bedroom", "LivingRoom", "DiningRoom", "Kitchen", "Bathroom",
                      "Balcony", "Storage", "Corridor", "OtherRoom", "Library",
                      "MasterBedroom", "SecondBedroom", "KidsRoom", "Study", "Entrance"]
        
        is_room = any(rt in name for rt in room_types)
        
        if is_room:
            room_layouts.add(name)
        else:
            scene_layouts.add(name)
    
    elapsed = time.time() - start
    logger.info(f"    Indexed {len(scene_layouts)} scene + {len(room_layouts)} room layouts in {elapsed:.2f}s")
    
    return scene_layouts, room_layouts


def build_graph_index(graphs_dir: Path) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Build index of graph files (jsons, texts, embeddings).
    Returns (json_index, text_index, embedding_index) mapping identifier -> relative path.
    """
    json_index = {}
    text_index = {}
    embedding_index = {}
    
    # JSON files
    jsons_dir = graphs_dir / "jsons"
    if jsons_dir.exists():
        logger.info(f"  Indexing graph JSON files...")
        for path in jsons_dir.glob("*.json"):
            # Remove suffixes like _scene_graph, _room_graph
            name = path.stem
            for suffix in ["_scene_graph", "_room_graph", "_graph"]:
                name = name.replace(suffix, "")
            json_index[name] = f"graphs/jsons/{path.name}"
    
    # Text files
    texts_dir = graphs_dir / "texts"
    if texts_dir.exists():
        logger.info(f"  Indexing graph text files...")
        for path in texts_dir.glob("*.txt"):
            name = path.stem
            for suffix in ["_scene_description", "_room_description", "_description", "_text"]:
                name = name.replace(suffix, "")
            text_index[name] = f"graphs/texts/{path.name}"
    
    # Embedding files
    embeddings_dir = graphs_dir / "embeddings"
    if embeddings_dir.exists():
        logger.info(f"  Indexing graph embedding files...")
        for path in embeddings_dir.glob("*.pt"):
            name = path.stem
            for suffix in ["_text", "_description"]:
                name = name.replace(suffix, "")
            embedding_index[name] = f"graphs/embeddings/{path.name}"
    
    logger.info(f"    Indexed {len(json_index)} JSONs, {len(text_index)} texts, {len(embedding_index)} embeddings")
    
    return json_index, text_index, embedding_index


# ============================================================================
# Metadata Loading
# ============================================================================

def load_all_scene_metadata(metadata_dir: Path) -> Dict[str, Path]:
    """Load index of all scene metadata files."""
    scenes_dir = metadata_dir / "scenes"
    if not scenes_dir.exists():
        return {}
    
    return {path.stem: path for path in scenes_dir.glob("*.json")}


def load_all_room_metadata(metadata_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load ALL room metadata into memory at once.
    Returns dict mapping (scene_id, room_id) -> room_data
    """
    rooms_dir = metadata_dir / "rooms"
    if not rooms_dir.exists():
        return {}
    
    logger.info("Loading all room metadata into memory...")
    start = time.time()
    
    rooms = {}
    room_files = list(rooms_dir.glob("*.json"))
    
    for room_path in room_files:
        try:
            with open(room_path, "r") as f:
                room_data = json.load(f)
            
            scene_id = room_data.get("scene_id")
            room_id = room_data.get("room_id")
            
            if scene_id and room_id:
                rooms[(scene_id, room_id)] = room_data
        except Exception as e:
            logger.warning(f"Failed to load {room_path}: {e}")
    
    elapsed = time.time() - start
    logger.info(f"  Loaded {len(rooms)} room metadata files in {elapsed:.2f}s")
    
    return rooms


# ============================================================================
# Main Collection (Fast Version)
# ============================================================================

def collect_manifest_data_fast(dataset_root: Path, variant: str) -> List[Dict[str, Any]]:
    """
    Collect manifest data using pre-built indexes.
    Much faster than scanning directories per-room.
    """
    start_time = time.time()
    
    # Directory paths
    metadata_dir = dataset_root / "metadata"
    layouts_dir = dataset_root / "layouts"
    povs_dir = dataset_root / "povs"
    graphs_dir = dataset_root / "graphs"
    
    # Build indexes ONCE
    logger.info(f"\nBuilding file indexes for variant: {variant}")
    index_start = time.time()
    
    pov_index = build_pov_index(povs_dir, variant)
    scene_layout_set, room_layout_set = build_layout_index(layouts_dir, variant)
    json_index, text_index, embedding_index = build_graph_index(graphs_dir)
    
    index_elapsed = time.time() - index_start
    logger.info(f"Index building complete in {index_elapsed:.2f}s")
    
    # Load all metadata
    logger.info("\nLoading metadata...")
    scene_meta_index = load_all_scene_metadata(metadata_dir)
    room_meta_all = load_all_room_metadata(metadata_dir)
    
    # Group rooms by scene
    rooms_by_scene = defaultdict(list)
    for (scene_id, room_id), room_data in room_meta_all.items():
        rooms_by_scene[scene_id].append((room_id, room_data))
    
    # Now collect rows using indexes (fast lookups)
    logger.info(f"\nCollecting manifest rows...")
    rows = []
    
    total_scenes = len(scene_meta_index)
    scenes_processed = 0
    rooms_processed = 0
    total_povs = 0
    
    for scene_id, scene_meta_path in scene_meta_index.items():
        # Scene-level entry
        scene_layout = None
        if scene_id in scene_layout_set:
            scene_layout = f"layouts/{variant}/{scene_id}_{variant}_layout.png"
        
        scene_graph_json = json_index.get(scene_id)
        scene_graph_text = text_index.get(scene_id)
        
        # Get room count for this scene
        scene_rooms = rooms_by_scene.get(scene_id, [])
        
        if scene_layout:
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
                "is_empty": False,
                "furniture_count": 0,
                "door_count": 0,
                "window_count": 0,
            })
        
        # Room-level entries
        for room_id, room_data in scene_rooms:
            room_type = room_data.get("room_type", "Unknown")
            is_empty = room_data.get("is_empty", False)
            furniture_count = room_data.get("furniture_count", 0)
            door_count = len(room_data.get("doors", []))
            window_count = len(room_data.get("windows", []))
            
            # Layout lookup
            room_key = f"{scene_id}_{room_id}"
            room_layout = None
            if room_key in room_layout_set:
                room_layout = f"layouts/{variant}/{room_key}_{variant}_layout.png"
            
            # Graph files lookup
            room_graph_json = json_index.get(room_key)
            room_graph_text = text_index.get(room_key)
            
            # POV lookup
            povs = pov_index.get((scene_id, room_id), [])
            pov_count = len(povs)
            total_povs += pov_count
            
            if pov_count == 0:
                # Still add room entry without POV
                if room_layout:
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
                # Add entry for each POV
                for pov_id, pov_path in povs:
                    rows.append({
                        "scene_id": scene_id,
                        "type": "room",
                        "room_type": room_type,
                        "room_id": room_id,
                        "pov_id": pov_id,
                        "pov_count": pov_count,
                        "layout_path": room_layout or "",
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
            
            rooms_processed += 1
        
        scenes_processed += 1
        
        # Progress update every 1000 scenes
        if scenes_processed % 1000 == 0:
            elapsed = time.time() - start_time
            rate = scenes_processed / elapsed
            eta = (total_scenes - scenes_processed) / rate
            logger.info(f"  Progress: {scenes_processed}/{total_scenes} scenes "
                       f"({100*scenes_processed/total_scenes:.1f}%) - "
                       f"ETA: {eta:.0f}s")
    
    elapsed_time = time.time() - start_time
    logger.info(f"\nCollection complete for variant: {variant}")
    logger.info(f"  Scenes processed: {scenes_processed}")
    logger.info(f"  Rooms processed: {rooms_processed}")
    logger.info(f"  Total POVs: {total_povs}")
    logger.info(f"  Total rows: {len(rows)}")
    logger.info(f"  Time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} min)")
    
    return rows


def compute_weights(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute sample weights for all rows."""
    if not rows:
        return rows
    
    logger.info(f"Computing weights for {len(rows)} rows...")
    
    # Compute inverse frequency weights
    types = [r["type"] for r in rows]
    is_empties = [r["is_empty"] for r in rows]
    
    type_counts = Counter(types)
    empty_counts = Counter(is_empties)
    
    total = len(rows)
    type_weights = {t: total / (len(type_counts) * c) for t, c in type_counts.items()}
    empty_weights = {e: total / (len(empty_counts) * c) for e, c in empty_counts.items()}
    
    logger.info(f"  Type distribution: {dict(type_counts)}")
    logger.info(f"  Empty distribution: {dict(empty_counts)}")
    
    # Always down-weight empty rooms to reduce their influence on training
    # This helps the model focus on non-empty rooms (which are more informative)
    # We invert the weights: empty rooms get lower weight, non-empty get higher weight
    if len(empty_weights) == 2:
        empty_true_count = empty_counts.get(True, 0)
        empty_false_count = empty_counts.get(False, 0)
        
        # Swap weights so empty rooms (True) always get lower weight
        logger.info(f"  Down-weighting empty rooms: swapping weights (empty: {empty_true_count}, non-empty: {empty_false_count})")
        temp = empty_weights[True]
        empty_weights[True] = empty_weights[False]
        empty_weights[False] = temp
        logger.info(f"  After swap: empty_weight[True]={empty_weights[True]:.6f}, empty_weight[False]={empty_weights[False]:.6f}")
    
    # Apply weights
    for row in rows:
        pov_count = row["pov_count"]
        pov_weight = 1.0 / pov_count if pov_count > 0 else 1.0
        type_weight = type_weights[row["type"]]
        empty_weight = empty_weights[row["is_empty"]]
        sample_weight = pov_weight * type_weight * empty_weight
        
        row["pov_weight"] = round(pov_weight, 6)
        row["type_weight"] = round(type_weight, 6)
        row["empty_weight"] = round(empty_weight, 6)
        row["sample_weight"] = round(sample_weight, 6)
    
    return rows


def write_manifest(rows: List[Dict[str, Any]], output_path: Path):
    """Write manifest to CSV file."""
    if not rows:
        logger.warning(f"No rows to write")
        return
    
    logger.info(f"Writing {len(rows)} rows to {output_path}")
    
    columns = [
        "scene_id", "type", "room_type", "room_id", "pov_id", "pov_count",
        "layout_path", "pov_path", "pov_embedding_path",
        "graph_json_path", "graph_text_path", "graph_embedding_path",
        "is_empty", "furniture_count", "door_count", "window_count",
        "pov_weight", "type_weight", "empty_weight", "sample_weight"
    ]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"  Written: {size_mb:.2f} MB")


def print_statistics(rows: List[Dict[str, Any]], variant: str):
    """Print manifest statistics."""
    if not rows:
        return
    
    total = len(rows)
    scenes = sum(1 for r in rows if r["type"] == "scene")
    rooms = sum(1 for r in rows if r["type"] == "room")
    empty = sum(1 for r in rows if r["is_empty"])
    with_pov = sum(1 for r in rows if r["pov_path"])
    
    unique_scenes = len(set(r["scene_id"] for r in rows))
    unique_rooms = len(set((r["scene_id"], r["room_id"]) for r in rows if r["type"] == "room"))
    
    room_types = Counter(r["room_type"] for r in rows if r["type"] == "room")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Manifest Statistics ({variant})")
    logger.info(f"{'='*50}")
    logger.info(f"Total rows: {total}")
    logger.info(f"  Scene rows: {scenes}")
    logger.info(f"  Room rows: {rooms}")
    logger.info(f"Unique scenes: {unique_scenes}")
    logger.info(f"Unique rooms: {unique_rooms}")
    logger.info(f"Empty rooms: {empty} ({100*empty/total:.1f}%)")
    logger.info(f"With POV: {with_pov} ({100*with_pov/total:.1f}%)")
    logger.info(f"\nRoom types:")
    for rt, count in room_types.most_common(10):
        logger.info(f"  {rt}: {count}")
    logger.info(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate manifest CSV (FAST version)")
    parser.add_argument("--dataset-root", required=True, help="Root directory of dataset")
    parser.add_argument("--output-tex", default=None)
    parser.add_argument("--output-seg", default=None)
    parser.add_argument("--tex-only", action="store_true")
    parser.add_argument("--seg-only", action="store_true")
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        logger.error(f"Dataset root not found: {dataset_root}")
        return
    
    manifests_dir = dataset_root / "manifests"
    output_tex = Path(args.output_tex) if args.output_tex else manifests_dir / "manifest_tex.csv"
    output_seg = Path(args.output_seg) if args.output_seg else manifests_dir / "manifest_seg.csv"
    
    variants = []
    if not args.seg_only:
        variants.append(("tex", output_tex))
    if not args.tex_only:
        variants.append(("seg", output_seg))
    
    total_start = time.time()
    
    for variant, output_path in variants:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing variant: {variant}")
        logger.info(f"{'='*60}")
        
        rows = collect_manifest_data_fast(dataset_root, variant)
        rows = compute_weights(rows)
        print_statistics(rows, variant)
        write_manifest(rows, output_path)
    
    total_elapsed = time.time() - total_start
    logger.info(f"\n{'='*60}")
    logger.info(f"DONE! Total time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()