#!/usr/bin/env python3
"""
Collect Manifest v2 - For POV-Normalized Layouts

Key changes from v1:
- Each datapoint is (POV, POV-specific layout) - no duplicates
- Layout orientation is normalized to POV viewing direction
- Includes rotation metadata in manifest
- Properly pairs POV images with their corresponding rotated layouts

The layout naming convention is:
- Scene layout: {scene_id}_{variant}_layout.png
- Room layout: {scene_id}_{room_id}_{variant}_layout.png
- POV layout: {scene_id}_{room_id}_{pov_id}_{variant}_layout.png

POV image naming:
- {scene_id}_{room_id}_{pov_id}_{variant}_pov.png
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
import math

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Directory Indexing
# ============================================================================

def build_pov_index(povs_dir: Path, variant: str) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
    """
    Build index of all POV files.
    Returns dict mapping (scene_id, room_id) -> list of (pov_id, relative_path)
    """
    variant_dir = povs_dir / variant
    if not variant_dir.exists():
        return {}
    
    index = defaultdict(list)
    suffix = f"_{variant}_pov.png"
    
    logger.info(f"  Indexing POV files in {variant_dir}...")
    start = time.time()
    
    files = list(variant_dir.glob(f"*{suffix}"))
    logger.info(f"    Found {len(files)} POV files")
    
    room_types = ["Bedroom", "LivingRoom", "DiningRoom", "Kitchen", "Bathroom", 
                  "Balcony", "Storage", "Corridor", "OtherRoom", "Library",
                  "MasterBedroom", "SecondBedroom", "KidsRoom", "Study", "Entrance"]
    
    for path in files:
        name = path.stem
        prefix = name.replace(f"_{variant}_pov", "")
        
        parts = prefix.rsplit("_", 1)
        if len(parts) != 2:
            continue
        
        scene_room = parts[0]
        pov_id = parts[1]
        
        scene_id = None
        room_id = None
        
        for rt in room_types:
            for pattern in [f"_{rt}_", f"_{rt}"]:
                idx = scene_room.find(pattern)
                if idx != -1:
                    scene_id = scene_room[:idx]
                    room_id = scene_room[idx+1:]
                    break
            if scene_id:
                break
        
        if not scene_id:
            match = re.match(r'^([a-f0-9-]{8,})_(.+)$', scene_room, re.IGNORECASE)
            if match:
                scene_id = match.group(1)
                room_id = match.group(2)
            else:
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


def build_pov_layout_index(layouts_dir: Path, variant: str) -> Dict[Tuple[str, str, str], str]:
    """
    Build index of POV-specific layout files.
    These are in layouts_pov/ directory (rotated versions of original layouts).
    Returns dict mapping (scene_id, room_id, pov_id) -> relative_path
    """
    # Check layouts_pov directory (new structure from stage4 v2)
    pov_layouts_dir = layouts_dir.parent / "layouts_pov" / variant
    if not pov_layouts_dir.exists():
        # Fallback to old structure in layouts/
        pov_layouts_dir = layouts_dir / variant
    
    if not pov_layouts_dir.exists():
        return {}
    
    logger.info(f"  Indexing POV layout files in {pov_layouts_dir}...")
    start = time.time()
    
    index = {}
    suffix = f"_{variant}_layout.png"
    files = list(pov_layouts_dir.glob(f"*{suffix}"))
    
    # Filter to only POV-specific layouts (contain door/window in name)
    pov_pattern = re.compile(r'(door|window)\d+')
    pov_files = [f for f in files if pov_pattern.search(f.stem)]
    
    logger.info(f"    Found {len(pov_files)} POV layout files (out of {len(files)} total)")
    
    room_types = ["Bedroom", "LivingRoom", "DiningRoom", "Kitchen", "Bathroom", 
                  "Balcony", "Storage", "Corridor", "OtherRoom", "Library",
                  "MasterBedroom", "SecondBedroom", "KidsRoom", "Study", "Entrance"]
    
    for path in pov_files:
        name = path.stem.replace(f"_{variant}_layout", "")
        
        # Extract pov_id (door0, window1, etc.) from the end
        match = pov_pattern.search(name)
        if not match:
            continue
        
        pov_id = match.group(0)
        pov_start = name.rfind(f"_{pov_id}")
        if pov_start == -1:
            continue
        
        scene_room = name[:pov_start]
        
        # Parse scene_id and room_id
        scene_id = None
        room_id = None
        
        for rt in room_types:
            for pattern in [f"_{rt}_", f"_{rt}"]:
                idx = scene_room.find(pattern)
                if idx != -1:
                    scene_id = scene_room[:idx]
                    room_id = scene_room[idx+1:]
                    break
            if scene_id:
                break
        
        if not scene_id:
            match = re.match(r'^([a-f0-9-]{8,})_(.+)$', scene_room, re.IGNORECASE)
            if match:
                scene_id = match.group(1)
                room_id = match.group(2)
        
        if scene_id and room_id:
            # Use layouts_pov path
            relative_path = f"layouts_pov/{variant}/{path.name}"
            index[(scene_id, room_id, pov_id)] = relative_path
    
    elapsed = time.time() - start
    logger.info(f"    Indexed {len(index)} POV layouts in {elapsed:.2f}s")
    
    return index


def build_room_layout_index(layouts_dir: Path, variant: str) -> Dict[Tuple[str, str], str]:
    """
    Build index of room-level layout files (non-POV-specific).
    Returns dict mapping (scene_id, room_id) -> relative_path
    """
    variant_dir = layouts_dir / variant
    if not variant_dir.exists():
        return {}
    
    logger.info(f"  Indexing room layout files in {variant_dir}...")
    start = time.time()
    
    index = {}
    suffix = f"_{variant}_layout.png"
    files = list(variant_dir.glob(f"*{suffix}"))
    
    # Filter to non-POV layouts (no door/window in name but has room type)
    pov_pattern = re.compile(r'(door|window)\d+')
    room_types = ["Bedroom", "LivingRoom", "DiningRoom", "Kitchen", "Bathroom", 
                  "Balcony", "Storage", "Corridor", "OtherRoom", "Library",
                  "MasterBedroom", "SecondBedroom", "KidsRoom", "Study", "Entrance"]
    
    room_files = [f for f in files 
                  if not pov_pattern.search(f.stem) 
                  and any(rt in f.stem for rt in room_types)]
    
    logger.info(f"    Found {len(room_files)} room layout files")
    
    for path in room_files:
        name = path.stem.replace(f"_{variant}_layout", "")
        
        scene_id = None
        room_id = None
        
        for rt in room_types:
            for pattern in [f"_{rt}_", f"_{rt}"]:
                idx = name.find(pattern)
                if idx != -1:
                    scene_id = name[:idx]
                    room_id = name[idx+1:]
                    break
            if scene_id:
                break
        
        if not scene_id:
            match = re.match(r'^([a-f0-9-]{8,})_(.+)$', name, re.IGNORECASE)
            if match:
                scene_id = match.group(1)
                room_id = match.group(2)
        
        if scene_id and room_id:
            relative_path = f"layouts/{variant}/{path.name}"
            index[(scene_id, room_id)] = relative_path
    
    elapsed = time.time() - start
    logger.info(f"    Indexed {len(index)} room layouts in {elapsed:.2f}s")
    
    return index


def build_graph_index(graphs_dir: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build index of room-level graph files (non-POV-specific)."""
    json_index = {}
    text_index = {}
    
    jsons_dir = graphs_dir / "jsons"
    if jsons_dir.exists():
        for path in jsons_dir.glob("*_room_graph.json"):
            name = path.stem.replace("_room_graph", "")
            # Skip POV-specific graphs (contain door/window)
            if not re.search(r'(door|window)\d+', name):
                json_index[name] = f"graphs/jsons/{path.name}"
    
    texts_dir = graphs_dir / "texts"
    if texts_dir.exists():
        for path in texts_dir.glob("*_room_description.txt"):
            name = path.stem.replace("_room_description", "")
            if not re.search(r'(door|window)\d+', name):
                text_index[name] = f"graphs/texts/{path.name}"
    
    return json_index, text_index


def build_pov_graph_index(graphs_dir: Path) -> Dict[Tuple[str, str, str], Tuple[str, str]]:
    """
    Build index of POV-specific graph files.
    Returns dict mapping (scene_id, room_id, pov_id) -> (json_path, text_path)
    """
    index = {}
    
    jsons_dir = graphs_dir / "jsons"
    if not jsons_dir.exists():
        return index
    
    logger.info(f"  Indexing POV graph files...")
    start = time.time()
    
    # Pattern: {scene_id}_{room_id}_{pov_id}_room_graph.json
    pov_pattern = re.compile(r'(door|window)\d+')
    
    room_types = ["Bedroom", "LivingRoom", "DiningRoom", "Kitchen", "Bathroom", 
                  "Balcony", "Storage", "Corridor", "OtherRoom", "Library",
                  "MasterBedroom", "SecondBedroom", "KidsRoom", "Study", "Entrance"]
    
    for path in jsons_dir.glob("*_room_graph.json"):
        name = path.stem.replace("_room_graph", "")
        
        # Must contain POV ID
        match = pov_pattern.search(name)
        if not match:
            continue
        
        pov_id = match.group(0)
        pov_start = name.rfind(f"_{pov_id}")
        if pov_start == -1:
            continue
        
        scene_room = name[:pov_start]
        
        # Parse scene_id and room_id
        scene_id = None
        room_id = None
        
        for rt in room_types:
            for pattern in [f"_{rt}_", f"_{rt}"]:
                idx = scene_room.find(pattern)
                if idx != -1:
                    scene_id = scene_room[:idx]
                    room_id = scene_room[idx+1:]
                    break
            if scene_id:
                break
        
        if not scene_id:
            match = re.match(r'^([a-f0-9-]{8,})_(.+)$', scene_room, re.IGNORECASE)
            if match:
                scene_id = match.group(1)
                room_id = match.group(2)
        
        if scene_id and room_id:
            json_path = f"graphs/jsons/{path.name}"
            text_path = f"graphs/texts/{name}_room_description.txt"
            index[(scene_id, room_id, pov_id)] = (json_path, text_path)
    
    elapsed = time.time() - start
    logger.info(f"    Indexed {len(index)} POV graphs in {elapsed:.2f}s")
    
    return index


def load_pov_layouts_info(dataset_root: Path) -> Dict[Tuple[str, str, str], Dict]:
    """
    Load POV layout metadata from pov_info.json (generated by stage4 v2).
    Returns dict mapping (scene_id, room_id, pov_id) -> metadata
    """
    # Try new location first (stage4 v2)
    info_path = dataset_root / "povs" / "pov_info.json"
    if not info_path.exists():
        # Fallback to old location
        info_path = dataset_root / "layouts" / "pov_layouts_info.json"
    
    if not info_path.exists():
        return {}
    
    logger.info(f"  Loading POV info from {info_path}...")
    
    with open(info_path, "r") as f:
        info_list = json.load(f)
    
    index = {}
    for info in info_list:
        key = (info["scene_id"], info["room_id"], info["pov_id"])
        index[key] = info
    
    logger.info(f"    Loaded {len(index)} POV info entries")
    return index


# ============================================================================
# Manifest Collection
# ============================================================================

def collect_manifest_data_pov_normalized(
    dataset_root: Path,
    variant: str
) -> List[Dict[str, Any]]:
    """
    Collect manifest data with POV-normalized layouts and graphs.
    
    Each row represents a unique (POV, layout, graph) triple where:
    - Layout is rotated to match the POV viewing direction
    - Graph has POV-relative spatial relations and descriptive object names
    """
    logger.info(f"\nCollecting manifest data for variant: {variant}")
    
    # Build indexes
    metadata_dir = dataset_root / "metadata"
    povs_dir = dataset_root / "povs"
    layouts_dir = dataset_root / "layouts"
    graphs_dir = dataset_root / "graphs"
    
    # Index POV files
    pov_index = build_pov_index(povs_dir, variant)
    
    # Index POV-specific layouts
    pov_layout_index = build_pov_layout_index(layouts_dir, variant)
    
    # Index room layouts (fallback)
    room_layout_index = build_room_layout_index(layouts_dir, variant)
    
    # Load POV layout metadata
    pov_layout_info = load_pov_layouts_info(layouts_dir)
    
    # Index POV-specific graphs
    pov_graph_index = build_pov_graph_index(graphs_dir)
    
    # Index room graphs (fallback)
    graph_json_index, graph_text_index = build_graph_index(graphs_dir)
    
    # Load room metadata
    rooms_dir = metadata_dir / "rooms"
    if not rooms_dir.exists():
        logger.warning(f"Rooms directory not found: {rooms_dir}")
        return []
    
    room_meta_files = list(rooms_dir.glob("*.json"))
    logger.info(f"  Found {len(room_meta_files)} room metadata files")
    
    rows = []
    start_time = time.time()
    
    for room_meta_path in room_meta_files:
        with open(room_meta_path, "r") as f:
            room_meta = json.load(f)
        
        scene_id = room_meta.get("scene_id", "")
        room_id = room_meta.get("room_id", "")
        room_type = room_meta.get("room_type", "UnknownRoom")
        is_empty = room_meta.get("is_empty", False)
        furniture_count = room_meta.get("furniture_count", 0)
        door_count = len(room_meta.get("doors", []))
        window_count = len(room_meta.get("windows", []))
        
        # Get POVs for this room
        povs = pov_index.get((scene_id, room_id), [])
        
        if not povs:
            continue
        
        # Get room-level paths (fallback)
        room_key = f"{scene_id}_{room_id}"
        room_graph_json = graph_json_index.get(room_key, "")
        room_graph_text = graph_text_index.get(room_key, "")
        room_layout = room_layout_index.get((scene_id, room_id), "")
        
        # Create a row for each POV
        for pov_id, pov_path in povs:
            # Try to get POV-specific layout
            pov_layout_path = pov_layout_index.get((scene_id, room_id, pov_id), "")
            
            # Try to get POV-specific graph
            pov_graph_paths = pov_graph_index.get((scene_id, room_id, pov_id))
            if pov_graph_paths:
                pov_graph_json, pov_graph_text = pov_graph_paths
            else:
                pov_graph_json, pov_graph_text = "", ""
            
            # Get rotation info if available
            pov_info = pov_layout_info.get((scene_id, room_id, pov_id), {})
            rotation_angle_deg = pov_info.get("rotation_angle_deg", 0.0)
            rotation_angle_rad = pov_info.get("rotation_angle_rad", 0.0)
            
            # Use POV-specific paths if available, otherwise room-level
            layout_path = pov_layout_path if pov_layout_path else room_layout
            graph_json_path = pov_graph_json if pov_graph_json else room_graph_json
            graph_text_path = pov_graph_text if pov_graph_text else room_graph_text
            
            is_pov_normalized = bool(pov_layout_path)
            has_pov_graph = bool(pov_graph_json)
            
            row = {
                "scene_id": scene_id,
                "room_type": room_type,
                "room_id": room_id,
                "pov_id": pov_id,
                "pov_type": "door" if pov_id.startswith("door") else "window",
                "pov_index": int(''.join(filter(str.isdigit, pov_id)) or 0),
                "layout_path": layout_path,
                "pov_path": pov_path,
                "is_pov_normalized": is_pov_normalized,
                "has_pov_graph": has_pov_graph,
                "rotation_angle_deg": rotation_angle_deg,
                "rotation_angle_rad": rotation_angle_rad,
                "graph_json_path": graph_json_path,
                "graph_text_path": graph_text_path,
                "is_empty": is_empty,
                "furniture_count": furniture_count,
                "door_count": door_count,
                "window_count": window_count
            }
            
            rows.append(row)
    
    elapsed = time.time() - start_time
    logger.info(f"  Collected {len(rows)} POV-layout-graph triples in {elapsed:.2f}s")
    
    return rows


def compute_weights(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute sample weights for balanced training."""
    if not rows:
        return rows
    
    logger.info(f"Computing weights for {len(rows)} rows...")
    
    # Count distributions
    room_type_counts = Counter(r["room_type"] for r in rows)
    empty_counts = Counter(r["is_empty"] for r in rows)
    pov_type_counts = Counter(r["pov_type"] for r in rows)
    
    total = len(rows)
    
    # Inverse frequency weights
    room_type_weights = {t: total / (len(room_type_counts) * c) for t, c in room_type_counts.items()}
    empty_weights = {e: total / (len(empty_counts) * c) for e, c in empty_counts.items()}
    pov_type_weights = {t: total / (len(pov_type_counts) * c) for t, c in pov_type_counts.items()}
    
    # Down-weight empty rooms
    if len(empty_weights) == 2:
        temp = empty_weights[True]
        empty_weights[True] = empty_weights[False]
        empty_weights[False] = temp
    
    logger.info(f"  Room type distribution: {dict(room_type_counts.most_common(5))}")
    logger.info(f"  Empty distribution: {dict(empty_counts)}")
    logger.info(f"  POV type distribution: {dict(pov_type_counts)}")
    
    # Apply weights
    for row in rows:
        room_weight = room_type_weights[row["room_type"]]
        empty_weight = empty_weights[row["is_empty"]]
        pov_weight = pov_type_weights[row["pov_type"]]
        
        sample_weight = room_weight * empty_weight * pov_weight
        
        row["room_type_weight"] = round(room_weight, 6)
        row["empty_weight"] = round(empty_weight, 6)
        row["pov_type_weight"] = round(pov_weight, 6)
        row["sample_weight"] = round(sample_weight, 6)
    
    return rows


def write_manifest(rows: List[Dict[str, Any]], output_path: Path):
    """Write manifest to CSV file."""
    if not rows:
        logger.warning("No rows to write")
        return
    
    logger.info(f"Writing {len(rows)} rows to {output_path}")
    
    columns = [
        "scene_id", "room_type", "room_id", "pov_id", "pov_type", "pov_index",
        "layout_path", "pov_path", "is_pov_normalized", "has_pov_graph",
        "rotation_angle_deg", "rotation_angle_rad",
        "graph_json_path", "graph_text_path",
        "is_empty", "furniture_count", "door_count", "window_count",
        "room_type_weight", "empty_weight", "pov_type_weight", "sample_weight"
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
    normalized_layouts = sum(1 for r in rows if r["is_pov_normalized"])
    has_pov_graphs = sum(1 for r in rows if r["has_pov_graph"])
    empty = sum(1 for r in rows if r["is_empty"])
    
    unique_scenes = len(set(r["scene_id"] for r in rows))
    unique_rooms = len(set((r["scene_id"], r["room_id"]) for r in rows))
    
    room_types = Counter(r["room_type"] for r in rows)
    pov_types = Counter(r["pov_type"] for r in rows)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Manifest Statistics ({variant}) - POV Normalized")
    logger.info(f"{'='*50}")
    logger.info(f"Total POV datapoints: {total}")
    logger.info(f"Unique scenes: {unique_scenes}")
    logger.info(f"Unique rooms: {unique_rooms}")
    logger.info(f"POV-normalized layouts: {normalized_layouts} ({100*normalized_layouts/total:.1f}%)")
    logger.info(f"POV-specific graphs: {has_pov_graphs} ({100*has_pov_graphs/total:.1f}%)")
    logger.info(f"Empty rooms: {empty} ({100*empty/total:.1f}%)")
    logger.info(f"\nPOV types:")
    for pt, count in pov_types.most_common():
        logger.info(f"  {pt}: {count}")
    logger.info(f"\nTop room types:")
    for rt, count in room_types.most_common(10):
        logger.info(f"  {rt}: {count}")
    logger.info(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate POV-normalized manifest CSV")
    parser.add_argument("--dataset-root", required=True, help="Root directory of dataset")
    parser.add_argument("--output-tex", default=None, help="Output path for tex manifest")
    parser.add_argument("--output-seg", default=None, help="Output path for seg manifest")
    parser.add_argument("--tex-only", action="store_true", help="Only generate tex manifest")
    parser.add_argument("--seg-only", action="store_true", help="Only generate seg manifest")
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        logger.error(f"Dataset root not found: {dataset_root}")
        return
    
    manifests_dir = dataset_root / "manifests"
    output_tex = Path(args.output_tex) if args.output_tex else manifests_dir / "manifest_tex_pov_normalized.csv"
    output_seg = Path(args.output_seg) if args.output_seg else manifests_dir / "manifest_seg_pov_normalized.csv"
    
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
        
        rows = collect_manifest_data_pov_normalized(dataset_root, variant)
        rows = compute_weights(rows)
        print_statistics(rows, variant)
        write_manifest(rows, output_path)
    
    total_elapsed = time.time() - total_start
    logger.info(f"\n{'='*60}")
    logger.info(f"DONE! Total time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
