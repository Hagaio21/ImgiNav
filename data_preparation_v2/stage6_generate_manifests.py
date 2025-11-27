#!/usr/bin/env python3
"""
Stage 6: Manifest Generation

Creates global index files (manifests) that reference all scenes, rooms,
metadata, geometry, renders, and graphs.

Output:
- manifests/scenes.json
- manifests/rooms.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

# No dependencies on old pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def safe_mkdir(path: Path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def write_json(data: dict, path: Path):
    """Write JSON file."""
    safe_mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def generate_scenes_manifest(
    dataset_root: Path,
    output_path: Path
) -> Dict:
    """
    Generate scenes manifest.
    
    Args:
        dataset_root: Root directory of dataset
        output_path: Output path for scenes manifest
    
    Returns:
        Manifest dictionary
    """
    scenes_manifest = {
        "version": "2.0",
        "dataset_root": str(dataset_root),
        "scenes": []
    }
    
    # Find all scene metadata files
    metadata_dir = dataset_root / "metadata" / "scenes"
    if not metadata_dir.exists():
        logger.warning(f"Metadata directory not found: {metadata_dir}")
        return scenes_manifest
    
    scene_meta_files = list(metadata_dir.glob("*.json"))
    logger.info(f"Found {len(scene_meta_files)} scene metadata files")
    
    for scene_meta_path in tqdm(scene_meta_files, desc="Processing scenes"):
        scene_id = scene_meta_path.stem
        
        # Build scene entry
        scene_entry = {
            "scene_id": scene_id,
            "metadata": {
                "path": str(scene_meta_path.relative_to(dataset_root))
            },
            "geometry": {
                "tex": {
                    "path": f"geometry/tex/{scene_id}_tex.glb"
                },
                "seg": {
                    "path": f"geometry/seg/{scene_id}_seg.glb"
                }
            },
            "layouts": {
                "tex": {
                    "path": f"layouts/tex/{scene_id}_tex_layout.png"
                },
                "seg": {
                    "path": f"layouts/seg/{scene_id}_seg_layout.png"
                }
            },
            "graph": {
                "path": f"graphs/jsons/{scene_id}_scene_graph.json"
            },
            "rooms": []
        }
        
        # Load scene metadata to get room list
        try:
            with open(scene_meta_path, "r") as f:
                scene_meta = json.load(f)
            
            # Add scene-level info
            scene_entry["bbox"] = scene_meta.get("bbox", {})
            scene_entry["doors_count"] = len(scene_meta.get("doors", []))
            scene_entry["windows_count"] = len(scene_meta.get("windows", []))
            scene_entry["furniture_count"] = scene_meta.get("furniture_count", 0)
            
            # Add rooms
            for room in scene_meta.get("rooms", []):
                room_id = room.get("room_id")
                if room_id:
                    scene_entry["rooms"].append({
                        "room_id": room_id,
                        "room_type": room.get("room_type", "UnknownRoom")
                    })
        
        except Exception as e:
            logger.warning(f"Failed to load metadata for {scene_id}: {e}")
        
        scenes_manifest["scenes"].append(scene_entry)
    
    return scenes_manifest


def generate_rooms_manifest(
    dataset_root: Path,
    output_path: Path
) -> Dict:
    """
    Generate rooms manifest.
    
    Args:
        dataset_root: Root directory of dataset
        output_path: Output path for rooms manifest
    
    Returns:
        Manifest dictionary
    """
    rooms_manifest = {
        "version": "2.0",
        "dataset_root": str(dataset_root),
        "rooms": []
    }
    
    # Find all room metadata files
    metadata_dir = dataset_root / "metadata" / "rooms"
    if not metadata_dir.exists():
        logger.warning(f"Room metadata directory not found: {metadata_dir}")
        return rooms_manifest
    
    room_meta_files = list(metadata_dir.glob("*.json"))
    logger.info(f"Found {len(room_meta_files)} room metadata files")
    
    for room_meta_path in tqdm(room_meta_files, desc="Processing rooms"):
        # Extract scene_id and room_id from filename
        # Format: <scene_id>_<room_id>.json
        stem = room_meta_path.stem
        parts = stem.split("_", 1)  # Split on first underscore only
        if len(parts) < 2:
            logger.warning(f"Unexpected filename format: {room_meta_path.name}")
            continue
        
        scene_id = parts[0]
        room_id = parts[1]
        
        # Build room entry
        room_entry = {
            "scene_id": scene_id,
            "room_id": room_id,
            "metadata": {
                "path": str(room_meta_path.relative_to(dataset_root))
            },
            "layouts": {
                "tex": {
                    "path": f"layouts/tex/{scene_id}_{room_id}_tex_layout.png"
                },
                "seg": {
                    "path": f"layouts/seg/{scene_id}_{room_id}_seg_layout.png"
                }
            },
            "povs": {
                "tex": {
                    "path": f"povs/tex/{scene_id}_{room_id}_tex_pov.png"
                },
                "seg": {
                    "path": f"povs/seg/{scene_id}_{room_id}_seg_pov.png"
                }
            },
            "graph": {
                "path": f"graphs/jsons/{scene_id}_{room_id}_room_graph.json"
            }
        }
        
        # Load room metadata to get additional info
        try:
            with open(room_meta_path, "r") as f:
                room_meta = json.load(f)
            
            room_entry["room_type"] = room_meta.get("room_type", "UnknownRoom")
            room_entry["bbox"] = room_meta.get("bbox", {})
            room_entry["centroid"] = room_meta.get("centroid", [0, 0, 0])
            room_entry["doors_count"] = len(room_meta.get("doors", []))
            room_entry["windows_count"] = len(room_meta.get("windows", []))
            room_entry["furniture_count"] = room_meta.get("furniture_count", 0)
            room_entry["has_pov_camera"] = room_meta.get("pov_camera") is not None
        
        except Exception as e:
            logger.warning(f"Failed to load metadata for {scene_id}_{room_id}: {e}")
        
        rooms_manifest["rooms"].append(room_entry)
    
    return rooms_manifest


def main():
    parser = argparse.ArgumentParser(
        description="Stage 6: Generate manifests"
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root directory of dataset (e.g., dataset/)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for manifests (e.g., dataset/manifests)"
    )
    parser.add_argument(
        "--scenes-only",
        action="store_true",
        help="Only generate scenes manifest"
    )
    parser.add_argument(
        "--rooms-only",
        action="store_true",
        help="Only generate rooms manifest"
    )
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    
    if args.scenes_only and args.rooms_only:
        logger.error("Cannot specify both --scenes-only and --rooms-only")
        return
    
    safe_mkdir(output_dir)
    
    # Generate scenes manifest
    if not args.rooms_only:
        logger.info("Generating scenes manifest...")
        scenes_manifest = generate_scenes_manifest(dataset_root, output_dir / "scenes.json")
        write_json(scenes_manifest, output_dir / "scenes.json")
        logger.info(f"Generated scenes manifest with {len(scenes_manifest['scenes'])} scenes")
    
    # Generate rooms manifest
    if not args.scenes_only:
        logger.info("Generating rooms manifest...")
        rooms_manifest = generate_rooms_manifest(dataset_root, output_dir / "rooms.json")
        write_json(rooms_manifest, output_dir / "rooms.json")
        logger.info(f"Generated rooms manifest with {len(rooms_manifest['rooms'])} rooms")


if __name__ == "__main__":
    main()

