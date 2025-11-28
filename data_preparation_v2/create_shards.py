#!/usr/bin/env python3
"""
Create shard files from a list of scene IDs.

Usage:
    python create_shards.py --scene-list all_scenes.txt --output-dir shards/ --num-shards 10
    python create_shards.py --scenes-dir /path/to/3D-FRONT/ --output-dir shards/ --num-shards 10
"""

import argparse
import os
from pathlib import Path
from typing import List


def load_scene_list(scene_list_path: Path) -> List[str]:
    """Load scene IDs from a text file (one per line)."""
    scenes = []
    with open(scene_list_path, "r", encoding="utf-8") as f:
        for line in f:
            scene_id = line.strip()
            if scene_id and not scene_id.startswith("#"):
                scenes.append(scene_id)
    return scenes


def discover_scenes(scenes_dir: Path) -> List[str]:
    """Discover scene IDs from a directory of JSON files."""
    scenes = []
    for json_file in scenes_dir.rglob("*.json"):
        # Skip non-scene files (like model_info.json, etc.)
        stem = json_file.stem
        # 3D-FRONT scenes have UUID format
        if len(stem) >= 32 and "-" in stem:
            scenes.append(stem)
    return sorted(scenes)


def create_shards(scene_ids: List[str], output_dir: Path, num_shards: int):
    """Split scene IDs into shards and save them."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate scenes per shard
    total_scenes = len(scene_ids)
    scenes_per_shard = (total_scenes + num_shards - 1) // num_shards
    
    print(f"Creating {num_shards} shards from {total_scenes} scenes")
    print(f"Approximately {scenes_per_shard} scenes per shard")
    
    for shard_idx in range(1, num_shards + 1):
        start_idx = (shard_idx - 1) * scenes_per_shard
        end_idx = min(shard_idx * scenes_per_shard, total_scenes)
        
        shard_scenes = scene_ids[start_idx:end_idx]
        shard_file = output_dir / f"shard_{shard_idx}.txt"
        
        with open(shard_file, "w", encoding="utf-8") as f:
            for scene_id in shard_scenes:
                f.write(f"{scene_id}\n")
        
        print(f"  shard_{shard_idx}.txt: {len(shard_scenes)} scenes")
    
    # Also create all_scenes.txt for reference
    all_scenes_file = output_dir / "all_scenes.txt"
    with open(all_scenes_file, "w", encoding="utf-8") as f:
        for scene_id in scene_ids:
            f.write(f"{scene_id}\n")
    
    print(f"\nCreated {num_shards} shard files in {output_dir}")
    print(f"All scenes also saved to {all_scenes_file}")


def main():
    parser = argparse.ArgumentParser(description="Create shard files for parallel processing")
    parser.add_argument("--scene-list", type=str, help="Path to file with scene IDs (one per line)")
    parser.add_argument("--scenes-dir", type=str, help="Directory to discover scenes from")
    parser.add_argument("--output-dir", required=True, type=str, help="Output directory for shard files")
    parser.add_argument("--num-shards", type=int, default=10, help="Number of shards to create")
    args = parser.parse_args()
    
    if args.scene_list:
        scene_ids = load_scene_list(Path(args.scene_list))
        print(f"Loaded {len(scene_ids)} scenes from {args.scene_list}")
    elif args.scenes_dir:
        scene_ids = discover_scenes(Path(args.scenes_dir))
        print(f"Discovered {len(scene_ids)} scenes in {args.scenes_dir}")
    else:
        parser.error("Must provide either --scene-list or --scenes-dir")
    
    if not scene_ids:
        print("ERROR: No scenes found")
        return 1
    
    create_shards(scene_ids, Path(args.output_dir), args.num_shards)
    return 0


if __name__ == "__main__":
    exit(main())
