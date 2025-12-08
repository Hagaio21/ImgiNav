#!/usr/bin/env python3
"""
Generate multi-POVs for validation scenes only.

Steps:
1. Load your manifest
2. Get validation split (same seed as training)
3. Extract unique scenes from validation
4. Filter pov_info to those scenes
5. Generate multi-POVs only for those

Usage:
    python generate_val_multi_povs.py \
        --manifest /path/to/manifest.csv \
        --pov-info /path/to/pov_info.json \
        --dataset-root /path/to/dataset \
        --train-split 0.8 \
        --seed 42
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def get_validation_scenes(manifest_path: Path, train_split: float = 0.8, seed: int = 42) -> set:
    """Get scene IDs that are in the validation split."""
    
    df = pd.read_csv(manifest_path)
    df.columns = df.columns.str.strip()
    
    # Apply same split logic as training
    np.random.seed(seed)
    n = len(df)
    indices = np.random.permutation(n)
    train_end = int(n * train_split)
    val_indices = indices[train_end:]
    
    # Get validation rows
    val_df = df.iloc[val_indices]
    
    # Extract unique scene IDs
    if 'scene_id' in val_df.columns:
        val_scenes = set(val_df['scene_id'].unique())
    else:
        # Try to extract from room_id (format: sceneID_roomID)
        val_scenes = set()
        for room_id in val_df['room_id'].unique():
            parts = str(room_id).rsplit('_', 1)
            if len(parts) >= 1:
                val_scenes.add(parts[0])
    
    logger.info(f"Total samples: {n}")
    logger.info(f"Validation samples: {len(val_indices)}")
    logger.info(f"Validation scenes: {len(val_scenes)}")
    
    return val_scenes


def filter_pov_info_to_scenes(pov_info_path: Path, scenes: set) -> list:
    """Filter POV info to only include specified scenes."""
    
    with open(pov_info_path, 'r') as f:
        all_povs = json.load(f)
    
    filtered = [p for p in all_povs if p.get('scene_id') in scenes]
    
    logger.info(f"Total POVs: {len(all_povs)}")
    logger.info(f"Filtered POVs (val scenes): {len(filtered)}")
    
    return filtered


def check_povs_per_room(povs: list) -> dict:
    """Check distribution of POVs per room."""
    
    rooms = defaultdict(list)
    for p in povs:
        key = f"{p.get('scene_id')}_{p.get('room_id')}"
        rooms[key].append(p)
    
    # Count distribution
    counts = defaultdict(int)
    for room_povs in rooms.values():
        counts[len(room_povs)] += 1
    
    return dict(counts), rooms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Path to manifest.csv")
    parser.add_argument("--pov-info", type=Path, required=True,
                        help="Path to pov_info.json")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path for filtered pov_info")
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true",
                        help="Only check POV distribution, don't save")
    args = parser.parse_args()
    
    # Get validation scenes
    val_scenes = get_validation_scenes(args.manifest, args.train_split, args.seed)
    
    # Filter POVs
    val_povs = filter_pov_info_to_scenes(args.pov_info, val_scenes)
    
    # Check distribution
    dist, rooms = check_povs_per_room(val_povs)
    
    print("\n=== POVs per Room Distribution (Validation) ===")
    for n_povs in sorted(dist.keys()):
        print(f"  {n_povs} POV(s): {dist[n_povs]} rooms")
    
    total_rooms = len(rooms)
    rooms_with_multi = sum(1 for r in rooms.values() if len(r) >= 2)
    print(f"\nRooms with 2+ POVs: {rooms_with_multi}/{total_rooms} ({100*rooms_with_multi/total_rooms:.1f}%)")
    
    # Show some examples
    print("\n=== Example Rooms with Multiple POVs ===")
    multi_rooms = [(k, v) for k, v in rooms.items() if len(v) >= 2]
    for room_id, room_povs in multi_rooms[:5]:
        print(f"\n{room_id}:")
        for p in room_povs:
            pov_id = p.get('pov_id', 'unknown')
            pov_type = p.get('pov_type', 'unknown')
            print(f"  - {pov_id} ({pov_type})")
    
    if not args.check_only:
        # Save filtered POV info
        if args.output is None:
            args.output = args.pov_info.parent / "pov_info_val.json"
        
        with open(args.output, 'w') as f:
            json.dump(val_povs, f, indent=2)
        
        logger.info(f"\nSaved validation POVs to: {args.output}")
        
        # Also save scene list for multi-POV generation
        scene_list_path = args.output.parent / "val_scenes.txt"
        with open(scene_list_path, 'w') as f:
            for scene in sorted(val_scenes):
                f.write(f"{scene}\n")
        logger.info(f"Saved validation scene list to: {scene_list_path}")


if __name__ == "__main__":
    main()
