#!/usr/bin/env python3
"""
collect_all.py

Creates a manifest for conditional diffusion training:
- Each row represents one training sample
- Links POV → Graph Text → Layout
- Filters out empty rooms based on pov_manifest
- Includes BOTH seg and tex POVs with pov_type column
"""

import argparse
import json
import csv
from pathlib import Path
from tqdm import tqdm


def load_empty_rooms(pov_manifest_path: str):
    """
    Load set of empty rooms from POV manifest.
    Returns set of (scene_id, room_id) tuples that are empty.
    """
    empty_rooms = set()
    
    with open(pov_manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['is_empty'] == '1':
                empty_rooms.add((row['scene_id'], row['room_id']))
    
    return empty_rooms


def create_manifest(data_root: str, pov_manifest: str, output_manifest: str):
    """
    Create training manifest with columns:
    - sample_id: unique identifier
    - scene_id: scene UUID
    - room_id: room ID (empty for scene-level)
    - sample_type: 'room' or 'scene'
    - pov_type: 'seg' or 'tex' (empty for scene-level)
    - pov_image: path to POV image (empty for scene-level)
    - pov_embedding: path to POV embedding (empty for scene-level)
    - graph_text: path to graph text file
    - graph_embedding: path to graph embedding (if exists)
    - layout_image: path to layout image
    - layout_embedding: path to layout embedding (if exists)
    """
    
    data_root = Path(data_root)
    
    # Load empty rooms to filter out
    print(f"Loading empty rooms from {pov_manifest}")
    empty_rooms = load_empty_rooms(pov_manifest)
    print(f"Found {len(empty_rooms)} empty room(s) to skip")
    
    samples = []
    skipped_empty = 0
    
    # Find all scene directories
    scene_dirs = [d for d in data_root.iterdir() if d.is_dir()]
    
    print(f"Found {len(scene_dirs)} scenes")
    
    for scene_dir in tqdm(scene_dirs, desc="Processing scenes"):
        scene_id = scene_dir.name
        
        # ================================================================
        # SCENE-LEVEL SAMPLES (no POVs)
        # ================================================================
        scene_graph_json = scene_dir / f"{scene_id}_scene_graph.json"
        scene_graph_txt = scene_dir / f"{scene_id}_scene_graph.txt"
        scene_graph_emb = scene_dir / f"{scene_id}_scene_graph.pt"
        scene_layout_img = scene_dir / "layouts" / f"{scene_id}_scene_layout.png"
        scene_layout_emb = scene_dir / "layouts" / f"{scene_id}_layout_emb.pt"
        
        if scene_graph_txt.exists() and scene_layout_img.exists():
            samples.append({
                'sample_id': f"{scene_id}_scene",
                'scene_id': scene_id,
                'room_id': '',
                'sample_type': 'scene',
                'pov_type': '',
                'pov_image': '',
                'pov_embedding': '',
                'graph_text': str(scene_graph_txt),
                'graph_embedding': str(scene_graph_emb) if scene_graph_emb.exists() else '',
                'layout_image': str(scene_layout_img),
                'layout_embedding': str(scene_layout_emb) if scene_layout_emb.exists() else '',
            })
        
        # ================================================================
        # ROOM-LEVEL SAMPLES (with POVs - BOTH seg and tex)
        # ================================================================
        rooms_dir = scene_dir / "rooms"
        if not rooms_dir.exists():
            continue
        
        for room_dir in rooms_dir.iterdir():
            if not room_dir.is_dir():
                continue
            
            room_id = room_dir.name
            
            # Skip empty rooms
            if (scene_id, room_id) in empty_rooms:
                skipped_empty += 1
                continue
            
            # Room graph and layout
            room_graph_json = room_dir / "layouts" / f"{scene_id}_{room_id}_graph.json"
            room_graph_txt = room_dir / "layouts" / f"{scene_id}_{room_id}_graph.txt"
            room_layout_img = room_dir / "layouts" / f"{scene_id}_{room_id}_room_seg_layout.png"
            
            if not room_graph_txt.exists() or not room_layout_img.exists():
                continue
            
            # Process BOTH POV types: seg and tex
            for pov_type in ['seg', 'tex']:
                pov_dir = room_dir / "povs" / pov_type
                
                if not pov_dir.exists():
                    continue
                
                # Get all POV variants (v01, v02, etc.) for this type
                pov_images = sorted(pov_dir.glob(f"{scene_id}_{room_id}_v*_pov_{pov_type}.png"))
                
                for pov_img in pov_images:
                    # Extract viewpoint ID (v01, v02, etc.)
                    viewpoint = pov_img.stem.split('_')[-3]  # e.g., 'v01'
                    
                    pov_emb = pov_img.with_suffix('.pt')
                    
                    sample_id = f"{scene_id}_{room_id}_{pov_type}_{viewpoint}"
                    
                    samples.append({
                        'sample_id': sample_id,
                        'scene_id': scene_id,
                        'room_id': room_id,
                        'sample_type': 'room',
                        'pov_type': pov_type,
                        'pov_image': str(pov_img),
                        'pov_embedding': str(pov_emb) if pov_emb.exists() else '',
                        'graph_text': str(room_graph_txt),
                        'graph_embedding': '',
                        'layout_image': str(room_layout_img),
                        'layout_embedding': '',
                    })
    
    # Write manifest
    output_path = Path(output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        'sample_id',
        'scene_id', 
        'room_id',
        'sample_type',
        'pov_type',
        'pov_image',
        'pov_embedding',
        'graph_text',
        'graph_embedding',
        'layout_image',
        'layout_embedding'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)
    
    # Print statistics
    room_samples_seg = sum(1 for s in samples if s['sample_type'] == 'room' and s['pov_type'] == 'seg')
    room_samples_tex = sum(1 for s in samples if s['sample_type'] == 'room' and s['pov_type'] == 'tex')
    scene_samples = sum(1 for s in samples if s['sample_type'] == 'scene')
    
    print(f"\n✓ Created manifest with {len(samples)} total samples")
    print(f"  - Room-level samples (seg POVs): {room_samples_seg}")
    print(f"  - Room-level samples (tex POVs): {room_samples_tex}")
    print(f"  - Scene-level samples: {scene_samples}")
    print(f"  - Skipped empty rooms: {skipped_empty}")
    print(f"✓ Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create training manifest for conditional diffusion pipeline"
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root directory containing scene folders"
    )
    parser.add_argument(
        "--pov-manifest",
        required=True,
        help="Path to POV manifest CSV with is_empty column"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for training_manifest.csv"
    )
    
    args = parser.parse_args()
    
    create_manifest(
        data_root=args.data_root,
        pov_manifest=args.pov_manifest,
        output_manifest=args.output
    )


if __name__ == "__main__":
    main()