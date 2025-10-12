#!/usr/bin/env python3
"""
create_graph_manifest.py

Creates a graph manifest CSV from layout manifest, adding graph_path column.
Graphs are expected to be in the same directory as layout images.
"""

import argparse
import csv
from pathlib import Path


def create_graph_manifest(layout_manifest_path: Path, output_path: Path):
    """
    Read layout manifest and create graph manifest with graph_path column.
    
    Args:
        layout_manifest_path: Path to input layout manifest CSV
        output_path: Path to output graph manifest CSV
    """
    rows_written = 0
    rows_with_graphs = 0
    
    with open(layout_manifest_path, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        
        # Verify required columns exist
        required_cols = ['scene_id', 'type', 'room_id', 'layout_path', 'is_empty']
        if not all(col in reader.fieldnames for col in required_cols):
            raise ValueError(f"Layout manifest must contain columns: {required_cols}")
        
        # Open output file
        with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
            fieldnames = ['scene_id', 'type', 'room_id', 'layout_path', 'graph_path', 'is_empty']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                scene_id = row['scene_id']
                room_id = row['room_id']
                layout_path = Path(row['layout_path'])
                
                # Construct graph path in same directory as layout
                if room_id == 'scene':
                    # Scene-level graph
                    graph_filename = f"{scene_id}_scene_graph.json"
                else:
                    # Room-level graph
                    graph_filename = f"{scene_id}_{room_id}_graph.json"
                
                graph_path = layout_path.parent / graph_filename
                
                # Add graph_path to row
                output_row = {
                    'scene_id': scene_id,
                    'type': row['type'],
                    'room_id': room_id,
                    'layout_path': str(layout_path),
                    'graph_path': str(graph_path),
                    'is_empty': row['is_empty']
                }
                
                writer.writerow(output_row)
                rows_written += 1
                
                # Check if graph file exists
                if graph_path.exists():
                    rows_with_graphs += 1
    
    print(f"✓ Created graph manifest: {output_path}")
    print(f"  Total rows: {rows_written}")
    print(f"  Graphs found: {rows_with_graphs}")
    print(f"  Graphs missing: {rows_written - rows_with_graphs}")


def main():
    parser = argparse.ArgumentParser(
        description="Create graph manifest CSV from layout manifest"
    )
    parser.add_argument(
        "--layout_manifest",
        required=True,
        help="Path to input layout manifest CSV"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output graph manifest CSV"
    )
    
    args = parser.parse_args()
    
    layout_manifest = Path(args.layout_manifest)
    output_path = Path(args.output)
    
    if not layout_manifest.exists():
        print(f"[error] Layout manifest not found: {layout_manifest}")
        return
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_graph_manifest(layout_manifest, output_path)


if __name__ == "__main__":
    main()