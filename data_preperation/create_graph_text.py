#!/usr/bin/env python3
"""
create_graph_text.py

Reads graphs.csv manifest, converts graphs to text using graph2text,
and saves each as a .txt file.
"""

import argparse
import csv
import json
from pathlib import Path
from tqdm import tqdm

from utils.text_utils import graph2text
from utils.semantic_utils import Taxonomy


def process_graphs(manifest_path: str, taxonomy_path: str):
    """
    Process all graphs in manifest: convert to text and save as .txt files.
    
    Args:
        manifest_path: Path to graphs.csv
        taxonomy_path: Path to taxonomy.json
    """
    # Load taxonomy
    print(f"Loading taxonomy from {taxonomy_path}")
    taxonomy = Taxonomy(taxonomy_path)
    
    # Read manifest
    print(f"Reading manifest: {manifest_path}")
    manifest_path = Path(manifest_path)
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Found {len(rows)} graphs to process")
    
    # Process each graph
    skipped = 0
    
    for row in tqdm(rows, desc="Creating text files"):
        graph_path = row['graph_path']
        
        try:
            text = graph2text(graph_path, taxonomy)
            
            if not text:
                print(f"Warning: Empty text for {graph_path}")
                skipped += 1
                continue
            
            # Save as .txt file
            txt_path = Path(graph_path).with_suffix('.txt')
            txt_path.write_text(text, encoding='utf-8')
            
        except Exception as e:
            print(f"Error processing {graph_path}: {e}")
            skipped += 1
    
    print(f"\n✓ Processed {len(rows) - skipped}/{len(rows)} graphs successfully")
    print(f"✓ Skipped {skipped} graphs")


def main():
    parser = argparse.ArgumentParser(
        description="Convert graph JSON files to text files"
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to graphs.csv manifest"
    )
    parser.add_argument(
        "--taxonomy",
        required=True,
        help="Path to taxonomy.json"
    )
    
    args = parser.parse_args()
    
    process_graphs(
        manifest_path=args.manifest,
        taxonomy_path=args.taxonomy
    )


if __name__ == "__main__":
    main()