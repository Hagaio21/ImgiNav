#!/usr/bin/env python3
"""
generate_palette.py (backward compatibility wrapper)
----------------------------------------------------

DEPRECATED: This script is now consolidated into utils.semantic_utils.
Use: from utils.semantic_utils import generate_palette_for_labels

This wrapper is kept for backward compatibility with existing scripts.
"""

import argparse
import sys
from pathlib import Path

# Import from consolidated location
sys.path.insert(0, str(Path(__file__).parent))
from utils.semantic_utils import generate_palette_for_labels


def main(json_path: Path):
    """Generate palette for semantic_maps.json file."""
    generate_palette_for_labels(json_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate color palette for semantic_maps.json")
    parser.add_argument("json_path", type=Path, help="Path to semantic_maps.json")
    args = parser.parse_args()
    main(args.json_path)
