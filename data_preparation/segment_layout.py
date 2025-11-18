#!/usr/bin/env python3
"""
Example script demonstrating how to use the LayoutSegmentor.

This script segments a layout image by finding the closest color for each pixel
and assigning it to a category or super-category.
"""

import argparse
from pathlib import Path
from PIL import Image
import numpy as np

from common.taxonomy import Taxonomy
from data_preparation.utils.layout_analysis import LayoutSegmentor, segment_layout


def main():
    parser = argparse.ArgumentParser(
        description="Segment a layout image by finding closest colors to taxonomy categories"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input layout image"
    )
    parser.add_argument(
        "--taxonomy",
        type=str,
        default="config/taxonomy.json",
        help="Path to taxonomy.json file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["category", "super"],
        default="category",
        help="Segment by category or super-category (default: category)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save segmentation visualization (optional)"
    )
    parser.add_argument(
        "--output-seg-map",
        type=str,
        default=None,
        help="Path to save segmentation map as numpy array (optional)"
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Print statistics about the segmentation"
    )
    
    args = parser.parse_args()
    
    # Load taxonomy
    taxonomy = Taxonomy(args.taxonomy)
    
    # Create segmentor
    segmentor = LayoutSegmentor(taxonomy, mode=args.mode)
    
    # Load image
    image = Image.open(args.input).convert("RGB")
    print(f"[INFO] Loaded image: {image.size[0]}x{image.size[1]}")
    
    # Segment image
    seg_map = segmentor.segment(image)
    print(f"[INFO] Segmentation map shape: {seg_map.shape}")
    print(f"[INFO] Unique classes found: {len(np.unique(seg_map))}")
    
    # Print statistics if requested
    if args.show_stats:
        unique_ids, counts = np.unique(seg_map, return_counts=True)
        print("\n[STATS] Class distribution:")
        for class_id, count in zip(unique_ids, counts):
            class_name = taxonomy.id_to_name(int(class_id))
            percentage = (count / seg_map.size) * 100
            print(f"  {class_name} (ID {class_id}): {count} pixels ({percentage:.2f}%)")
    
    # Save visualization
    if args.output:
        vis_image = segmentor.segment_to_image(image)
        vis_image.save(args.output)
        print(f"[INFO] Saved visualization to: {args.output}")
    
    # Save segmentation map
    if args.output_seg_map:
        np.save(args.output_seg_map, seg_map)
        print(f"[INFO] Saved segmentation map to: {args.output_seg_map}")
    
    print("[INFO] Segmentation complete!")


if __name__ == "__main__":
    main()

