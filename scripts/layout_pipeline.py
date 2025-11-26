#!/usr/bin/env python3
"""Unified pipeline: clean layouts, create bboxes/centroids, and compare."""
import argparse
import sys
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Unified layout processing pipeline")
    parser.add_argument("--image1", type=Path, required=True, help="First image")
    parser.add_argument("--image2", type=Path, required=True, help="Second image")
    parser.add_argument("--taxonomy", type=Path, required=True, help="Taxonomy JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--min-pixels", type=int, default=20, help="Minimum pixels per object (default: 20)")
    parser.add_argument("--alpha", type=float, default=0.6, help="Transparency for overlay (default: 0.6)")
    parser.add_argument("--centroid-radius", type=int, default=8, help="Centroid radius (default: 8)")
    args = parser.parse_args()
    
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stem1 = args.image1.stem
    stem2 = args.image2.stem
    
    print(f"\n{'='*60}")
    print(f"Layout Processing Pipeline")
    print(f"{'='*60}")
    print(f"Image1: {args.image1.name}")
    print(f"Image2: {args.image2.name}")
    print(f"Output: {output_dir}")
    print(f"Min pixels: {args.min_pixels}")
    print(f"{'='*60}\n")
    
    # Step 1: Clean images
    print("[1/4] Cleaning images...")
    cmd1 = [
        sys.executable, "scripts/clean_layout_colors.py",
        "--image", str(args.image1), str(args.image2),
        "--taxonomy", str(args.taxonomy),
        "--output-dir", str(output_dir)
    ]
    subprocess.run(cmd1, check=True)
    print("  ✓ Cleaned images saved\n")
    
    # Step 2: Create bboxes and centroids
    print("[2/4] Creating bboxes and centroids...")
    cleaned1 = output_dir / f"{stem1}_cleaned.png"
    cleaned2 = output_dir / f"{stem2}_cleaned.png"
    
    cmd2 = [
        sys.executable, "scripts/create_bboxes_from_cleaned.py",
        "--image", str(cleaned1), str(cleaned2),
        "--taxonomy", str(args.taxonomy),
        "--output-dir", str(output_dir),
        "--min-pixels", str(args.min_pixels),
        "--centroid-radius", str(args.centroid_radius)
    ]
    subprocess.run(cmd2, check=True)
    print("  ✓ Bbox and centroid images saved\n")
    
    # Step 3: Create comparison visualizations
    print("[3/4] Creating comparison visualizations...")
    bbox1 = output_dir / f"{stem1}_bboxes.png"
    centroid1 = output_dir / f"{stem1}_centroids.png"
    bbox2 = output_dir / f"{stem2}_bboxes.png"
    centroid2 = output_dir / f"{stem2}_centroids.png"
    
    cmd3 = [
        sys.executable, "scripts/visualize_comparison.py",
        "--image1-cleaned", str(cleaned1),
        "--image1-bbox", str(bbox1),
        "--image1-centroid", str(centroid1),
        "--image2-cleaned", str(cleaned2),
        "--image2-bbox", str(bbox2),
        "--image2-centroid", str(centroid2),
        "--taxonomy", str(args.taxonomy),
        "--output-dir", str(output_dir),
        "--min-pixels", str(args.min_pixels),
        "--alpha", str(args.alpha)
    ]
    subprocess.run(cmd3, check=True)
    print("  ✓ Comparison visualizations saved\n")
    
    # Step 4: Compare layouts (metrics)
    print("[4/4] Computing comparison metrics...")
    cmd4 = [
        sys.executable, "scripts/compare_layouts.py",
        "--image1", str(cleaned1),
        "--image2", str(cleaned2),
        "--taxonomy", str(args.taxonomy),
        "--output", str(output_dir / "comparison_metrics.json"),
        "--min-pixels", str(args.min_pixels)
    ]
    subprocess.run(cmd4, check=True)
    print("  ✓ Metrics saved\n")
    
    print(f"{'='*60}")
    print("Pipeline complete!")
    print(f"Results in: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

