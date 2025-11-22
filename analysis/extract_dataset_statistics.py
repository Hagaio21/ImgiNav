#!/usr/bin/env python3
"""
Extract baseline statistics from the ground truth dataset.

This script analyzes all layouts in the dataset to extract:
- Object count distributions (per class, per room type, overall)
- Co-occurrence patterns between object classes
- Statistics about object counts per scene
- Reference distributions for comparison with generated images
"""

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from common.taxonomy import Taxonomy
from analysis.evaluation_metrics import LayoutEvaluator
from common.utils import safe_mkdir, write_json


def load_manifest(manifest_path: Path) -> List[Dict]:
    """Load layout manifest CSV."""
    rows = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def extract_dataset_statistics(
    manifest_path: Path,
    taxonomy_path: Path,
    output_dir: Path,
    mode: str = "super",
    cooccurrence_radius: float = 0.15,
    filter_empty: bool = True,
    class_pairs: Optional[List[Tuple[str, str]]] = None,
    max_samples: Optional[int] = None,
    room_type_filter: Optional[List[str]] = None
) -> Dict:
    """
    Extract statistics from the dataset.
    
    Args:
        manifest_path: Path to layout manifest CSV
        taxonomy_path: Path to taxonomy.json
        output_dir: Output directory for statistics
        mode: "super" or "category" - evaluation mode
        cooccurrence_radius: Radius for co-occurrence analysis
        filter_empty: If True, skip empty layouts
        class_pairs: Optional list of specific class pairs to analyze
        max_samples: Optional maximum number of samples to process
        room_type_filter: Optional list of room types to include
    
    Returns:
        Dictionary with extracted statistics
    """
    safe_mkdir(output_dir)
    
    # Load taxonomy and initialize evaluator
    print("Loading taxonomy...")
    taxonomy = Taxonomy(taxonomy_path)
    evaluator = LayoutEvaluator(
        taxonomy=taxonomy,
        mode=mode,
        cooccurrence_radius=cooccurrence_radius
    )
    
    # Load manifest
    print(f"Loading manifest from {manifest_path}...")
    manifest_rows = load_manifest(manifest_path)
    print(f"Found {len(manifest_rows)} entries in manifest")
    
    # Filter rows
    if filter_empty:
        manifest_rows = [r for r in manifest_rows if int(r.get("is_empty", 0)) == 0]
        print(f"After filtering empty layouts: {len(manifest_rows)} entries")
    
    if room_type_filter:
        manifest_rows = [r for r in manifest_rows if r.get("type", "") in room_type_filter]
        print(f"After filtering by room type: {len(manifest_rows)} entries")
    
    if max_samples:
        manifest_rows = manifest_rows[:max_samples]
        print(f"Limiting to {max_samples} samples")
    
    # Statistics to collect
    all_object_counts = defaultdict(int)  # Overall counts
    counts_by_room_type = defaultdict(lambda: defaultdict(int))  # Per room type
    counts_per_image = []  # List of count dicts for each image
    all_cooccurrences = defaultdict(list)  # Co-occurrence rates per pair
    all_bbox_stats = defaultdict(list)  # Bbox statistics per class
    all_densities = []  # Density metrics per image
    
    # Process each layout
    print(f"\nProcessing {len(manifest_rows)} layouts...")
    failed = 0
    
    for row in tqdm(manifest_rows, desc="Extracting statistics"):
        layout_path = Path(row.get("layout_path", ""))
        if not layout_path.exists():
            failed += 1
            continue
        
        room_type = row.get("type", "unknown")
        
        try:
            # Count objects
            counts = evaluator.count_objects(layout_path)
            counts_per_image.append(counts)
            
            # Aggregate counts
            for class_name, count in counts.items():
                all_object_counts[class_name] += count
                counts_by_room_type[room_type][class_name] += count
            
            # Compute co-occurrences
            if class_pairs is None:
                # Compute all pairs for this image
                cooc_results = evaluator.compute_all_cooccurrences(layout_path)
            else:
                # Compute only specified pairs
                cooc_results = evaluator.compute_all_cooccurrences(layout_path, class_pairs)
            
            # Aggregate co-occurrence rates
            for pair_key, result in cooc_results.items():
                if result["num_class_a"] > 0:  # Only if class A exists
                    all_cooccurrences[pair_key].append(result["cooccurrence_rate"])
            
            # Analyze bbox statistics
            bbox_stats = evaluator.analyze_bbox_statistics(layout_path)
            for class_name, stats in bbox_stats.items():
                all_bbox_stats[class_name].append(stats)
            
            # Analyze density
            density = evaluator.analyze_bbox_density(layout_path)
            all_densities.append(density)
        
        except Exception as e:
            print(f"\nWarning: Failed to process {layout_path}: {e}")
            failed += 1
            continue
    
    if failed > 0:
        print(f"\nWarning: Failed to process {failed} layouts")
    
    print(f"\nSuccessfully processed {len(counts_per_image)} layouts")
    
    # Compute statistics
    print("\nComputing statistics...")
    
    # Overall object count distribution
    total_objects = sum(all_object_counts.values())
    object_distribution = {
        class_name: {
            "count": count,
            "percentage": (count / total_objects * 100) if total_objects > 0 else 0.0,
            "frequency": sum(1 for c in counts_per_image if class_name in c) / len(counts_per_image) if counts_per_image else 0.0
        }
        for class_name, count in sorted(all_object_counts.items(), key=lambda x: -x[1])
    }
    
    # Per-room-type distributions
    room_type_distributions = {}
    for room_type, counts in counts_by_room_type.items():
        total = sum(counts.values())
        room_type_distributions[room_type] = {
            class_name: {
                "count": count,
                "percentage": (count / total * 100) if total > 0 else 0.0
            }
            for class_name, count in sorted(counts.items(), key=lambda x: -x[1])
        }
    
    # Object count statistics per image
    num_objects_per_image = [sum(c.values()) for c in counts_per_image]
    unique_classes_per_image = [len(c) for c in counts_per_image]
    
    count_stats = {
        "total_images": len(counts_per_image),
        "objects_per_image": {
            "mean": float(np.mean(num_objects_per_image)) if num_objects_per_image else 0.0,
            "std": float(np.std(num_objects_per_image)) if num_objects_per_image else 0.0,
            "min": int(np.min(num_objects_per_image)) if num_objects_per_image else 0,
            "max": int(np.max(num_objects_per_image)) if num_objects_per_image else 0,
            "median": float(np.median(num_objects_per_image)) if num_objects_per_image else 0.0
        },
        "unique_classes_per_image": {
            "mean": float(np.mean(unique_classes_per_image)) if unique_classes_per_image else 0.0,
            "std": float(np.std(unique_classes_per_image)) if unique_classes_per_image else 0.0,
            "min": int(np.min(unique_classes_per_image)) if unique_classes_per_image else 0,
            "max": int(np.max(unique_classes_per_image)) if unique_classes_per_image else 0,
            "median": float(np.median(unique_classes_per_image)) if unique_classes_per_image else 0.0
        }
    }
    
    # Co-occurrence statistics
    cooccurrence_stats = {}
    for pair_key, rates in all_cooccurrences.items():
        if len(rates) > 0:
            cooccurrence_stats[pair_key] = {
                "mean_rate": float(np.mean(rates)),
                "std_rate": float(np.std(rates)),
                "min_rate": float(np.min(rates)),
                "max_rate": float(np.max(rates)),
                "median_rate": float(np.median(rates)),
                "num_samples": len(rates)
            }
    
    # Aggregate bbox statistics
    bbox_statistics = {}
    for class_name, stats_list in all_bbox_stats.items():
        if len(stats_list) == 0:
            continue
        
        # Aggregate across all images
        widths = []
        heights = []
        areas = []
        aspect_ratios = []
        
        for stats in stats_list:
            if "width" in stats and "mean" in stats["width"]:
                widths.extend([stats["width"]["mean"]] * stats.get("count", 1))
            if "height" in stats and "mean" in stats["height"]:
                heights.extend([stats["height"]["mean"]] * stats.get("count", 1))
            if "area" in stats and "mean" in stats["area"]:
                areas.extend([stats["area"]["mean"]] * stats.get("count", 1))
            if "aspect_ratio" in stats and "mean" in stats["aspect_ratio"]:
                aspect_ratios.extend([stats["aspect_ratio"]["mean"]] * stats.get("count", 1))
        
        if len(widths) > 0:
            bbox_statistics[class_name] = {
                "width": {
                    "mean": float(np.mean(widths)),
                    "std": float(np.std(widths)),
                    "min": float(np.min(widths)),
                    "max": float(np.max(widths))
                },
                "height": {
                    "mean": float(np.mean(heights)),
                    "std": float(np.std(heights)),
                    "min": float(np.min(heights)),
                    "max": float(np.max(heights))
                },
                "area": {
                    "mean": float(np.mean(areas)),
                    "std": float(np.std(areas)),
                    "min": float(np.min(areas)),
                    "max": float(np.max(areas))
                },
                "aspect_ratio": {
                    "mean": float(np.mean(aspect_ratios)),
                    "std": float(np.std(aspect_ratios)),
                    "min": float(np.min(aspect_ratios)),
                    "max": float(np.max(aspect_ratios))
                },
                "num_samples": len(stats_list)
            }
    
    # Aggregate density statistics
    if all_densities:
        overall_densities = [d["overall_density"] for d in all_densities]
        density_stats = {
            "overall_density": {
                "mean": float(np.mean(overall_densities)),
                "std": float(np.std(overall_densities)),
                "min": float(np.min(overall_densities)),
                "max": float(np.max(overall_densities)),
                "median": float(np.median(overall_densities))
            }
        }
        
        # Density by class
        density_by_class = defaultdict(list)
        for d in all_densities:
            for class_name, density in d.get("density_by_class", {}).items():
                density_by_class[class_name].append(density)
        
        density_stats["by_class"] = {
            class_name: {
                "mean": float(np.mean(densities)),
                "std": float(np.std(densities)),
                "min": float(np.min(densities)),
                "max": float(np.max(densities))
            }
            for class_name, densities in density_by_class.items()
        }
    else:
        density_stats = {}
    
    # Compile results
    results = {
        "dataset_info": {
            "manifest_path": str(manifest_path),
            "taxonomy_path": str(taxonomy_path),
            "mode": mode,
            "cooccurrence_radius": cooccurrence_radius,
            "total_samples": len(counts_per_image),
            "failed_samples": failed
        },
        "object_distribution": object_distribution,
        "object_distribution_by_room_type": room_type_distributions,
        "count_statistics": count_stats,
        "cooccurrence_statistics": cooccurrence_stats,
        "bbox_statistics": bbox_statistics,
        "density_statistics": density_stats
    }
    
    # Save results
    output_json = output_dir / "dataset_statistics.json"
    write_json(results, output_json)
    print(f"\nStatistics saved to {output_json}")
    
    # Print summary
    print("\n" + "="*60)
    print("Dataset Statistics Summary")
    print("="*60)
    print(f"Total samples analyzed: {len(counts_per_image)}")
    print(f"Total objects: {total_objects}")
    print(f"\nTop 10 most common object classes:")
    for i, (class_name, stats) in enumerate(list(object_distribution.items())[:10], 1):
        print(f"  {i:2d}. {class_name:25s}: {stats['count']:6d} objects ({stats['percentage']:5.1f}%) "
              f"[present in {stats['frequency']*100:5.1f}% of images]")
    
    print(f"\nObject count per image:")
    print(f"  Mean: {count_stats['objects_per_image']['mean']:.1f} ± {count_stats['objects_per_image']['std']:.1f}")
    print(f"  Range: {count_stats['objects_per_image']['min']} - {count_stats['objects_per_image']['max']}")
    print(f"  Median: {count_stats['objects_per_image']['median']:.1f}")
    
    print(f"\nUnique classes per image:")
    print(f"  Mean: {count_stats['unique_classes_per_image']['mean']:.1f} ± {count_stats['unique_classes_per_image']['std']:.1f}")
    print(f"  Range: {count_stats['unique_classes_per_image']['min']} - {count_stats['unique_classes_per_image']['max']}")
    
    if cooccurrence_stats:
        print(f"\nTop 10 co-occurrence patterns:")
        sorted_cooc = sorted(cooccurrence_stats.items(), 
                           key=lambda x: x[1]["mean_rate"], 
                           reverse=True)[:10]
        for i, (pair_key, stats) in enumerate(sorted_cooc, 1):
            class_a, class_b = pair_key.split("__")
            print(f"  {i:2d}. {class_a:20s} -> {class_b:20s}: "
                  f"{stats['mean_rate']:.2%} ± {stats['std_rate']:.2%} "
                  f"(n={stats['num_samples']})")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract baseline statistics from ground truth dataset"
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to layout manifest CSV"
    )
    parser.add_argument(
        "--taxonomy",
        default="config/taxonomy.json",
        type=Path,
        help="Path to taxonomy.json file"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Output directory for statistics"
    )
    parser.add_argument(
        "--mode",
        default="super",
        choices=["super", "category"],
        help="Evaluation mode: super or category"
    )
    parser.add_argument(
        "--cooccurrence_radius",
        type=float,
        default=0.15,
        help="Radius for co-occurrence (as fraction of image size, default 0.15)"
    )
    parser.add_argument(
        "--include_empty",
        action="store_true",
        help="Include empty layouts in analysis"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Maximum number of samples to process (for testing)"
    )
    parser.add_argument(
        "--room_types",
        nargs="+",
        help="Filter by room types (e.g., Bedroom LivingRoom)"
    )
    parser.add_argument(
        "--class_pairs",
        nargs="+",
        help="Specific class pairs to analyze (format: 'ClassA,ClassB ClassC,ClassD')"
    )
    
    args = parser.parse_args()
    
    # Parse class pairs if provided
    class_pairs = None
    if args.class_pairs:
        class_pairs = []
        for pair_str in args.class_pairs:
            parts = pair_str.split(",")
            if len(parts) == 2:
                class_pairs.append((parts[0].strip(), parts[1].strip()))
            else:
                print(f"Warning: Invalid class pair format: {pair_str}")
    
    extract_dataset_statistics(
        manifest_path=args.manifest,
        taxonomy_path=args.taxonomy,
        output_dir=args.output_dir,
        mode=args.mode,
        cooccurrence_radius=args.cooccurrence_radius,
        filter_empty=not args.include_empty,
        class_pairs=class_pairs,
        max_samples=args.max_samples,
        room_type_filter=args.room_types
    )


if __name__ == "__main__":
    main()

