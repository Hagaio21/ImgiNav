#!/usr/bin/env python3
"""
Compare generated images against baseline statistics extracted from the dataset.

This script:
1. Loads baseline statistics from extract_dataset_statistics.py
2. Evaluates generated images
3. Compares generated statistics against baseline
4. Reports differences and metrics
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm

from common.taxonomy import Taxonomy
from analysis.evaluation_metrics import LayoutEvaluator
from common.utils import safe_mkdir, write_json


def load_baseline_statistics(baseline_path: Path) -> Dict:
    """Load baseline statistics from JSON file."""
    with open(baseline_path, 'r') as f:
        return json.load(f)


def compare_to_baseline(
    generated_paths: List[Path],
    baseline_statistics_path: Path,
    taxonomy_path: Path,
    output_dir: Path,
    mode: str = "super",
    cooccurrence_radius: float = 0.15,
    class_pairs: Optional[List[tuple]] = None
) -> Dict:
    """
    Compare generated images against baseline statistics.
    
    Args:
        generated_paths: List of paths to generated layout images
        baseline_statistics_path: Path to baseline statistics JSON
        taxonomy_path: Path to taxonomy.json
        output_dir: Output directory for comparison results
        mode: "super" or "category" - evaluation mode
        cooccurrence_radius: Radius for co-occurrence analysis
        class_pairs: Optional list of class pairs to analyze
    
    Returns:
        Dictionary with comparison results
    """
    safe_mkdir(output_dir)
    
    # Load baseline statistics
    print(f"Loading baseline statistics from {baseline_statistics_path}...")
    baseline = load_baseline_statistics(baseline_statistics_path)
    
    # Verify mode matches
    if baseline["dataset_info"]["mode"] != mode:
        print(f"Warning: Baseline mode ({baseline['dataset_info']['mode']}) != current mode ({mode})")
    
    # Initialize evaluator
    print("Initializing evaluator...")
    taxonomy = Taxonomy(taxonomy_path)
    evaluator = LayoutEvaluator(
        taxonomy=taxonomy,
        mode=mode,
        cooccurrence_radius=cooccurrence_radius
    )
    
    # Process generated images
    print(f"\nProcessing {len(generated_paths)} generated images...")
    all_counts = {}
    all_cooccurrences = {}
    failed = 0
    
    for gen_path in tqdm(generated_paths, desc="Evaluating generated images"):
        if not gen_path.exists():
            print(f"Warning: File not found: {gen_path}")
            failed += 1
            continue
        
        try:
            # Count objects
            counts = evaluator.count_objects(gen_path)
            for class_name, count in counts.items():
                if class_name not in all_counts:
                    all_counts[class_name] = []
                all_counts[class_name].append(count)
            
            # Compute co-occurrences
            if class_pairs is None:
                cooc_results = evaluator.compute_all_cooccurrences(gen_path)
            else:
                cooc_results = evaluator.compute_all_cooccurrences(gen_path, class_pairs)
            
            for pair_key, result in cooc_results.items():
                if pair_key not in all_cooccurrences:
                    all_cooccurrences[pair_key] = []
                if result["num_class_a"] > 0:
                    all_cooccurrences[pair_key].append(result["cooccurrence_rate"])
        
        except Exception as e:
            print(f"\nWarning: Failed to process {gen_path}: {e}")
            failed += 1
            continue
    
    if failed > 0:
        print(f"\nWarning: Failed to process {failed} images")
    
    num_generated = len(generated_paths) - failed
    print(f"\nSuccessfully processed {num_generated} images")
    
    # Compute generated statistics
    print("\nComputing generated statistics...")
    
    # Aggregate object counts
    generated_object_distribution = {}
    for class_name, counts_list in all_counts.items():
        total_count = sum(counts_list)
        generated_object_distribution[class_name] = {
            "total_count": total_count,
            "mean_per_image": float(np.mean(counts_list)),
            "std_per_image": float(np.std(counts_list)),
            "frequency": len(counts_list) / num_generated if num_generated > 0 else 0.0
        }
    
    # Aggregate co-occurrence rates
    generated_cooccurrence_stats = {}
    for pair_key, rates in all_cooccurrences.items():
        if len(rates) > 0:
            generated_cooccurrence_stats[pair_key] = {
                "mean_rate": float(np.mean(rates)),
                "std_rate": float(np.std(rates)),
                "num_samples": len(rates)
            }
    
    # Compare with baseline
    print("\nComparing with baseline...")
    
    baseline_dist = baseline["object_distribution"]
    baseline_cooc = baseline.get("cooccurrence_statistics", {})
    
    # Object distribution comparison
    all_classes = set(generated_object_distribution.keys()) | set(baseline_dist.keys())
    distribution_comparison = {}
    
    for class_name in all_classes:
        gen_stats = generated_object_distribution.get(class_name, {})
        baseline_stats = baseline_dist.get(class_name, {})
        
        gen_total = gen_stats.get("total_count", 0)
        baseline_total = baseline_stats.get("count", 0)
        
        gen_freq = gen_stats.get("frequency", 0.0)
        baseline_freq = baseline_stats.get("frequency", 0.0)
        
        distribution_comparison[class_name] = {
            "generated": {
                "total_count": gen_total,
                "mean_per_image": gen_stats.get("mean_per_image", 0.0),
                "frequency": gen_freq
            },
            "baseline": {
                "total_count": baseline_total,
                "frequency": baseline_freq
            },
            "difference": {
                "count_diff": gen_total - baseline_total,
                "count_ratio": gen_total / baseline_total if baseline_total > 0 else float('inf'),
                "freq_diff": gen_freq - baseline_freq,
                "freq_ratio": gen_freq / baseline_freq if baseline_freq > 0 else float('inf')
            }
        }
    
    # Co-occurrence comparison
    cooccurrence_comparison = {}
    all_cooc_keys = set(generated_cooccurrence_stats.keys()) | set(baseline_cooc.keys())
    
    for pair_key in all_cooc_keys:
        gen_stats = generated_cooccurrence_stats.get(pair_key, {})
        baseline_stats = baseline_cooc.get(pair_key, {})
        
        gen_rate = gen_stats.get("mean_rate", 0.0)
        baseline_rate = baseline_stats.get("mean_rate", 0.0)
        
        cooccurrence_comparison[pair_key] = {
            "generated": {
                "mean_rate": gen_rate,
                "std_rate": gen_stats.get("std_rate", 0.0),
                "num_samples": gen_stats.get("num_samples", 0)
            },
            "baseline": {
                "mean_rate": baseline_rate,
                "std_rate": baseline_stats.get("std_rate", 0.0),
                "num_samples": baseline_stats.get("num_samples", 0)
            },
            "difference": {
                "rate_diff": gen_rate - baseline_rate,
                "rate_ratio": gen_rate / baseline_rate if baseline_rate > 0 else float('inf')
            }
        }
    
    # Compile results
    results = {
        "comparison_info": {
            "baseline_path": str(baseline_statistics_path),
            "num_generated_images": num_generated,
            "num_failed": failed,
            "mode": mode,
            "cooccurrence_radius": cooccurrence_radius
        },
        "generated_statistics": {
            "object_distribution": generated_object_distribution,
            "cooccurrence_statistics": generated_cooccurrence_stats
        },
        "comparison": {
            "object_distribution": distribution_comparison,
            "cooccurrence": cooccurrence_comparison
        }
    }
    
    # Save results
    output_json = output_dir / "comparison_results.json"
    write_json(results, output_json)
    print(f"\nComparison results saved to {output_json}")
    
    # Print summary
    print("\n" + "="*60)
    print("Comparison Summary")
    print("="*60)
    
    print(f"\nObject Distribution Comparison:")
    print(f"{'Class':<25} {'Generated':<15} {'Baseline':<15} {'Count Diff':<12} {'Freq Diff':<12}")
    print("-" * 80)
    
    sorted_classes = sorted(distribution_comparison.items(), 
                          key=lambda x: abs(x[1]["difference"]["count_diff"]), 
                          reverse=True)
    
    for class_name, comp in sorted_classes[:15]:
        gen_total = comp["generated"]["total_count"]
        baseline_total = comp["baseline"]["total_count"]
        count_diff = comp["difference"]["count_diff"]
        freq_diff = comp["difference"]["freq_diff"]
        
        print(f"{class_name:<25} {gen_total:>6} ({comp['generated']['frequency']:>5.1%}) "
              f"{baseline_total:>6} ({comp['baseline']['frequency']:>5.1%}) "
              f"{count_diff:>+10.0f} {freq_diff:>+10.2%}")
    
    if cooccurrence_comparison:
        print(f"\nCo-occurrence Comparison (top 10 by absolute difference):")
        print(f"{'Pair':<45} {'Generated':<12} {'Baseline':<12} {'Difference':<12}")
        print("-" * 85)
        
        sorted_cooc = sorted(cooccurrence_comparison.items(),
                           key=lambda x: abs(x[1]["difference"]["rate_diff"]),
                           reverse=True)
        
        for pair_key, comp in sorted_cooc[:10]:
            class_a, class_b = pair_key.split("__")
            pair_str = f"{class_a[:20]} -> {class_b[:20]}"
            gen_rate = comp["generated"]["mean_rate"]
            baseline_rate = comp["baseline"]["mean_rate"]
            rate_diff = comp["difference"]["rate_diff"]
            
            print(f"{pair_str:<45} {gen_rate:>10.2%} {baseline_rate:>10.2%} {rate_diff:>+10.2%}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare generated images against baseline statistics"
    )
    parser.add_argument(
        "--generated",
        required=True,
        nargs="+",
        type=Path,
        help="Paths to generated layout images"
    )
    parser.add_argument(
        "--baseline",
        required=True,
        type=Path,
        help="Path to baseline statistics JSON (from extract_dataset_statistics.py)"
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
        help="Output directory for comparison results"
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
        help="Radius for co-occurrence (as fraction of image size)"
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
    
    compare_to_baseline(
        generated_paths=args.generated,
        baseline_statistics_path=args.baseline,
        taxonomy_path=args.taxonomy,
        output_dir=args.output_dir,
        mode=args.mode,
        cooccurrence_radius=args.cooccurrence_radius,
        class_pairs=class_pairs
    )


if __name__ == "__main__":
    main()

