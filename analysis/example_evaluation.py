#!/usr/bin/env python3
"""
Example script demonstrating how to use the evaluation metrics module.

This script shows how to:
1. Evaluate single image pairs
2. Evaluate batches of images
3. Analyze specific co-occurrence patterns
4. Compare distributions
"""

from pathlib import Path
from analysis.evaluation_metrics import LayoutEvaluator, evaluate_layouts
from common.taxonomy import Taxonomy
from PIL import Image


def example_single_evaluation():
    """Example: Evaluate a single generated image against ground truth."""
    print("="*60)
    print("Example 1: Single Image Evaluation")
    print("="*60)
    
    # Initialize evaluator
    taxonomy = Taxonomy("config/taxonomy.json")
    evaluator = LayoutEvaluator(taxonomy, mode="super", cooccurrence_radius=0.15)
    
    # Example paths (replace with your actual paths)
    gen_path = "path/to/generated.png"
    gt_path = "path/to/ground_truth.png"
    
    # Check if files exist before evaluating
    if not Path(gen_path).exists() or not Path(gt_path).exists():
        print(f"Note: Example files not found. Replace paths with actual image files.")
        return
    
    # Evaluate
    result = evaluator.evaluate_single(gen_path, gt_path)
    
    # Print results
    print("\nObject Counts:")
    print("  Generated:", result["object_counts"]["generated"])
    print("  Ground Truth:", result["object_counts"]["ground_truth"])
    
    print("\nDistribution Metrics:")
    for key, value in result["distribution_metrics"].items():
        print(f"  {key}: {value:.4f}")
    
    print("\nCo-occurrence Examples:")
    # Show a few co-occurrence comparisons
    for key in list(result["cooccurrence_metrics"]["comparison"].keys())[:3]:
        comp = result["cooccurrence_metrics"]["comparison"][key]
        print(f"  {key}:")
        print(f"    Generated rate: {comp['generated_rate']:.2%}")
        print(f"    Ground truth rate: {comp['ground_truth_rate']:.2%}")
        print(f"    Difference: {comp['difference']:.2%}")


def example_batch_evaluation():
    """Example: Evaluate a batch of images."""
    print("\n" + "="*60)
    print("Example 2: Batch Evaluation")
    print("="*60)
    
    # Example paths (replace with your actual paths)
    generated_paths = [
        "path/to/generated1.png",
        "path/to/generated2.png",
        "path/to/generated3.png"
    ]
    ground_truth_paths = [
        "path/to/gt1.png",
        "path/to/gt2.png",
        "path/to/gt3.png"
    ]
    
    # Check if files exist
    if not all(Path(p).exists() for p in generated_paths + ground_truth_paths):
        print(f"Note: Example files not found. Replace paths with actual image files.")
        return
    
    # Evaluate batch
    results = evaluate_layouts(
        generated_paths=generated_paths,
        ground_truth_paths=ground_truth_paths,
        taxonomy_path="config/taxonomy.json",
        mode="super",
        cooccurrence_radius=0.15,
        output_path="evaluation_results.json"
    )
    
    # Print summary
    print(f"\nEvaluated {results['num_samples']} image pairs")
    print("\nAverage Distribution Metrics:")
    for key, value in results["average_distribution_metrics"].items():
        if not key.endswith("_std"):
            print(f"  {key}: {value:.4f}")
    
    print("\nAggregate Object Counts:")
    print("  Generated:")
    for class_name, count in sorted(results["aggregate_object_counts"]["generated"].items()):
        print(f"    {class_name}: {count}")
    print("  Ground Truth:")
    for class_name, count in sorted(results["aggregate_object_counts"]["ground_truth"].items()):
        print(f"    {class_name}: {count}")


def example_cooccurrence_analysis():
    """Example: Analyze specific co-occurrence patterns."""
    print("\n" + "="*60)
    print("Example 3: Co-occurrence Analysis")
    print("="*60)
    
    taxonomy = Taxonomy("config/taxonomy.json")
    evaluator = LayoutEvaluator(taxonomy, mode="super", cooccurrence_radius=0.15)
    
    # Example path
    image_path = "path/to/layout.png"
    
    if not Path(image_path).exists():
        print(f"Note: Example file not found. Replace path with actual image file.")
        return
    
    # Analyze specific class pairs
    class_pairs = [
        ("Bed", "Cabinet/Shelf/Desk"),
        ("Bed", "Chair"),
        ("Sofa", "Table"),
        ("Sofa", "Cabinet/Shelf/Desk")
    ]
    
    print("\nCo-occurrence Analysis:")
    for class_a, class_b in class_pairs:
        result = evaluator.compute_cooccurrence(image_path, class_a, class_b)
        print(f"\n{class_a} -> {class_b}:")
        print(f"  Co-occurrence rate: {result['cooccurrence_rate']:.2%}")
        print(f"  Average distance: {result['avg_distance']:.1f} pixels")
        print(f"  Number of {class_a} objects: {result['num_class_a']}")
        print(f"  Number of {class_b} objects: {result['num_class_b']}")


def example_object_counting():
    """Example: Count objects in an image."""
    print("\n" + "="*60)
    print("Example 4: Object Counting")
    print("="*60)
    
    taxonomy = Taxonomy("config/taxonomy.json")
    evaluator = LayoutEvaluator(taxonomy, mode="super")
    
    # Example path
    image_path = "path/to/layout.png"
    
    if not Path(image_path).exists():
        print(f"Note: Example file not found. Replace path with actual image file.")
        return
    
    # Count objects
    counts = evaluator.count_objects(image_path)
    
    print("\nObjects in image:")
    total_objects = sum(counts.values())
    for class_name, count in sorted(counts.items(), key=lambda x: -x[1]):
        percentage = (count / total_objects * 100) if total_objects > 0 else 0
        print(f"  {class_name:25s}: {count:3d} objects ({percentage:5.1f}%)")
    print(f"\n  Total: {total_objects} objects")


if __name__ == "__main__":
    print("\nEvaluation Metrics Examples")
    print("="*60)
    print("\nNote: These examples use placeholder paths.")
    print("Replace the paths with actual image files to run the examples.\n")
    
    # Run examples
    example_single_evaluation()
    example_batch_evaluation()
    example_cooccurrence_analysis()
    example_object_counting()
    
    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)

