#!/usr/bin/env python3
"""
Analysis script for Phase 1.2: Spatial Resolution Test
Loads metrics from all experiments and creates comparison visualizations.
"""

import sys
from pathlib import Path

# Import from phase1_1_analysis (reuse functions)
sys.path.insert(0, str(Path(__file__).parent))
from phase1_1_analysis import (
    load_metrics, create_loss_curves, create_final_metrics_comparison,
    create_convergence_analysis, create_summary_report,
    load_autoencoder_checkpoints, get_test_samples, create_visual_comparison
)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Phase 1.2 experiments")
    parser.add_argument(
        "--phase-dir",
        type=str,
        default="outputs/phase1_2_spatial_resolution",
        help="Path to phase directory containing metrics CSVs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for analysis plots (default: same as phase-dir/analysis)"
    )
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default="/work3/s233249/ImgiNav/experiments/phase1",
        help="Base directory containing individual experiment folders"
    )
    parser.add_argument(
        "--dataset-manifest",
        type=str,
        default="/work3/s233249/ImgiNav/datasets/layouts.csv",
        help="Path to dataset manifest for getting test samples"
    )
    parser.add_argument(
        "--skip-visual",
        action="store_true",
        help="Skip visual comparison (faster if only metrics needed)"
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    phase_dir = Path(args.phase_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = phase_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Phase 1.2: Spatial Resolution Test Analysis")
    print("=" * 80)
    print(f"Loading metrics from: {phase_dir}")
    print(f"Saving plots to: {output_dir}")
    print()
    
    # Load all metrics
    all_data = load_metrics(phase_dir)
    exp_names = list(all_data.keys())
    print(f"\nLoaded {len(exp_names)} experiments\n")
    
    # Create visualizations from metrics
    print("Creating metric visualizations...")
    create_loss_curves(all_data, output_dir)
    create_final_metrics_comparison(all_data, output_dir)
    create_convergence_analysis(all_data, output_dir)
    create_summary_report(all_data, output_dir)
    
    # Visual comparison with actual models
    if not args.skip_visual:
        print("\n" + "-" * 80)
        print("Loading autoencoder checkpoints for visual comparison...")
        print("-" * 80)
        checkpoints = load_autoencoder_checkpoints(exp_names, args.experiments_dir)
        
        if checkpoints:
            print("\nLoading test samples from dataset...")
            test_scenes, test_rooms = get_test_samples(args.dataset_manifest)
            
            if test_scenes and test_rooms:
                print("\nCreating visual comparisons...")
                create_visual_comparison(checkpoints, test_scenes, test_rooms, output_dir)
            else:
                print("  Could not find test samples, skipping visual comparison")
        else:
            print("  Could not load any checkpoints, skipping visual comparison")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

