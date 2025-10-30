#!/usr/bin/env python3
"""
Test script to run the complete data processing pipeline on 5 scenes.

This script:
1. Reads configuration from test_config.yaml
2. Processes 5 scenes through stages 1-4 + graph building
3. Validates outputs between stages
4. Uses test_dataset as the output directory
"""

import argparse
import os
import subprocess
import sys
import yaml
from pathlib import Path

# Add parent directory to path to import project modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.utils import safe_mkdir, load_config_with_profile


def get_env_with_pythonpath():
    """Get environment variables with PYTHONPATH set to project root."""
    env = os.environ.copy()
    pythonpath = str(PROJECT_ROOT)
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = pythonpath + os.pathsep + env['PYTHONPATH']
    else:
        env['PYTHONPATH'] = pythonpath
    return env


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise RuntimeError(f"Failed to load config: {e}")


def resolve_path(path_str: str, base_dir: Path = None) -> Path:
    """Resolve path, handling relative and absolute paths."""
    if base_dir is None:
        base_dir = PROJECT_ROOT
    
    path = Path(path_str)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def validate_stage_outputs(stage_name: str, output_dir: Path, expected_files: list):
    """Validate that expected output files exist after a stage."""
    missing = []
    for pattern in expected_files:
        files = list(output_dir.rglob(pattern))
        if not files:
            missing.append(pattern)
    
    if missing:
        print(f"  ⚠ Warning: Missing expected outputs for {stage_name}:")
        for m in missing:
            print(f"    - {m}")
        return False
    
    print(f"  ✓ Stage {stage_name} outputs validated")
    return True


def run_stage1(config: dict) -> bool:
    """Run Stage 1: Build scenes from JSON → point clouds."""
    print("\n" + "="*70)
    print("STAGE 1: Building Scenes")
    print("="*70)
    
    stage1_config = config.get("stage1", {})
    output_dir = resolve_path(config["output_dir"])
    
    # Add scene source (check scenes_root or use glob)
    scenes_root = resolve_path(config.get("scenes_root"))
    scene_list_file = None
    if scenes_root.exists():
        # Find first N JSON files
        limit = stage1_config.get("limit", 5)
        scene_files = sorted(list(scenes_root.rglob("*.json")))[:limit]
        if scene_files:
            print(f"  Found {len(scene_files)} scene files (limit: {limit})")
            # Create a temporary list file
            scene_list_file = PROJECT_ROOT / "tests" / "temp_scene_list.txt"
            scene_list_file.parent.mkdir(exist_ok=True)
            scene_list_file.write_text("\n".join(str(f) for f in scene_files))
        else:
            print(f"  ✗ Error: No JSON scene files found in {scenes_root}")
            return False
    else:
        print(f"  ✗ Error: Scenes root directory not found: {scenes_root}")
        return False
    
    # Build command
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "data_preperation" / "stage1_build_scenes.py"),
        "--out_dir", str(output_dir),
        "--model_dir", str(resolve_path(config["model_dir"])),
        "--model_info", str(resolve_path(config["model_info"])),
        "--taxonomy", str(resolve_path(config["taxonomy"], PROJECT_ROOT)),
        "--total_points", str(stage1_config.get("total_points", 500000)),
        "--min_pts_per_mesh", str(stage1_config.get("min_pts_per_mesh", 100)),
        "--max_pts_per_mesh", str(stage1_config.get("max_pts_per_mesh", 0)),
        "--ppsm", str(stage1_config.get("ppsm", 0.0)),
        "--limit", str(stage1_config.get("limit", 5)),
    ]
    
    if stage1_config.get("save_parquet", True):
        cmd.append("--save_parquet")
    
    if stage1_config.get("per_scene_subdir", True):
        cmd.append("--per_scene_subdir")
    
    if scene_list_file:
        cmd.extend(["--scene_list", str(scene_list_file)])
    
    # Run command
    print(f"  Command: {' '.join(cmd[:10])}... (truncated)")
    try:
        env = get_env_with_pythonpath()
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True, capture_output=False)
        print(f"  ✓ Stage 1 completed successfully")
        
        # Validate outputs
        expected_patterns = ["*_sem_pointcloud.parquet", "*_scene_info.json"]
        validate_stage_outputs("1", output_dir, expected_patterns)
        
        # Cleanup temp scene list file
        if scene_list_file and scene_list_file.exists():
            scene_list_file.unlink()
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Stage 1 failed with exit code {e.returncode}")
        # Cleanup temp file even on failure
        if scene_list_file and scene_list_file.exists():
            scene_list_file.unlink()
        return False


def run_stage2(config: dict) -> bool:
    """Run Stage 2: Split scenes into rooms."""
    print("\n" + "="*70)
    print("STAGE 2: Splitting Scenes to Rooms")
    print("="*70)
    
    stage2_config = config.get("stage2", {})
    output_dir = resolve_path(config["output_dir"])
    
    # Build command
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "data_preperation" / "stage2_split2rooms.py"),
        "--in_dir", str(output_dir),
        "--glob", stage2_config.get("glob", "*_sem_pointcloud.parquet"),
        "--taxonomy", str(resolve_path(config["taxonomy"], PROJECT_ROOT)),
    ]
    
    if stage2_config.get("inplace", True):
        cmd.append("--inplace")
    
    if stage2_config.get("compute_frames", True):
        cmd.append("--compute-frames")
        map_band = stage2_config.get("map_band", [0.05, 0.50])
        cmd.extend(["--map-band", str(map_band[0]), str(map_band[1])])
    
    # Run command
    print(f"  Command: {' '.join(cmd)}")
    try:
        env = get_env_with_pythonpath()
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True, capture_output=False)
        print(f"  ✓ Stage 2 completed successfully")
        
        # Validate outputs
        expected_patterns = ["rooms/*/*.parquet", "rooms/*/*_meta.json"]
        validate_stage_outputs("2", output_dir, expected_patterns)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Stage 2 failed with exit code {e.returncode}")
        return False


def run_stage3(config: dict) -> bool:
    """Run Stage 3: Create room and scene layouts."""
    print("\n" + "="*70)
    print("STAGE 3: Creating Layouts")
    print("="*70)
    
    stage3_config = config.get("stage3", {})
    output_dir = resolve_path(config["output_dir"])
    
    # Build command
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "data_preperation" / "stage3_create_room_scenes_layouts.py"),
        "--in_root", str(output_dir),
        "--taxonomy", str(resolve_path(config["taxonomy"], PROJECT_ROOT)),
        "--mode", stage3_config.get("mode", "both"),
        "--res", str(stage3_config.get("resolution", 512)),
        "--hmin", str(stage3_config.get("height_min", 0.1)),
        "--hmax", str(stage3_config.get("height_max", 1.8)),
        "--point-size", str(stage3_config.get("point_size", 10)),
        "--color-mode", stage3_config.get("color_mode", "super"),
    ]
    
    # Run command
    print(f"  Command: {' '.join(cmd)}")
    try:
        env = get_env_with_pythonpath()
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True, capture_output=False)
        print(f"  ✓ Stage 3 completed successfully")
        
        # Validate outputs
        mode = stage3_config.get("mode", "both")
        expected_patterns = []
        if mode in ("room", "both"):
            expected_patterns.append("rooms/*/layouts/*_room_seg_layout.png")
        if mode in ("scene", "both"):
            expected_patterns.append("layouts/*_scene_layout.png")
        
        validate_stage_outputs("3", output_dir, expected_patterns)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Stage 3 failed with exit code {e.returncode}")
        return False


def run_stage4(config: dict) -> bool:
    """Run Stage 4: Create POV renderings."""
    print("\n" + "="*70)
    print("STAGE 4: Creating POVs")
    print("="*70)
    
    stage4_config = config.get("stage4", {})
    output_dir = resolve_path(config["output_dir"])
    
    # Build command
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "data_preperation" / "stage4_create_room_povs.py"),
        "--dataset-root", str(output_dir),
        "--taxonomy", str(resolve_path(config["taxonomy"], PROJECT_ROOT)),
        "--width", str(stage4_config.get("width", 512)),
        "--height", str(stage4_config.get("height", 512)),
        "--fov-deg", str(stage4_config.get("fov_deg", 70.0)),
        "--eye-height", str(stage4_config.get("eye_height", 1.6)),
        "--point-size", str(stage4_config.get("point_size", 3.0)),
        "--bg", *[str(x) for x in stage4_config.get("bg_rgb", [0, 0, 0])],
        "--num-views", str(stage4_config.get("num_views", 6)),
        "--seed", str(stage4_config.get("seed", 1)),
    ]
    
    if stage4_config.get("hpc", False):
        cmd.append("--hpc")
    
    # Run command
    print(f"  Command: {' '.join(cmd)}")
    try:
        env = get_env_with_pythonpath()
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True, capture_output=False)
        print(f"  ✓ Stage 4 completed successfully")
        
        # Validate outputs
        expected_patterns = [
            "rooms/*/povs/tex/*_pov_tex.png",
            "rooms/*/povs/seg/*_pov_seg.png",
            "rooms/*/povs/*_pov_meta.json",
        ]
        validate_stage_outputs("4", output_dir, expected_patterns)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Stage 4 failed with exit code {e.returncode}")
        return False


def run_stage5(config: dict) -> bool:
    """Run Stage 5: Build graphs (room and scene)."""
    print("\n" + "="*70)
    print("STAGE 5: Building Graphs")
    print("="*70)
    
    stage5_config = config.get("stage5", {})
    output_dir = resolve_path(config["output_dir"])
    
    success = True
    
    # Run room graphs
    if "room" in stage5_config:
        print("\n  Building room graphs...")
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "data_preperation" / "build_graphs.py"),
            "--type", "room",
            "--in_dir", str(output_dir),
            "--taxonomy", str(resolve_path(config["taxonomy"], PROJECT_ROOT)),
        ]
        
        print(f"    Command: {' '.join(cmd)}")
        try:
            env = get_env_with_pythonpath()
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True, capture_output=False)
            print("    ✓ Room graphs completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"    ✗ Room graphs failed with exit code {e.returncode}")
            success = False
    
    # Run scene graphs
    if "scene" in stage5_config:
        print("\n  Building scene graphs...")
        scene_config = stage5_config["scene"]
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "data_preperation" / "build_graphs.py"),
            "--type", "scene",
            "--in_dir", str(output_dir),
            "--taxonomy", str(resolve_path(config["taxonomy"], PROJECT_ROOT)),
            "--adjacency_thresh", str(scene_config.get("adjacency_thresh", 0.3)),
        ]
        
        print(f"    Command: {' '.join(cmd)}")
        try:
            env = get_env_with_pythonpath()
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True, capture_output=False)
            print("    ✓ Scene graphs completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"    ✗ Scene graphs failed with exit code {e.returncode}")
            success = False
    
    # Validate outputs
    expected_patterns = ["*_graph.json"]
    validate_stage_outputs("5", output_dir, expected_patterns)
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Test data processing pipeline on 5 scenes"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="tests/test_config.yaml",
        help="Path to configuration YAML file (default: tests/test_config.yaml)"
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run only a specific stage (1-5)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = resolve_path(args.config)
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Validate required paths
    required_paths = ["scenes_root", "model_dir", "model_info", "taxonomy", "output_dir"]
    for key in required_paths:
        if key not in config:
            print(f"✗ Error: Missing required config key: {key}")
            sys.exit(1)
    
    # Resolve and validate input paths
    print("\nValidating input paths...")
    scenes_root = resolve_path(config["scenes_root"])
    model_dir = resolve_path(config["model_dir"])
    model_info = resolve_path(config["model_info"])
    taxonomy = resolve_path(config["taxonomy"], PROJECT_ROOT)
    
    if not scenes_root.exists():
        print(f"✗ Error: Scenes root not found: {scenes_root}")
        sys.exit(1)
    if not model_dir.exists():
        print(f"✗ Error: Model directory not found: {model_dir}")
        sys.exit(1)
    if not model_info.exists():
        print(f"✗ Error: Model info file not found: {model_info}")
        sys.exit(1)
    if not taxonomy.exists():
        print(f"✗ Error: Taxonomy file not found: {taxonomy}")
        sys.exit(1)
    
    print("  ✓ All input paths validated")
    
    # Create output directory
    output_dir = resolve_path(config["output_dir"])
    safe_mkdir(output_dir)
    print(f"\nOutput directory: {output_dir}")
    
    # Run stages
    stages = {
        1: run_stage1,
        2: run_stage2,
        3: run_stage3,
        4: run_stage4,
        5: run_stage5,
    }
    
    if args.stage:
        # Run single stage
        print(f"\n{'='*70}")
        print(f"Running only Stage {args.stage}")
        print(f"{'='*70}")
        success = stages[args.stage](config)
        sys.exit(0 if success else 1)
    else:
        # Run all stages sequentially
        print(f"\n{'='*70}")
        print("Running Complete Pipeline (Stages 1-5)")
        print(f"{'='*70}")
        
        results = {}
        for stage_num, stage_func in stages.items():
            results[stage_num] = stage_func(config)
            if not results[stage_num]:
                print(f"\n✗ Pipeline failed at Stage {stage_num}")
                print("Stopping execution.")
                sys.exit(1)
        
        # Summary
        print(f"\n{'='*70}")
        print("PIPELINE SUMMARY")
        print(f"{'='*70}")
        for stage_num, success in results.items():
            status = "✓" if success else "✗"
            print(f"  Stage {stage_num}: {status}")
        
        all_success = all(results.values())
        if all_success:
            print(f"\n✓ Pipeline completed successfully!")
            print(f"  Output directory: {output_dir}")
        else:
            print(f"\n✗ Pipeline completed with errors")
            sys.exit(1)


if __name__ == "__main__":
    main()

