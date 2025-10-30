#!/usr/bin/env python3
"""
Test script to run Stage 3 (layout generation) only on existing test_dataset.

This script reads configuration from test_config.yaml and generates layouts
for all rooms and scenes in the test_dataset directory.
"""

import argparse
import os
import subprocess
import sys
import yaml
from pathlib import Path
import numpy as np
from PIL import Image

# Add parent directory to path to import project modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.utils import safe_mkdir


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


def count_unique_colors(image_path: Path) -> int:
    """Count the number of unique colors in an image."""
    try:
        img = Image.open(image_path)
        img_array = np.array(img.convert("RGB"))
        
        # Reshape to (pixels, 3) and get unique rows
        pixels = img_array.reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)
        
        return len(unique_colors)
    except Exception as e:
        print(f"  Error counting colors in {image_path}: {e}", flush=True)
        return 0


def validate_layout_colors(output_dir: Path, min_colors: int = 3) -> bool:
    """Validate that layout images have sufficient color diversity."""
    print("\n" + "="*70)
    print("VALIDATING LAYOUT COLORS")
    print("="*70)
    
    scene_layouts = list(output_dir.rglob("layouts/*_scene_layout.png"))
    room_layouts = list(output_dir.rglob("rooms/*/layouts/*_room_seg_layout.png"))
    
    all_valid = True
    
    # Check scene layouts
    if scene_layouts:
        print(f"\nChecking {len(scene_layouts)} scene layouts:")
        for layout_path in scene_layouts:
            num_colors = count_unique_colors(layout_path)
            status = "✓" if num_colors >= min_colors else "✗"
            print(f"  {status} {layout_path.name}: {num_colors} unique colors", flush=True)
            if num_colors < min_colors:
                all_valid = False
                print(f"    ⚠ WARNING: Only {num_colors} colors found (expected at least {min_colors})", flush=True)
    
    # Check room layouts (sample first 10)
    if room_layouts:
        print(f"\nChecking {min(len(room_layouts), 10)} room layouts (sample):")
        for layout_path in room_layouts[:10]:
            num_colors = count_unique_colors(layout_path)
            status = "✓" if num_colors >= min_colors else "✗"
            print(f"  {status} {layout_path.name}: {num_colors} unique colors", flush=True)
            if num_colors < min_colors:
                all_valid = False
                print(f"    ⚠ WARNING: Only {num_colors} colors found (expected at least {min_colors})", flush=True)
    
    return all_valid


def run_stage3(config: dict, test_dataset_dir: Path = None) -> bool:
    """Run Stage 3: Create room and scene layouts."""
    print("\n" + "="*70)
    print("STAGE 3: Creating Layouts")
    print("="*70)
    
    stage3_config = config.get("stage3", {})
    
    # Use test_dataset if not provided
    if test_dataset_dir is None:
        output_dir = resolve_path(config["output_dir"])
    else:
        output_dir = resolve_path(test_dataset_dir, PROJECT_ROOT)
    
    if not output_dir.exists():
        print(f"  ✗ Error: Test dataset directory not found: {output_dir}")
        return False
    
    print(f"  Input directory: {output_dir}")
    
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
        
        # Validate outputs exist
        mode = stage3_config.get("mode", "both")
        expected_patterns = []
        if mode in ("room", "both"):
            expected_patterns.append("rooms/*/layouts/*_room_seg_layout.png")
        if mode in ("scene", "both"):
            expected_patterns.append("layouts/*_scene_layout.png")
        
        missing = []
        for pattern in expected_patterns:
            files = list(output_dir.rglob(pattern))
            if not files:
                missing.append(pattern)
            else:
                print(f"    ✓ Found {len(files)} files matching {pattern}")
        
        if missing:
            print(f"  ⚠ Warning: Missing expected outputs:")
            for m in missing:
                print(f"    - {m}")
            return False
        
        print(f"  ✓ Stage 3 outputs validated")
        
        # Validate color diversity
        color_valid = validate_layout_colors(output_dir, min_colors=3)
        if not color_valid:
            print(f"  ✗ Color validation failed - layouts have too few colors")
            return False
        
        print(f"  ✓ Color validation passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Stage 3 failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test Stage 3 (layout generation) on existing test_dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="tests/test_config.yaml",
        help="Path to configuration YAML file (default: tests/test_config.yaml)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to test_dataset directory (default: uses output_dir from config)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = resolve_path(args.config)
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Resolve dataset directory
    if args.dataset:
        dataset_dir = Path(args.dataset)
    else:
        dataset_dir = resolve_path(config.get("output_dir", "test_dataset"))
    
    print(f"\nDataset directory: {dataset_dir}")
    
    if not dataset_dir.exists():
        print(f"✗ Error: Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    # Run Stage 3
    success = run_stage3(config, str(dataset_dir))
    
    if success:
        print(f"\n✓ Stage 3 test completed successfully!")
        print(f"  Layouts generated in: {dataset_dir}")
    else:
        print(f"\n✗ Stage 3 test failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

