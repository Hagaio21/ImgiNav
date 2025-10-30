#!/usr/bin/env python3
"""
Test script to test collection and manifest creation.

This script:
1. Reads configuration from test_config.yaml
2. Tests collection functions (layouts, graphs, all)
3. Validates manifest outputs
"""

import argparse
import os
import subprocess
import sys
import yaml
import csv
from pathlib import Path
from typing import Dict, List

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


def validate_layouts_manifest(manifest_path: Path) -> tuple[bool, str]:
    """Validate layouts manifest structure and content."""
    if not manifest_path.exists():
        return False, f"Manifest file does not exist: {manifest_path}"
    
    required_cols = {"scene_id", "type", "room_id", "layout_path", "is_empty"}
    
    try:
        with open(manifest_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            cols = set(reader.fieldnames or [])
            
            if not required_cols.issubset(cols):
                missing = required_cols - cols
                return False, f"Missing required columns: {missing}"
            
            rows = list(reader)
            if len(rows) == 0:
                return False, "Manifest is empty"
            
            # Validate some rows
            scene_count = sum(1 for r in rows if r.get("type") == "scene")
            room_count = sum(1 for r in rows if r.get("type") == "room")
            
            if scene_count == 0 and room_count == 0:
                return False, "No scene or room layouts found in manifest"
            
            # Check that layout paths exist
            missing_files = []
            for row in rows[:10]:  # Sample check
                layout_path = row.get("layout_path", "")
                if layout_path and not Path(layout_path).exists():
                    missing_files.append(layout_path)
            
            if missing_files:
                return False, f"Some layout paths don't exist (sample): {missing_files[:3]}"
            
            return True, f"Valid manifest with {len(rows)} entries ({scene_count} scenes, {room_count} rooms)"
    
    except Exception as e:
        return False, f"Error reading manifest: {e}"


def validate_graphs_manifest(manifest_path: Path) -> tuple[bool, str]:
    """Validate graphs manifest structure and content."""
    if not manifest_path.exists():
        return False, f"Manifest file does not exist: {manifest_path}"
    
    required_cols = {"scene_id", "type", "room_id", "layout_path", "graph_path", "is_empty"}
    
    try:
        with open(manifest_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            cols = set(reader.fieldnames or [])
            
            if not required_cols.issubset(cols):
                missing = required_cols - cols
                return False, f"Missing required columns: {missing}"
            
            rows = list(reader)
            if len(rows) == 0:
                return False, "Manifest is empty"
            
            # Validate some rows
            scene_count = sum(1 for r in rows if r.get("type") == "scene")
            room_count = sum(1 for r in rows if r.get("type") == "room")
            
            if scene_count == 0 and room_count == 0:
                return False, "No scene or room graphs found in manifest"
            
            return True, f"Valid manifest with {len(rows)} entries ({scene_count} scenes, {room_count} rooms)"
    
    except Exception as e:
        return False, f"Error reading manifest: {e}"


def validate_all_manifest(manifest_path: Path) -> tuple[bool, str]:
    """Validate comprehensive 'all' manifest structure and content."""
    if not manifest_path.exists():
        return False, f"Manifest file does not exist: {manifest_path}"
    
    required_cols = {
        "sample_id", "sample_type", "scene_id", "room_id",
        "pov_type", "viewpoint", "pov_image", "pov_embedding",
        "graph_text", "graph_embedding", "layout_image", "layout_embedding",
        "is_empty"
    }
    
    try:
        with open(manifest_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            cols = set(reader.fieldnames or [])
            
            if not required_cols.issubset(cols):
                missing = required_cols - cols
                return False, f"Missing required columns: {missing}"
            
            rows = list(reader)
            if len(rows) == 0:
                return False, "Manifest is empty"
            
            # Validate sample types
            scene_samples = sum(1 for r in rows if r.get("sample_type") == "scene")
            room_samples = sum(1 for r in rows if r.get("sample_type") == "room")
            
            if scene_samples == 0 and room_samples == 0:
                return False, "No scene or room samples found in manifest"
            
            return True, f"Valid manifest with {len(rows)} entries ({scene_samples} scene samples, {room_samples} room samples)"
    
    except Exception as e:
        return False, f"Error reading manifest: {e}"


def run_collect_layouts(config: dict, dataset_root: Path, output_dir: Path) -> bool:
    """Run collection for layouts."""
    print("\n" + "="*70)
    print("COLLECTION: Layouts")
    print("="*70)
    
    output_manifest = output_dir / "layouts_manifest.csv"
    workers = config.get("collect", {}).get("workers", None)
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "data_preperation" / "collect.py"),
        "--type", "layouts",
        "--root", str(dataset_root),
        "--output", str(output_manifest),
    ]
    
    if workers is not None:
        cmd.extend(["--workers", str(workers)])
    
    print(f"  Command: {' '.join(cmd)}")
    try:
        env = get_env_with_pythonpath()
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True, capture_output=False)
        print(f"  ✓ Layouts collection completed successfully")
        
        # Validate manifest
        is_valid, msg = validate_layouts_manifest(output_manifest)
        if is_valid:
            print(f"  ✓ {msg}")
            return True
        else:
            print(f"  ✗ Validation failed: {msg}")
            return False
    
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Collection failed with exit code {e.returncode}")
        return False


def run_collect_graphs(config: dict, dataset_root: Path, output_dir: Path) -> bool:
    """Run collection for graphs."""
    print("\n" + "="*70)
    print("COLLECTION: Graphs")
    print("="*70)
    
    output_manifest = output_dir / "graphs_manifest.csv"
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "data_preperation" / "collect.py"),
        "--type", "graphs",
        "--root", str(dataset_root),
        "--output", str(output_manifest),
    ]
    
    print(f"  Command: {' '.join(cmd)}")
    try:
        env = get_env_with_pythonpath()
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True, capture_output=False)
        print(f"  ✓ Graphs collection completed successfully")
        
        # Validate manifest
        is_valid, msg = validate_graphs_manifest(output_manifest)
        if is_valid:
            print(f"  ✓ {msg}")
            return True
        else:
            print(f"  ✗ Validation failed: {msg}")
            return False
    
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Collection failed with exit code {e.returncode}")
        return False


def run_collect_all(config: dict, dataset_root: Path, output_dir: Path) -> bool:
    """Run comprehensive collection ('all')."""
    print("\n" + "="*70)
    print("COLLECTION: All (Comprehensive)")
    print("="*70)
    
    output_manifest = output_dir / "all_manifest.csv"
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "data_preperation" / "collect.py"),
        "--type", "all",
        "--data_root", str(dataset_root),
        "--output", str(output_manifest),
    ]
    
    print(f"  Command: {' '.join(cmd)}")
    
    try:
        env = get_env_with_pythonpath()
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True, capture_output=False)
        print(f"  ✓ Comprehensive collection completed successfully")
        
        # Validate manifest
        is_valid, msg = validate_all_manifest(output_manifest)
        if is_valid:
            print(f"  ✓ {msg}")
            return True
        else:
            print(f"  ✗ Validation failed: {msg}")
            return False
    
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Collection failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test collection and manifest creation.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "tests" / "test_config.yaml",
                        help="Path to the configuration YAML file.")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Root directory of the dataset to process (e.g., 'test_dataset'). "
                             "Defaults to output_dir from config if not specified.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for manifest outputs. Defaults to dataset/manifests if not specified.")
    parser.add_argument("--type", choices=["layouts", "graphs", "all", "both"], default="both",
                        help="Type of collection to run. 'both' runs layouts, graphs, and all.")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    dataset_root = args.dataset if args.dataset else resolve_path(config["output_dir"])
    
    if not dataset_root.exists():
        print(f"Error: Dataset root directory not found: {dataset_root}")
        sys.exit(1)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = dataset_root / "manifests"
    
    safe_mkdir(output_dir)
    
    print("\n" + "="*70)
    print("TESTING COLLECTION AND MANIFEST CREATION")
    print("="*70)
    print(f"Dataset directory: {dataset_root}")
    print(f"Output directory: {output_dir}")
    
    success = True
    
    if args.type in ("layouts", "both"):
        if not run_collect_layouts(config, dataset_root, output_dir):
            success = False
    
    if args.type in ("graphs", "both"):
        if not run_collect_graphs(config, dataset_root, output_dir):
            success = False
    
    if args.type in ("all", "both"):
        if not run_collect_all(config, dataset_root, output_dir):
            success = False
    
    if success:
        print("\n" + "="*70)
        print("✔ All collection tests passed successfully!")
        print("="*70)
        print(f"\nManifests created in: {output_dir}")
        sys.exit(0)
    else:
        print("\n" + "="*70)
        print("✗ Some collection tests failed")
        print("="*70)
        sys.exit(1)


if __name__ == "__main__":
    main()

