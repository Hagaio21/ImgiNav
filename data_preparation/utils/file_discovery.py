#!/usr/bin/env python3

import csv
import glob
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple


def _read_manifest_files(manifest: Path, column_name: str, 
                        filter_func: Optional[callable] = None) -> List[Path]:
    """Helper function to read files from manifest with optional filtering."""
    files = []
    try:
        from common.file_io import read_manifest
        rows = read_manifest(manifest)
        for row_num, row in enumerate(rows, start=2):
            if column_name in row and row[column_name]:
                try:
                    path = Path(row[column_name]).expanduser().resolve()
                    if path.exists() and (filter_func is None or filter_func(row)):
                        files.append(path)
                    elif not path.exists():
                        print(f"Warning: File in manifest row {row_num} doesn't exist: {path}")
                except Exception as e:
                    print(f"Warning: Invalid path in manifest row {row_num}: {row[column_name]} - {e}")
    except Exception as e:
        print(f"Error reading manifest {manifest}: {e}")
    return files


def discover_files(data_type: str, root: Path, manifest: Optional[Path] = None,
                   pattern: Optional[str] = None, column_name: str = None) -> List[Path]:

    # Method 1: From manifest
    if manifest and manifest.exists():
        if column_name is None:
            column_name = _get_default_column_name(data_type)
        
        files = _read_manifest_files(manifest, column_name)
        if files:
            return files
        else:
            print(f"Warning: No valid files found in manifest {manifest} using column '{column_name}'")
    
    # Method 2: Pattern-based
    if pattern:
        files = sorted(root.rglob(pattern))
        if files:
            return files
        else:
            print(f"Warning: No files found with pattern '{pattern}' in {root}")
    
    # Method 3: Default patterns by data type
    default_patterns = {
        'parquet': ["part-*.parquet", "*_*[0-9].parquet", "rooms/*/*.parquet", "*_sem_pointcloud.parquet"],
        'layout': ["*_room_seg_layout.png", "*_scene_layout.png", "*layout.png"],
        'graph': ["*_graph.json"],
        'pov': ["povs/tex/*.png", "povs/seg/*.png"],
        'pointcloud': ["*_sem_pointcloud.parquet"],
    }
    
    patterns = default_patterns.get(data_type, ["**/*"])
    for default_pattern in patterns:
        files = sorted(root.rglob(default_pattern))
        if files:
            print(f"Found {len(files)} files using default pattern '{default_pattern}'")
            return files
    
    print(f"Error: No files found in {root} using any method for type '{data_type}'")
    return []


def _get_default_column_name(data_type: str) -> str:
    column_map = {
        'parquet': 'room_parquet_file_path',
        'layout': 'layout_path',
        'graph': 'graph_path',
        'pov': 'pov_path',
        'pointcloud': 'parquet_file_path',
    }
    return column_map.get(data_type, 'file_path')


def find_layouts(root: Optional[Path], manifest: Optional[Path]) -> List[Tuple[str, str, Path]]:
    layouts = []
    if manifest and manifest.exists():
        def filter_func(row):
            return row.get("room_id") != "scene"
        
        files = _read_manifest_files(manifest, "layout_path", filter_func)
        for layout_path in files:
            # Re-read manifest to get scene_id and room_id
            with open(manifest, newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    if Path(row.get("layout_path", "")).resolve() == layout_path:
                        sid = row.get("scene_id", "")
                        rid = row.get("room_id", "")
                        layouts.append((sid, rid, layout_path))
                        break
    elif root:
        for p in root.rglob("*_room_seg_layout.png"):
            parts = p.stem.split("_")
            if len(parts) >= 3 and parts[1] == "scene":
                continue
            if len(parts) >= 3:
                sid, rid = parts[0], parts[1]
                layouts.append((sid, rid, p))
    return layouts


def find_scene_pointclouds(root: Path, manifest: Optional[Path] = None) -> List[Tuple[str, Path]]:
    scenes = []
    if manifest and manifest.exists():
        files = _read_manifest_files(manifest, "parquet_file_path")
        for pc_path in files:
            # Re-read manifest to get scene_id
            with open(manifest, newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    if Path(row.get("parquet_file_path", "")).resolve() == pc_path:
                        scene_id = row.get("scene_id", "")
                        scenes.append((scene_id, pc_path))
                        break
    else:
        for pc_path in root.rglob("*_sem_pointcloud.parquet"):
            scene_id = pc_path.stem.replace("_sem_pointcloud", "")
            scenes.append((scene_id, pc_path))
    return scenes


def find_room_files(root: Path, manifest: Optional[Path] = None, 
                     pattern: Optional[str] = None) -> List[Path]:
    """
    Unified function to find room parquet files from manifest or filesystem scan.
    Replaces find_room_parquets, find_room_files (old), and discover_rooms.
    """
    if manifest is not None and manifest.exists():
        files = discover_files('parquet', root, manifest, column_name='room_parquet_file_path')
        if files:
            return files
    
    if pattern:
        files = sorted(root.rglob(pattern))
        if files:
            return files
    
    # Try multiple filesystem patterns in order
    patterns = [
        "scene_id=*/room_id=*/*.parquet",  # old structured format
        "part-*.parquet",                  # old naming
        "*_*[0-9].parquet",                # new format
        "rooms/*/*.parquet",               # structured dirs
        "*.parquet"                        # fallback
    ]
    
    for pat in patterns:
        files = sorted(root.rglob(pat))
        if files:
            return files
    
    return []


def gather_paths_from_sources(file_path: Optional[str] = None, patterns: Optional[List[str]] = None,
                             list_file: Optional[str] = None) -> List[Path]:
    all_paths = []
    
    # Single file
    if file_path:
        all_paths.append(Path(file_path))
    
    # Pattern list
    if patterns:
        for pat in patterns:
            expanded = [Path(p) for p in glob.glob(pat)]
            all_paths.extend(expanded if expanded else [Path(pat)])
    
    # List file (JSON array or line-separated)
    if list_file:
        path = Path(list_file)
        txt = path.read_text(encoding="utf-8").strip()
        
        # Try JSON first
        try:
            arr = json.loads(txt)
            if isinstance(arr, list):
                all_paths.extend(Path(p) for p in arr)
            else:
                raise ValueError("Not a list")
        except Exception:
            # Line-separated fallback
            for line in txt.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    all_paths.append(Path(line))
    
    # Deduplicate
    seen = set()
    unique = []
    for p in all_paths:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    
    return unique


def infer_ids_from_path(path: Path) -> Tuple[str, int]:
    stem = path.stem  # filename without extension

    # Normalize known modality suffixes (ignore sem/rgb variants)
    for token in ("_sem_pointcloud", "_pointcloud", "_sem", "_rgb"):
        if stem.endswith(token):
            stem = stem.replace(token, "")
            break

    # Strategy 1: Simple split on first underscore
    parts = stem.split("_", 1)
    if len(parts) >= 1:
        scene_id = parts[0]
        if len(parts) > 1:
            # Check if second part is numeric (room_id)
            remaining = parts[1]
            # Try to extract room_id
            if remaining.isdigit():
                return scene_id, int(remaining)
            # Try regex for patterns like "scene_room.parquet"
            match = re.match(r"(\d+)", remaining)
            if match:
                return scene_id, int(match.group(1))

    # Strategy 2: Regex for parquet files (new format)
    match = re.match(r"([0-9a-fA-F-]+)_(\d+)\.parquet$", path.name)
    if match:
        return match.group(1), int(match.group(2))

    # Strategy 3: Path-based (old format: scene_id=<ID>/room_id=<ID>/)
    path_str = str(path)
    scene_match = re.search(r"scene_id=([^/\\]+)", path_str)
    room_match = re.search(r"room_id=(\d+)", path_str)
    if scene_match:
        scene_id = scene_match.group(1)
        room_id = int(room_match.group(1)) if room_match else -1
        return scene_id, room_id

    # Fallback: return stem as scene_id
    return stem, -1


def discover_scenes(room_files: Optional[List[Path]] = None, 
                   manifest: Optional[Path] = None) -> List[str]:

    scene_ids = set()
    
    if room_files:
        for room_file in room_files:
            scene_id, _ = infer_ids_from_path(room_file)
            if scene_id:
                scene_ids.add(scene_id)
    
    if manifest and manifest.exists():
        try:
            with open(manifest, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "scene_id" in row and row["scene_id"]:
                        scene_ids.add(row["scene_id"])
        except Exception as e:
            print(f"Warning: Error reading manifest for scene discovery: {e}")
    
    return sorted(scene_ids)

