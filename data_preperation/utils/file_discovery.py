#!/usr/bin/env python3

import csv
import glob
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple


def discover_files(data_type: str, root: Path, manifest: Optional[Path] = None,
                   pattern: Optional[str] = None, column_name: str = None) -> List[Path]:

    # Method 1: From manifest
    if manifest and manifest.exists():
        files = []
        # Auto-infer column name if not provided
        if column_name is None:
            column_name = _get_default_column_name(data_type)
        
        try:
            with open(manifest, newline='', encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row_num, row in enumerate(reader, start=2):
                    if column_name in row and row[column_name]:
                        try:
                            path = Path(row[column_name]).expanduser().resolve()
                            if path.exists():
                                files.append(path)
                            else:
                                print(f"Warning: File in manifest row {row_num} doesn't exist: {path}")
                        except Exception as e:
                            print(f"Warning: Invalid path in manifest row {row_num}: {row[column_name]} - {e}")
            
            if files:
                return files
            else:
                print(f"Warning: No valid files found in manifest {manifest} using column '{column_name}'")
        except Exception as e:
            print(f"Error reading manifest {manifest}: {e}")
            # Fall through to other methods
    
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
        with open(manifest, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                # Skip scene-level layouts
                if row.get("room_id") == "scene":
                    continue
                
                layout_path = Path(row.get("layout_path", ""))
                if layout_path.exists():
                    sid = row.get("scene_id", "")
                    rid = row.get("room_id", "")
                    layouts.append((sid, rid, layout_path))
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
        with open(manifest, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                scene_id = row.get("scene_id", "")
                pc_path = Path(row.get("parquet_file_path", ""))
                if pc_path.exists():
                    scenes.append((scene_id, pc_path))
    else:
        for pc_path in root.rglob("*_sem_pointcloud.parquet"):
            scene_id = pc_path.stem.replace("_sem_pointcloud", "")
            scenes.append((scene_id, pc_path))
    return scenes


def find_room_parquets(root: Path, manifest: Optional[Path] = None) -> List[Path]:
    return discover_files('parquet', root, manifest, column_name='room_parquet_file_path')


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


# =============================================================================
# ID Inference from Paths
# =============================================================================

def infer_ids_from_path(path: Path) -> Tuple[str, int]:
    stem = path.stem  # filename without extension
    
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


def infer_scene_id(path: Path) -> str:
    scene_id, _ = infer_ids_from_path(path)
    return scene_id


def infer_room_id(path: Path) -> int:
    _, room_id = infer_ids_from_path(path)
    return room_id


# =============================================================================
# Additional Discovery Functions
# =============================================================================

def find_room_files(root: Path, manifest: Optional[Path] = None) -> List[Path]:
    if manifest is not None and manifest.exists():
        files = []
        with open(manifest, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try multiple column name variations
                for col_name in ["room_parquet", "room_parquet_file_path"]:
                    if col_name in row and row[col_name]:
                        try:
                            path = Path(row[col_name]).expanduser().resolve()
                            if path.exists():
                                files.append(path)
                        except Exception:
                            pass
        if files:
            return files
    
    # Filesystem scan with multiple patterns
    files = sorted(root.rglob("scene_id=*/room_id=*/*.parquet"))
    if not files:
        files = sorted(root.rglob("*.parquet"))
    return files


def discover_rooms(root: Path, pattern: Optional[str] = None,
                  manifest: Optional[Path] = None) -> List[Path]:
    if manifest:
        return discover_files('parquet', root, manifest, column_name='room_parquet_file_path')
    
    # Default discovery patterns
    if pattern:
        files = sorted(root.rglob(pattern))
        if files:
            return files
    
    files = list(root.rglob("part-*.parquet"))        # old format
    files.extend(root.rglob("*_*[0-9].parquet"))      # new format
    if not files:
        files = list(root.rglob("rooms/*/*.parquet"))
    
    return sorted(files)


def discover_scenes_from_rooms(room_files: List[Path]) -> List[str]:
    scene_ids = set()
    for room_file in room_files:
        scene_id, _ = infer_ids_from_path(room_file)
        if scene_id:
            scene_ids.add(scene_id)
    return sorted(scene_ids)


def discover_scenes_from_manifest(manifest: Path) -> List[str]:
    scene_ids = set()
    with open(manifest, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "scene_id" in row and row["scene_id"]:
                scene_ids.add(row["scene_id"])
    return sorted(scene_ids)

