#!/usr/bin/env python3

import csv
from pathlib import Path
from typing import List, Optional, Tuple

def infer_ids_from_path(path: Path) -> Tuple[str, ...]:
    """Extract scene_id and optionally room_id from file path."""
    path = Path(path)
    stem = path.stem
    
    # Try to extract IDs from filename patterns
    # Common patterns: scene_id.json, scene_id_room_id.parquet, etc.
    parts = stem.split('_')
    
    if len(parts) >= 1:
        scene_id = parts[0]
        if len(parts) >= 2 and parts[1].isdigit():
            return (scene_id, int(parts[1]))
        return (scene_id,)
    
    # Fallback: use stem as scene_id
    return (stem,)

def gather_paths_from_sources(scene_file: Optional[str] = None,
                              scenes: Optional[List[str]] = None,
                              scene_list: Optional[str] = None) -> List[Path]:
    """Gather scene file paths from various sources."""
    paths = []
    
    # Priority 1: scene_list file
    if scene_list:
        list_path = Path(scene_list)
        if list_path.exists():
            with open(list_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        p = Path(line)
                        # Try to resolve relative paths
                        if not p.is_absolute():
                            # Try relative to current directory first
                            if not p.exists():
                                # Try relative to list file's directory
                                p = list_path.parent / p
                        if p.exists():
                            paths.append(p.resolve())
            if paths:
                return paths
    
    # Priority 2: explicit scene files
    if scenes:
        for s in scenes:
            p = Path(s)
            if p.exists():
                paths.append(p.resolve())
        if paths:
            return paths
    
    # Priority 3: single scene_file (can be a directory or file)
    if scene_file:
        p = Path(scene_file)
        if p.exists():
            if p.is_dir():
                # If it's a directory, find all JSON files in it
                json_files = list(p.glob("*.json"))
                if json_files:
                    return [f.resolve() for f in json_files]
            else:
                return [p.resolve()]
        # Try as glob pattern
        parent = p.parent if p.parent != Path('.') else Path('.')
        pattern = p.name
        found = list(parent.glob(pattern))
        if found:
            return [f.resolve() for f in found]
    
    return paths

