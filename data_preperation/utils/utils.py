#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
util.py - Shared utilities for 3D scene processing pipeline

Common functionality extracted from stage1-4 scripts:
- File discovery and path resolution
- Semantic maps handling
- Configuration loading
- Scene/room ID inference
- Taxonomy and category resolution
- Common data structures and transformations
"""

import json
import csv
import re
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

# ----------------------------
# File Discovery & Path Utils
# ----------------------------

def find_files_by_pattern(root: Path, pattern: str) -> List[Path]:
    """Find files using glob pattern."""
    return sorted(root.rglob(pattern))

def find_files_from_manifest(manifest_path: Path, column_name: str = "room_parquet") -> List[Path]:
    """Load file paths from CSV manifest."""
    files = []
    with open(manifest_path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if column_name in row and row[column_name]:
                path = Path(row[column_name]).expanduser().resolve()
                if path.exists():
                    files.append(path)
    return files

def discover_files(root: Path, pattern: str = None, manifest: Path = None, 
                   column_name: str = "room_parquet") -> List[Path]:
    """
    Unified file discovery logic.
    Priority: manifest > pattern > default patterns
    """
    if manifest and manifest.exists():
        return find_files_from_manifest(manifest, column_name)
    
    if pattern:
        return find_files_by_pattern(root, pattern)
    
    # Default discovery patterns
    files = find_files_by_pattern(root, "part-*.parquet")  # old partitioned
    if not files:
        files = find_files_by_pattern(root, "*_*[0-9].parquet")  # new format
    if not files:
        files = find_files_by_pattern(root, "rooms/*/*.parquet")
    if not files:
        files = find_files_by_pattern(root, "*_sem_pointcloud.parquet")
    
    return files

def read_scene_list_file(path: Path) -> List[Path]:
    """Read list of scene paths from text or JSON file."""
    paths = []
    txt = path.read_text(encoding="utf-8").strip()
    
    # Try JSON array first
    try:
        arr = json.loads(txt)
        if isinstance(arr, list):
            return [Path(p) for p in arr]
    except Exception:
        pass
    
    # Fallback: one path per line
    for line in txt.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            paths.append(Path(line))
    
    return paths

def gather_scene_paths(scene_file: str = None, scenes: List[str] = None, 
                      scene_list: str = None) -> List[Path]:
    """Gather scene paths from multiple sources, deduplicated."""
    all_paths = []
    
    if scene_file:
        all_paths.append(Path(scene_file))
    
    if scenes:
        for pat in scenes:
            expanded = [Path(p) for p in glob.glob(pat)]
            if expanded:
                all_paths.extend(expanded)
            else:
                all_paths.append(Path(pat))
    
    if scene_list:
        all_paths.extend(read_scene_list_file(Path(scene_list)))
    
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for p in all_paths:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    
    return unique

def infer_scene_id(path: Path) -> str:
    """Infer scene ID from file path or name."""
    # New filename format: <scene_id>_<room_id>.parquet or <scene_id>_sem_pointcloud.parquet
    m = re.match(r"(.+)_sem_pointcloud\.parquet$", path.name)
    if m:
        return m.group(1)
    
    m = re.match(r"([0-9a-fA-F-]+)_\d+\.parquet$", path.name)
    if m:
        return m.group(1)
    
    # Old path format: .../scene_id=<ID>/...
    m = re.search(r"scene_id=([^/\\]+)", str(path))
    if m:
        return m.group(1)
    
    # Fallback to parent directory name or stem
    if path.parent.name != "." and not path.parent.name.startswith("room"):
        return path.parent.name
    
    return path.stem

def infer_room_id(path: Path) -> int:
    """Infer room ID from file path or name."""
    # New filename format: ..._<room_id>.parquet
    m = re.match(r".+_(\d+)\.parquet$", path.name)
    if m:
        return int(m.group(1))
    
    # Old path format: .../room_id=<num>/...
    m = re.search(r"room_id=(\d+)", str(path))
    if m:
        return int(m.group(1))
    
    return -1

# ----------------------------
# Semantic Maps & Categories
# ----------------------------

def find_semantic_maps_json(start: Path) -> Optional[Path]:
    """Walk up from 'start' to locate semantic_maps.json."""
    for p in [start, *start.parents]:
        cand = p / "semantic_maps.json"
        if cand.exists():
            return cand
    return None

def load_semantic_maps(maps_path: Path) -> Dict:
    """Load and return semantic maps dictionary."""
    return json.loads(maps_path.read_text(encoding="utf-8"))

def get_floor_label_ids(maps_path: Path) -> Tuple[int, ...]:
    """Extract floor label IDs from semantic maps."""
    maps = load_semantic_maps(maps_path)
    ids = set()
    
    # Check label2id mapping
    if "label2id" in maps:
        for name, lid in maps["label2id"].items():
            if str(name).strip().lower() == "floor":
                ids.add(int(lid))
    
    # Check id2label mapping
    if "id2label" in maps:
        for lid, name in maps["id2label"].items():
            if str(name).strip().lower() == "floor":
                try:
                    ids.add(int(lid))
                except (ValueError, TypeError):
                    pass
    
    if not ids:
        raise RuntimeError(f"'floor' not found in {maps_path}")
    
    return tuple(sorted(ids))

def load_global_palette(start: Path) -> Dict[int, Tuple[int, int, int]]:
    """Load color palette from semantic maps."""
    maps_path = find_semantic_maps_json(start)
    if not maps_path:
        raise RuntimeError("semantic_maps.json not found")
    
    maps = load_semantic_maps(maps_path)
    if "id2color" not in maps:
        raise RuntimeError("id2color missing in semantic_maps.json")
    
    # Normalize keys to int
    return {int(k): tuple(v) for k, v in maps["id2color"].items()}

def update_semantic_maps(root: Path, values_by_category: Dict[str, List[str]], 
                        freeze: bool = False) -> Tuple[Dict, Path]:
    """
    Update or create semantic maps with new values.
    
    Args:
        root: Directory to store semantic_maps.json
        values_by_category: Dict with keys like "label", "room", "category" etc.
        freeze: If True, raise error on unseen values
        
    Returns:
        (maps_dict, maps_path)
    """
    root.mkdir(parents=True, exist_ok=True)
    maps_path = root / "semantic_maps.json"
    
    maps = load_semantic_maps(maps_path) if maps_path.exists() else {}
    changed = False
    
    for category, values in values_by_category.items():
        key = f"{category}2id"
        current = {str(k): int(v) for k, v in maps.get(key, {}).items()}
        unique_values = sorted({v for v in values if v}, key=lambda s: s.lower())
        
        if freeze:
            unknown = [v for v in unique_values if v not in current]
            if unknown:
                raise RuntimeError(f"freeze_maps ON; unseen in '{category}': {unknown[:20]}")
        else:
            next_id = (max(current.values()) + 1) if current else 1
            for value in unique_values:
                if value not in current:
                    current[value] = next_id
                    next_id += 1
                    changed = True
        
        maps[key] = current
    
    if changed or not maps_path.exists():
        maps_path.write_text(json.dumps(maps, indent=2), encoding="utf-8")
    
    return maps, maps_path

# ----------------------------
# Configuration Loading
# ----------------------------

def load_config_file(path: Path) -> Dict:
    """Load JSON or YAML config file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    if path.suffix.lower() in (".yml", ".yaml"):
        try:
            import yaml
        except ImportError as e:
            raise RuntimeError("YAML config requested but 'pyyaml' is not installed") from e
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    
    if not isinstance(data, dict):
        raise ValueError("Top-level config must be an object/dict")
    
    return data

def resolve_config_with_profile(config_path: str = None, profile: str = None) -> Dict:
    """Load config and apply profile if specified."""
    if not config_path:
        return {}
    
    config = load_config_file(Path(config_path))
    
    # Apply profile if specified
    profile_name = profile or config.get("profile")
    if profile_name and "profiles" in config:
        if profile_name not in config["profiles"]:
            raise ValueError(f"Profile '{profile_name}' not found in config")
        
        profile_config = config["profiles"][profile_name]
        base_config = {k: v for k, v in config.items() 
                      if k not in ("profiles", "profile")}
        base_config.update(profile_config)
        return base_config
    
    return config

# ----------------------------
# Taxonomy & Category Resolution  
# ----------------------------

def normalize_label(label: str) -> str:
    """Normalize label for robust matching."""
    if not label:
        return ""
    
    import re
    s = label.lower().strip()
    s = re.sub(r"[/_]+", " ", s)
    s = re.sub(r"[^a-z0-9\s\-]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def load_taxonomy(taxonomy_path: Path) -> Dict:
    """Load taxonomy with categories, super-categories, and aliases."""
    if not taxonomy_path or not taxonomy_path.exists():
        raise FileNotFoundError(f"Taxonomy not found: {taxonomy_path}")
    
    data = load_config_file(taxonomy_path)
    
    supers = data.get("super_categories_3d", [])
    cats = data.get("categories_3d", [])
    aliases = data.get("aliases", {})
    
    super_by_name = {c["category"]: int(c["id"]) for c in supers}
    cat_by_name = {c["category"]: int(c["id"]) for c in cats}
    cat_to_super = {c["category"]: c.get("super", "Other") for c in cats}
    
    # Normalized lookup indices
    idx_super_norm = {normalize_label(k): k for k in super_by_name.keys()}
    idx_cat_norm = {normalize_label(k): k for k in cat_by_name.keys()}
    idx_alias_norm = {normalize_label(k): v for k, v in aliases.items()}
    
    return {
        "super_by_name": super_by_name,
        "cat_by_name": cat_by_name,
        "cat_to_super": cat_to_super,
        "idx_super_norm": idx_super_norm,
        "idx_cat_norm": idx_cat_norm,
        "idx_alias_norm": idx_alias_norm,
    }

def resolve_category(raw_label: str, model_info_item: Dict = None, 
                    taxonomy: Dict = None, alias_only_merge: bool = False) -> Dict[str, str]:
    """
    Resolve raw label to category information.
    
    Returns:
        Dict with keys: category, super, merged
    """
    if not taxonomy:
        return {
            "category": "",
            "super": raw_label or "unknown",
            "merged": raw_label or "unknown"
        }
    
    # Prefer model_info category if available
    candidate = (model_info_item or {}).get("category") or raw_label or "unknown"
    normalized = normalize_label(candidate)
    
    # Apply alias mapping first
    alias_target = taxonomy["idx_alias_norm"].get(normalized)
    if alias_target:
        candidate = alias_target
        normalized = normalize_label(candidate)
    
    # Try to match fine category
    category_name = taxonomy["idx_cat_norm"].get(normalized, "")
    if category_name:
        super_name = taxonomy["cat_to_super"].get(category_name, "Other")
    else:
        # Maybe it's already a super category
        super_name = taxonomy["idx_super_norm"].get(normalized, "Other")
    
    # Choose merged label
    if alias_only_merge:
        merged = category_name or super_name
    else:
        merged = super_name
    
    return {
        "category": category_name,
        "super": super_name,
        "merged": merged
    }

# ----------------------------
# Coordinate Transformations
# ----------------------------

def world_to_local_coords(points: np.ndarray, origin: np.ndarray, 
                         u: np.ndarray, v: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Transform world coordinates to local UVH frame."""
    R = np.stack([u, v, n], axis=1)  # world -> local transformation matrix
    return (points - origin) @ R

def get_bounds_from_points(points: np.ndarray) -> Tuple[float, float, float, float]:
    """Get 2D bounds (umin, umax, vmin, vmax) from UV points."""
    if points.shape[0] == 0:
        return (0.0, 1.0, 0.0, 1.0)
    
    umin, vmin = points.min(axis=0)
    umax, vmax = points.max(axis=0)
    return float(umin), float(umax), float(vmin), float(vmax)

# ----------------------------
# Common Data Loading
# ----------------------------

def load_room_meta(room_dir: Path) -> Optional[Dict]:
    """
    Load room metadata JSON. Tries multiple naming conventions:
    - <scene>_<room>_meta.json (new format)
    - meta.json (legacy)
    - room_meta.json (legacy)
    """
    # Try new format first
    candidates = list(room_dir.glob("*_meta.json"))
    if candidates:
        meta_path = candidates[0]
    else:
        # Try legacy formats
        for name in ("meta.json", "room_meta.json"):
            meta_path = room_dir / name
            if meta_path.exists():
                break
        else:
            return None
    
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None

def extract_frame_from_meta(meta: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple, float, Tuple]:
    """Extract coordinate frame information from room meta."""
    origin = np.array(meta["origin_world"], dtype=np.float32)
    u = np.array(meta["u_world"], dtype=np.float32)
    v = np.array(meta["v_world"], dtype=np.float32)
    n = np.array(meta["n_world"], dtype=np.float32)
    uv_bounds = tuple(meta["uv_bounds"])  # (umin, umax, vmin, vmax)
    yaw_auto = float(meta.get("yaw_auto", 0.0))
    map_band = tuple(meta.get("map_band_m", [0.05, 0.50]))
    
    return origin, u, v, n, uv_bounds, yaw_auto, map_band

# ----------------------------
# Common Validation & Helpers
# ----------------------------

def ensure_columns_exist(df, required_columns: List[str], source: str = "dataframe"):
    """Validate that required columns exist in DataFrame."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns {missing} in {source}")

def safe_mkdir(path: Path, parents: bool = True, exist_ok: bool = True):
    """Safe directory creation with error handling."""
    try:
        path.mkdir(parents=parents, exist_ok=exist_ok)
    except Exception as e:
        raise RuntimeError(f"Failed to create directory {path}: {e}")

def write_json(data: Dict, path: Path, indent: int = 2):
    """Write JSON data to file with error handling."""
    try:
        safe_mkdir(path.parent)
        path.write_text(json.dumps(data, indent=indent), encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to write JSON to {path}: {e}")

# ----------------------------
# Progress & Logging Helpers
# ----------------------------

def create_progress_tracker(total: int, description: str = "Processing"):
    """Create a simple progress tracking function."""
    def update_progress(current: int, item_name: str = "", success: bool = True):
        status = "✓" if success else "✗"
        percentage = (current / total) * 100 if total > 0 else 0
        print(f"[{current}/{total}] ({percentage:.1f}%) {status} {description} {item_name}", flush=True)
    
    return update_progress