#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py - Shared utilities for 3D scene processing pipeline (refactored)

Consolidated common functionality with reduced duplication.
"""

import json
import csv
import re
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

# ----------------------------
# File Discovery (consolidated)
# ----------------------------

def discover_files(root: Path, pattern: str = None, manifest: Path = None, 
                   column_name: str = "room_parquet") -> List[Path]:
    """
    Unified file discovery with all methods in one function.
    Priority: manifest > pattern > default patterns
    """
    # Method 1: From manifest
    if manifest and manifest.exists():
        files = []
        with open(manifest, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if column_name in row and row[column_name]:
                    path = Path(row[column_name]).expanduser().resolve()
                    if path.exists():
                        files.append(path)
        return files
    
    # Method 2: Pattern-based
    if pattern:
        return sorted(root.rglob(pattern))
    
    # Method 3: Default patterns
    for default_pattern in [
        "part-*.parquet", "*_*[0-9].parquet", "rooms/*/*.parquet", "*_sem_pointcloud.parquet"
    ]:
        files = sorted(root.rglob(default_pattern))
        if files:
            return files
    
    return []

def gather_paths_from_sources(file_path: str = None, patterns: List[str] = None, 
                             list_file: str = None) -> List[Path]:
    """Gather paths from multiple sources, deduplicated."""
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

# ----------------------------
# ID Inference (consolidated)
# ----------------------------

def infer_ids_from_path(path: Path) -> Tuple[str, int]:
    """Infer scene_id from the file name by taking the part before the first underscore.
    Room_id is parsed if a numeric part follows, otherwise -1.
    """
    stem = path.stem  # filename without extension
    parts = stem.split("_", 1)  # split into [before_first_underscore, rest]
    scene_id = parts[0]

    room_id = -1
    if len(parts) > 1 and parts[1].isdigit():
        room_id = int(parts[1])

    return scene_id, room_id


# ----------------------------
# Aliases / Legacy compatibility
# ----------------------------

def infer_scene_id(path: Path) -> str:
    """Return scene_id only, derived from infer_ids_from_path."""
    scene_id, _ = infer_ids_from_path(path)
    return scene_id

def find_semantic_maps_json(start_path: Path) -> Optional[Path]:
    """Walk up from start_path to find semantic_maps.json."""
    for p in [start_path, *start_path.parents]:
        cand = p / "semantic_maps.json"
        if cand.exists():
            return cand
    return None

def get_floor_label_ids(maps_path: Path) -> Tuple[int, ...]:
    """Load semantic_maps.json and return IDs mapped to 'floor'."""
    maps = json.loads(maps_path.read_text(encoding="utf-8"))
    ids = set()

    for mapping_name, is_key_label in [("label2id", True), ("id2label", False)]:
        if mapping_name in maps:
            for key, value in maps[mapping_name].items():
                label = str(key if is_key_label else value).strip().lower()
                if label == "floor":
                    try:
                        ids.add(int(value if is_key_label else key))
                    except (ValueError, TypeError):
                        pass

    if not ids:
        raise RuntimeError(f"'floor' not found in {maps_path}")
    return tuple(sorted(ids))

# ----------------------------
# Semantic Maps (consolidated)
# ----------------------------

class SemanticMaps:
    """Consolidated semantic maps handling."""
    
    def __init__(self, start_path: Path):
        self.maps_path = self._find_maps_file(start_path)
        self._maps_data = None
    
    def _find_maps_file(self, start: Path) -> Optional[Path]:
        """Walk up from start to locate semantic_maps.json."""
        for p in [start, *start.parents]:
            cand = p / "semantic_maps.json"
            if cand.exists():
                return cand
        return None
    
    @property
    def data(self) -> Dict:
        """Lazy-load maps data."""
        if self._maps_data is None:
            if not self.maps_path:
                raise RuntimeError("semantic_maps.json not found")
            self._maps_data = json.loads(self.maps_path.read_text(encoding="utf-8"))
        return self._maps_data
    
    def get_floor_label_ids(self) -> Tuple[int, ...]:
        """Extract floor label IDs."""
        ids = set()
        
        # Check both label2id and id2label mappings
        for mapping_name, is_key_label in [("label2id", True), ("id2label", False)]:
            if mapping_name in self.data:
                for key, value in self.data[mapping_name].items():
                    label = str(key if is_key_label else value).strip().lower()
                    if label == "floor":
                        try:
                            ids.add(int(value if is_key_label else key))
                        except (ValueError, TypeError):
                            pass
        
        if not ids:
            raise RuntimeError(f"'floor' not found in {self.maps_path}")
        
        return tuple(sorted(ids))
    
    def get_color_palette(self) -> Dict[int, Tuple[int, int, int]]:
        """Load color palette."""
        if "id2color" not in self.data:
            raise RuntimeError("id2color missing in semantic_maps.json")
        
        return {int(k): tuple(v) for k, v in self.data["id2color"].items()}
    
    def update_with_values(self, values_by_category: Dict[str, List[str]], freeze: bool = False):
        """Update maps with new values."""
        if not self.maps_path:
            raise RuntimeError("Cannot update: semantic_maps.json not found")
        
        maps = self.data.copy()
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
        
        if changed:
            self.maps_path.parent.mkdir(parents=True, exist_ok=True)
            self.maps_path.write_text(json.dumps(maps, indent=2), encoding="utf-8")
        
        self._maps_data = maps  # Update cache

# ----------------------------
# Configuration (simplified)
# ----------------------------

def load_config_with_profile(config_path: str = None, profile: str = None) -> Dict:
    """Load and resolve config in one step."""
    if not config_path:
        return {}
    
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    # Load based on extension
    if path.suffix.lower() in (".yml", ".yaml"):
        try:
            import yaml
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except ImportError as e:
            raise RuntimeError("YAML config requested but 'pyyaml' is not installed") from e
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    
    if not isinstance(data, dict):
        raise ValueError("Config must be a dictionary")
    
    # Apply profile if specified
    profile_name = profile or data.get("profile")
    if profile_name and "profiles" in data:
        if profile_name not in data["profiles"]:
            raise ValueError(f"Profile '{profile_name}' not found")
        
        base_config = {k: v for k, v in data.items() if k not in ("profiles", "profile")}
        base_config.update(data["profiles"][profile_name])
        return base_config
    
    return data

# ----------------------------
# Taxonomy (simplified)
# ----------------------------

def load_taxonomy_resolver(taxonomy_path: Path):
    """Load taxonomy and return resolver function (category → super)."""
    if not taxonomy_path or not taxonomy_path.exists():
        return None

    data = json.loads(taxonomy_path.read_text(encoding="utf-8"))

    category2super = data.get("category2super", {})
    aliases = data.get("aliases", {})

    def resolve(raw_label: str, model_info: Dict = None) -> Dict[str, str]:
        candidate = (model_info or {}).get("category") or raw_label or "unknown"

        # Exact category
        if candidate in category2super:
            return {"category": candidate, "super": category2super[candidate]}

        # Alias fallback
        if candidate in aliases:
            alias = aliases[candidate]
            return {"category": alias, "super": category2super.get(alias, "Other")}

        return {"category": "", "super": "Other"}

    return resolve


def load_global_palette(taxonomy_path: Path) -> Dict[int, Tuple[int, int, int]]:
    """
    Load global color palette (id -> RGB tuple) from a taxonomy JSON file.

    Args:
        taxonomy_path: Path to the taxonomy/semantic_maps.json file.

    Returns:
        Dict mapping int label IDs to (R, G, B) tuples.
    """
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"taxonomy file not found: {taxonomy_path}")

    data = json.loads(taxonomy_path.read_text(encoding="utf-8"))

    # Handle list-of-dicts format
    if isinstance(data, list):
        if not data:
            raise RuntimeError(f"taxonomy file {taxonomy_path} is empty")
        data = data[0]

    if "id2color" not in data:
        raise RuntimeError(f"id2color not found in {taxonomy_path}")

    return {int(k): tuple(v) for k, v in data["id2color"].items()}

# ----------------------------
# Room Metadata (simplified)
# ----------------------------

def load_room_meta(room_dir: Path):
    """Load the metadata JSON (room-level or scene-level)."""
    candidates = list(room_dir.glob("*_meta.json"))
    if not candidates:
        candidates = list(room_dir.glob("*_scene_info.json"))
    if not candidates:
        return None

    meta_path = candidates[0]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # Unwrap list
    if isinstance(meta, list) and len(meta) > 0:
        meta = meta[0]

    return meta

def extract_frame_from_meta(meta):
    """
    Extract origin, u, v, n, uv_bounds, yaw_auto, map_band
    from either a room-level *_meta.json or a scene-level *_scene_info.json.
    """
    # Handle list wrapper
    if isinstance(meta, list):
        if not meta:
            raise ValueError("Empty metadata list")
        meta = meta[0]

    # Case 1: Room-level meta.json
    if "origin_world" in meta:
        origin = np.array(meta["origin_world"], dtype=np.float32)
        u = np.array(meta["u_world"], dtype=np.float32)
        v = np.array(meta["v_world"], dtype=np.float32)
        n = np.array(meta["n_world"], dtype=np.float32)
        uv_bounds = tuple(meta["uv_bounds"])
        yaw_auto = float(meta.get("yaw_auto", 0.0))
        map_band = tuple(meta.get("map_band_m", [0.05, 0.50]))
        return origin, u, v, n, uv_bounds, yaw_auto, map_band

    # Case 2: Scene-level scene_info.json
    if "bounds" in meta:
        bounds = meta["bounds"]
        if not bounds or len(bounds) != 2:
            raise ValueError("Invalid bounds in scene_info.json")

        # Bounds are [[xmin,ymin,zmin],[xmax,ymax,zmax]]
        (xmin, ymin, zmin), (xmax, ymax, zmax) = bounds
        origin = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2], dtype=np.float32)

        # Default orthogonal frame
        u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        n = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        uv_bounds = (xmin, xmax, ymin, ymax)
        yaw_auto = 0.0
        map_band = (0.0, zmax - zmin)  # crude height range

        return origin, u, v, n, uv_bounds, yaw_auto, map_band

    raise KeyError("Unrecognized metadata format (no origin_world or bounds)")
# ----------------------------
# Common Helpers (kept)
# ----------------------------

def ensure_columns_exist(df, required_columns: List[str], source: str = "dataframe"):
    """Validate required columns exist."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns {missing} in {source}")

def safe_mkdir(path: Path, parents: bool = True, exist_ok: bool = True):
    """Safe directory creation."""
    try:
        path.mkdir(parents=parents, exist_ok=exist_ok)
    except Exception as e:
        raise RuntimeError(f"Failed to create directory {path}: {e}")

def write_json(data: Dict, path: Path, indent: int = 2):
    """Write JSON with error handling."""
    try:
        safe_mkdir(path.parent)
        path.write_text(json.dumps(data, indent=indent), encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to write JSON to {path}: {e}")

def create_progress_tracker(total: int, description: str = "Processing"):
    """Create simple progress function."""
    def update_progress(current: int, item_name: str = "", success: bool = True):
        status = "✓" if success else "✗"
        percentage = (current / total) * 100 if total > 0 else 0
        print(f"[{current}/{total}] ({percentage:.1f}%) {status} {description} {item_name}", flush=True)
    return update_progress