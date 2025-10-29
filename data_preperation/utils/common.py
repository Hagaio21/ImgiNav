#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Dict, List


def safe_mkdir(path: Path, parents: bool = True, exist_ok: bool = True):
    try:
        path.mkdir(parents=parents, exist_ok=exist_ok)
    except Exception as e:
        raise RuntimeError(f"Failed to create directory {path}: {e}")


def write_json(data: Dict, path: Path, indent: int = 2):
    try:
        safe_mkdir(path.parent)
        path.write_text(json.dumps(data, indent=indent), encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to write JSON to {path}: {e}")


def create_progress_tracker(total: int, description: str = "Processing"):
    def update_progress(current: int, item_name: str = "", success: bool = True):
        status = "✓" if success else "✗"
        percentage = (current / total) * 100 if total > 0 else 0
        print(f"[{current}/{total}] ({percentage:.1f}%) {status} {description} {item_name}", flush=True)
    return update_progress


def load_config_with_profile(config_path: str = None, profile: str = None) -> Dict:
    if not config_path:
        return {}
    
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
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
    
    profile_name = profile or data.get("profile")
    if profile_name and "profiles" in data:
        if profile_name not in data["profiles"]:
            raise ValueError(f"Profile '{profile_name}' not found")
        
        base_config = {k: v for k, v in data.items() if k not in ("profiles", "profile")}
        base_config.update(data["profiles"][profile_name])
        return base_config
    
    return data


def ensure_columns_exist(df, required_columns: List[str], source: str = "dataframe"):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns {missing} in {source}")

