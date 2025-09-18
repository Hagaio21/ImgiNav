"""
file_utils.py
-------------
Generic file operations for loading and saving data.
This module must never import from dataset-specific code.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict


class SavePolicy:
    """
    Policy for handling file overwrites and safety checks.
    For now this is just a placeholder, extend if you need
    overwrite/skip/backup behavior.
    """
    pass


# ---------- JSON ----------
def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str, indent: int = 2) -> None:
    """Save dict as JSON file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)


# ---------- NumPy ----------
def load_npy(path: str) -> np.ndarray:
    """Load numpy array from .npy file."""
    return np.load(path)


def save_npy(arr: np.ndarray, path: str) -> None:
    """Save numpy array to .npy file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)


def save_npz(path: str, **arrays: np.ndarray) -> None:
    """Save multiple numpy arrays into a compressed .npz file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)


# ---------- CSV ----------
def save_csv(df: pd.DataFrame, path: str) -> None:
    """Save pandas DataFrame as CSV."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


# ---------- Parquet ----------
def load_parquet(path: str) -> pd.DataFrame:
    """Load parquet file into a pandas DataFrame."""
    return pd.read_parquet(path)


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """Save pandas DataFrame as parquet file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="snappy", index=False)
