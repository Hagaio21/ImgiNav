import os
import json
import pandas as pd
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

def load_image(path, transform=None):
    """Load and optionally transform an image."""
    img = Image.open(path).convert("RGB")
    if transform:
        img = transform(img)
    return img

def load_embedding(path):
    """Load embedding from .pt, .pth, or .npy file."""
    path = path.strip()
    lower = path.lower()

    if lower.endswith(".pt") or lower.endswith(".pth"):
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):  # handles rare case if AE ever saves dicts
            for k in ("latent", "z", "embedding"):
                if k in data:
                    data = data[k]
                    break
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Invalid embedding type: {type(data)} in {path}")
        return data.float()

    elif lower.endswith(".npy"):
        return torch.from_numpy(np.load(path, allow_pickle=True)).float()

    else:
        raise ValueError(f"Unsupported embedding file type: {path}")

def load_graph_text(path):
    """Load graph text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def valid_path(x):
    """Check if a path string is valid (not empty, not placeholder)."""
    invalid = {"", "false", "0", "none"}
    return isinstance(x, str) and str(x).strip().lower() not in invalid

def compute_sample_weights(df: pd.DataFrame) -> torch.DoubleTensor:
    """Create grouping key: scene uses 'scene', rooms use room_id"""
    keys = df.apply(lambda r: f"{r['type']}:{r['room_id']}" if r["type"] == "room" else "scene", axis=1)
    counts = keys.value_counts()
    weights = keys.map(lambda k: 1.0 / counts[k])
    weights = weights / weights.sum()
    return torch.DoubleTensor(weights.values)
