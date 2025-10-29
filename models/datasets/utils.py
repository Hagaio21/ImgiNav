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

def load_data_with_embedding_fallback(row, embedding_key, raw_key, transform=None, device=None, use_embeddings=False):

    if use_embeddings and embedding_key in row:
        embedding_path = row[embedding_key]
        if valid_path(embedding_path):
            data = load_embedding(embedding_path)
            if device:
                data = data.to(device)
            return data
    
    # Fallback to raw data
    if raw_key in row:
        raw_path = row[raw_key]
        if valid_path(raw_path):
            if raw_path.endswith(('.txt', '.json')):
                # Text data
                with open(raw_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            else:
                # Image data
                return load_image(raw_path, transform=transform)
    
    return None

def compute_sample_weights(df: pd.DataFrame) -> torch.DoubleTensor:
    """Create grouping key: scene uses 'scene', rooms use room_id"""
    keys = df.apply(lambda r: f"{r['type']}:{r['room_id']}" if r["type"] == "room" else "scene", axis=1)
    counts = keys.value_counts()
    weights = keys.map(lambda k: 1.0 / counts[k])
    weights = weights / weights.sum()
    return torch.DoubleTensor(weights.values)

def build_datasets(dataset_cfg, transform=None):
    """
    Build train/val datasets from manifest and configuration.
    Supports both raw layouts and precomputed latent embeddings.
    
    Args:
        dataset_cfg: Dictionary with keys:
            - manifest: Path to manifest CSV
            - split_ratio: Train/val split ratio (default: 0.9)
            - seed: Random seed (default: 42)
            - one_hot: Whether to use one-hot encoding (default: False)
            - taxonomy_path: Path to taxonomy.json (optional)
            - return_embeddings: Whether to return embeddings (default: False)
            - skip_empty: Whether to skip empty samples (default: True)
        transform: Optional transform. If None, will use ToTensor() unless return_embeddings=True
    
    Returns:
        train_ds, val_ds: Tuple of train and validation datasets
    """
    from torch.utils.data import random_split
    from .datasets import LayoutDataset
    
    manifest_path = dataset_cfg["manifest"]
    split_ratio = dataset_cfg.get("split_ratio", 0.9)
    seed = dataset_cfg.get("seed", 42)
    
    # Set transform: use provided transform, or ToTensor() unless using embeddings
    if transform is None:
        transform = transforms.ToTensor() if not dataset_cfg.get("return_embeddings", False) else None
    
    dataset = LayoutDataset(
        manifest_path=manifest_path,
        transform=transform,
        mode="all",
        one_hot=dataset_cfg.get("one_hot", False),
        taxonomy_path=dataset_cfg.get("taxonomy_path"),
        return_embeddings=dataset_cfg.get("return_embeddings", False),
        skip_empty=dataset_cfg.get("skip_empty", True),
    )
    
    n_total = len(dataset)
    n_train = int(n_total * split_ratio)
    n_val = n_total - n_train
    gen = torch.Generator().manual_seed(seed)
    
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)
    return train_ds, val_ds

def build_dataloaders(dataset_cfg, transform=None):

    import random
    from torch.utils.data import DataLoader
    from .collate import collate_skip_none
    from common.utils import set_seeds
    
    seed = dataset_cfg.get("seed", 42)
    batch_size = dataset_cfg.get("batch_size", 16)
    num_workers = dataset_cfg.get("num_workers", 4)
    shuffle = dataset_cfg.get("shuffle", True)
    pin_memory = dataset_cfg.get("pin_memory", False)
    
    set_seeds(seed)
    
    train_ds, val_ds = build_datasets(dataset_cfg, transform=transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_skip_none
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_skip_none
    )
    return train_ds, val_ds, train_loader, val_loader

def save_split_csvs(train_ds, val_ds, output_dir):
    """
    Save train/val split information to CSV files.
    
    Args:
        train_ds: Training dataset (Subset from random_split)
        val_ds: Validation dataset (Subset from random_split)
        output_dir: Directory to save CSV files
    """
    train_paths = [train_ds.dataset.entries[i]["layout_path"] for i in train_ds.indices]
    val_paths = [val_ds.dataset.entries[i]["layout_path"] for i in val_ds.indices]
    
    train_df = pd.DataFrame({"layout_path": train_paths})
    val_df = pd.DataFrame({"layout_path": val_paths})
    
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "trained_on.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "evaluated_on.csv"), index=False)
