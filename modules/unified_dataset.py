import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


# ---------- Utility loaders ----------

def load_image(path, transform=None):
    img = Image.open(path).convert("RGB")
    if transform:
        img = transform(img)
    return img


def load_embedding(path):
    if path.endswith(".pt"):
        return torch.load(path)
    elif path.endswith(".npy"):
        return torch.from_numpy(np.load(path))
    else:
        raise ValueError(f"Unsupported embedding format: {path}")


def load_graph(path, use_embeddings=False):
    if use_embeddings:
        return load_embedding(path)
    with open(path, "r") as f:
        return json.load(f)


def valid_path(x):
    invalid = {"", "false", "0", "none"}
    return isinstance(x, str) and str(x).strip().lower() not in invalid


# ---------- Unified Layout Dataset ----------

class UnifiedLayoutDataset(Dataset):
    """
    Unified dataset combining room and scene manifests.
    Each item returns dict(pov, graph, layout) for diffusion training.
    - Room samples: pov, graph, layout all valid.
    - Scene samples: pov=None, graph+layout valid.
    
    Args:
        room_manifest: Path to room manifest CSV
        scene_manifest: Path to scene manifest CSV
        use_embeddings: If True, load embeddings instead of raw data
        pov_mode: Which POV to use - 'seg', 'tex', or None (no POV filtering)
        transform: Optional transform for images
        device: Device to load tensors to
    """

    def __init__(self, room_manifest, scene_manifest, use_embeddings=False, 
                 pov_type=None, transform=None, device=None):
        self.use_embeddings = use_embeddings
        self.pov_type = pov_type  # 'seg', 'tex', or None
        self.transform = transform
        self.device = device

        # Load both manifests
        room_df = pd.read_csv(room_manifest)
        scene_df = pd.read_csv(scene_manifest)

        # --- Standardize schemas ---
        room_df = room_df.rename(columns={
            "ROOM_GRAPH_PATH": "GRAPH_PATH",
            "ROOM_GRAPH_EMBEDDING_PATH": "GRAPH_EMBEDDING_PATH",
            "ROOM_LAYOUT_PATH": "LAYOUT_PATH",
            "ROOM_LAYOUT_EMBEDDING_PATH": "LAYOUT_EMBEDDING_PATH"
        })

        scene_df["ROOM_ID"] = ""
        scene_df["POV_TYPE"] = ""
        scene_df["POV_PATH"] = ""
        scene_df["POV_EMBEDDING_PATH"] = ""

        # --- Common column order ---
        cols = [
            "SCENE_ID", "ROOM_ID", "POV_TYPE", "POV_PATH", "POV_EMBEDDING_PATH",
            "GRAPH_PATH", "GRAPH_EMBEDDING_PATH", "LAYOUT_PATH", "LAYOUT_EMBEDDING_PATH"
        ]
        room_df = room_df[cols]
        scene_df = scene_df[cols]

        # --- Filter by POV mode if specified ---
        if pov_type is not None:
            # Only keep room samples with matching POV type
            room_df = room_df[room_df["POV_TYPE"] == pov_type].reset_index(drop=True)
            print(f"Filtered to POV mode '{pov_type}': {len(room_df)} room samples", flush=True)

        # --- Merge manifests ---
        df = pd.concat([room_df, scene_df], ignore_index=True)

        # --- Filter invalid paths ---
        mask = (
            df["GRAPH_PATH"].apply(valid_path)
            & df["GRAPH_EMBEDDING_PATH"].apply(valid_path)
            & df["LAYOUT_PATH"].apply(valid_path)
            & df["LAYOUT_EMBEDDING_PATH"].apply(valid_path)
        )
        df = df[mask].reset_index(drop=True)
        
        self.df = df
        self.entries = df.to_dict("records")
        
        # Print dataset statistics
        num_with_pov = (df["POV_PATH"].apply(valid_path)).sum()
        num_without_pov = len(df) - num_with_pov
        print(f"Dataset: {len(df)} total samples ({num_with_pov} with POV, {num_without_pov} without)", flush=True)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        row = self.entries[idx]

        # ----- POV -----
        pov = None
        pov_path = row["POV_EMBEDDING_PATH"] if self.use_embeddings else row["POV_PATH"]
        if valid_path(pov_path):
            pov = load_embedding(pov_path) if self.use_embeddings else load_image(pov_path, self.transform)
            if self.device and isinstance(pov, torch.Tensor):
                pov = pov.to(self.device)

        # ----- Graph -----
        graph_path = row["GRAPH_EMBEDDING_PATH"] if self.use_embeddings else row["GRAPH_PATH"]
        graph = load_graph(graph_path, use_embeddings=self.use_embeddings)
        if self.device and isinstance(graph, torch.Tensor):
            graph = graph.to(self.device)

        # ----- Layout -----
        layout_path = row["LAYOUT_EMBEDDING_PATH"] if self.use_embeddings else row["LAYOUT_PATH"]
        layout = load_embedding(layout_path) if self.use_embeddings else load_image(layout_path, self.transform)
        if self.device and isinstance(layout, torch.Tensor):
            layout = layout.to(self.device)

        return {
            "scene_id": row["SCENE_ID"],
            "room_id": row["ROOM_ID"] if row["ROOM_ID"] else None,
            "pov_type": row["POV_TYPE"] if row["POV_TYPE"] else None,
            "pov": pov,
            "graph": graph,
            "layout": layout
        }


def collate_fn(batch):
    """Custom collate function to handle None POV values in batches"""
    # Collect all non-None POVs
    pov_list = [b['pov'] for b in batch if b['pov'] is not None]
    
    return {
        'scene_id': [b['scene_id'] for b in batch],
        'room_id': [b['room_id'] for b in batch],
        'pov_type': [b['pov_type'] for b in batch],
        'pov': torch.stack(pov_list) if pov_list else None,
        'graph': torch.stack([b['graph'] for b in batch]),
        'layout': torch.stack([b['layout'] for b in batch]),
    }