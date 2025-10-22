import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

# This transform will be used to convert layout images to tensors
layout_transform = transforms.ToTensor()

def collate_fn(batch):
    """
    Collator function that handles a batch of samples.
    - Converts "layout" (PIL.Image) to Tensors and stacks them.
    - Returns "pov" (PIL.Image) and "graph" (str) as lists.
    """
    
    # Check if batch is empty
    if not batch:
        return {}

    # Separate items
    layout_items = []
    pov_items = []
    graph_items = []

    for sample in batch:
        if "layout" in sample:
            layout_items.append(sample["layout"])
        if "pov" in sample:
            pov_items.append(sample["pov"])
        if "graph" in sample:
            graph_items.append(sample["graph"])

    collated_batch = {}

    # Handle "layout" items
    if layout_items:
        # Check the type of the first item
        if isinstance(layout_items[0], torch.Tensor):
            # All are tensors, stack them
            collated_batch["layout"] = torch.stack(layout_items)
        elif isinstance(layout_items[0], Image.Image):
            # They are images, transform and stack
            tensor_layouts = [layout_transform(img) for img in layout_items]
            collated_batch["layout"] = torch.stack(tensor_layouts)
        else:
            # Handle unexpected type
            raise TypeError(f"Unexpected type for layout: {type(layout_items[0])}")
    
    # Return pov and graph items as lists
    if pov_items:
        collated_batch["pov"] = pov_items
    
    if graph_items:
        collated_batch["graph"] = graph_items

    return collated_batch


# ---------- Utility loaders ----------

def load_image(path, transform=None):
    img = Image.open(path).convert("RGB")
    if transform:
        img = transform(img)
    return img


def load_embedding(path):
    if path.endswith(".pt"):
        return torch.load(path, weights_only=False)
    elif path.endswith(".npy"):
        return torch.from_numpy(np.load(path))
    else:
        raise ValueError(f"Unsupported embedding format: {path}")


def load_graph_text(path):
    """Load graph text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def valid_path(x):
    """Check if a path string is valid (not empty, not placeholder)."""
    invalid = {"", "false", "0", "none"}
    return isinstance(x, str) and str(x).strip().lower() not in invalid


# ---------- Unified Layout Dataset ----------

class UnifiedLayoutDataset(Dataset):
    """
    Unified dataset for conditional diffusion training.
    Each item returns dict(pov, graph, layout) for diffusion training.
    - Room samples: pov, graph, layout all valid.
    - Scene samples: pov=None, graph+layout valid.
    
    Args:
        manifest_path: Path to training_manifest.csv (from collect_all.py)
        use_embeddings: If True, load embeddings instead of raw data
        sample_type: 'room', 'scene', or 'both' - which samples to include
        pov_type: 'seg', 'tex', or None - filter by POV type for room samples
        transform: Optional transform for images
        device: Device to load tensors to
    """

    def __init__(
        self, 
        manifest_path,
        use_embeddings=False,
        sample_type="both",  # 'room', 'scene', or 'both'
        pov_type=None,  # 'seg', 'tex', or None (use both)
        transform=None,
        device=None
    ):
        self.use_embeddings = use_embeddings
        self.sample_type = sample_type
        self.pov_type = pov_type
        self.transform = transform 
        self.device = device
        
        # Load manifest
        df = pd.read_csv(manifest_path)
        
        # Filter by sample type
        if sample_type == "room":
            df = df[df["sample_type"] == "room"].reset_index(drop=True)
            print(f"Filtered to room samples only: {len(df)} samples", flush=True)
        elif sample_type == "scene":
            df = df[df["sample_type"] == "scene"].reset_index(drop=True)
            print(f"Filtered to scene samples only: {len(df)} samples", flush=True)
        elif sample_type == "both":
            print(f"Using both room and scene samples: {len(df)} samples", flush=True)
        else:
            raise ValueError(f"Invalid sample_type: {sample_type}. Must be 'room', 'scene', or 'both'")
        
        # Filter by POV type (only for room samples)
        if pov_type is not None:
            if pov_type not in ['seg', 'tex']:
                raise ValueError(f"Invalid pov_type: {pov_type}. Must be 'seg', 'tex', or None")
            
            # Keep scenes (pov_type is empty) OR rooms with matching pov_type
            mask = (df["sample_type"] == "scene") | (df["pov_type"] == pov_type)
            df = df[mask].reset_index(drop=True)
            print(f"Filtered to pov_type='{pov_type}': {len(df)} samples", flush=True)
        
        # Filter out samples with invalid required paths
        valid_mask = (
            df["graph_text"].apply(valid_path) &
            df["layout_image"].apply(valid_path)
        )
        
        # For embeddings mode, also check embedding paths
        if use_embeddings:
            valid_mask = valid_mask & df["layout_embedding"].apply(valid_path)
        
        df = df[valid_mask].reset_index(drop=True)
        
        self.df = df
        self.entries = df.to_dict("records")
        
        # Print dataset statistics
        room_count = (df["sample_type"] == "room").sum()
        scene_count = (df["sample_type"] == "scene").sum()
        
        if pov_type:
            pov_type_count = (df["pov_type"] == pov_type).sum()
            print(f"Dataset loaded: {len(df)} total samples", flush=True)
            print(f"  - Room samples ({pov_type} POVs): {pov_type_count}", flush=True)
            print(f"  - Scene samples: {scene_count}", flush=True)
        else:
            seg_count = (df["pov_type"] == "seg").sum()
            tex_count = (df["pov_type"] == "tex").sum()
            print(f"Dataset loaded: {len(df)} total samples", flush=True)
            print(f"  - Room samples (seg POVs): {seg_count}", flush=True)
            print(f"  - Room samples (tex POVs): {tex_count}", flush=True)
            print(f"  - Scene samples: {scene_count}", flush=True)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        row = self.entries[idx]
        
        is_room = row["sample_type"] == "room"
        
        # ----- POV (only for room samples) -----
        pov = None
        if is_room:
            if self.use_embeddings:
                pov_path = row["pov_embedding"]
                if valid_path(pov_path):
                    pov = load_embedding(pov_path)
                    if self.device:
                        pov = pov.to(self.device)
            else:
                pov_path = row["pov_image"]
                if valid_path(pov_path):
                    pov = load_image(pov_path, self.transform)

        # ----- Graph -----
        if self.use_embeddings:
            graph_path = row["graph_embedding"]
            if valid_path(graph_path):
                graph = load_embedding(graph_path)
                if self.device:
                    graph = graph.to(self.device)
            else:
                # Fallback to text if embedding not available
                graph = load_graph_text(row["graph_text"])
        else:
            # Load as text
            graph = load_graph_text(row["graph_text"])

        # ----- Layout -----
        if self.use_embeddings:
            layout_path = row["layout_embedding"]
            layout = load_embedding(layout_path)
            if self.device:
                layout = layout.to(self.device)
        else:
            layout_path = row["layout_image"]
            layout = load_image(layout_path, self.transform)

        return {
            "sample_id": row["sample_id"],
            "scene_id": row["scene_id"],
            "room_id": row["room_id"] if is_room else None,
            "sample_type": row["sample_type"],
            "pov_type": row.get("pov_type", None),
            "pov": pov,
            "graph": graph,
            "layout": layout
        }


