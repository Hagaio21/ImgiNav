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

    # --- START MODIFICATION ---
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
    # --- END MODIFICATION ---
    
    # Return pov and graph items as lists (from previous fix)
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
             pov_type=None, data_mode="rooms_and_scenes",
             transform=None, device=None):
        self.use_embeddings = use_embeddings
        self.pov_type = pov_type
        self.data_mode = data_mode
        self.transform = transform 
        self.device = device
        # Load both manifests
        room_df = pd.read_csv(room_manifest)
        scene_df = pd.read_csv(scene_manifest)

    # --- Load and combine manifests based on data_mode ---
        dfs_to_concat = []

        if self.data_mode in ["rooms_only", "rooms_and_scenes"]:
            room_df = pd.read_csv(room_manifest)
            # --- Standardize room schema ---
            room_df = room_df.rename(columns={
                "ROOM_GRAPH_PATH": "GRAPH_PATH",
                "ROOM_GRAPH_EMBEDDING_PATH": "GRAPH_EMBEDDING_PATH",
                "ROOM_LAYOUT_PATH": "LAYOUT_PATH",
                "ROOM_LAYOUT_EMBEDDING_PATH": "LAYOUT_EMBEDDING_PATH"
            })
            # --- Common column order ---
            cols = [
                "SCENE_ID", "ROOM_ID", "POV_TYPE", "POV_PATH", "POV_EMBEDDING_PATH",
                "GRAPH_PATH", "GRAPH_EMBEDDING_PATH", "LAYOUT_PATH", "LAYOUT_EMBEDDING_PATH"
            ]
            room_df = room_df[cols]

            # --- Filter by POV mode if specified ---
            if pov_type is not None:
                room_df = room_df[room_df["POV_TYPE"] == pov_type].reset_index(drop=True)
                print(f"Filtered to POV mode '{pov_type}': {len(room_df)} room samples", flush=True)

            dfs_to_concat.append(room_df)

        if self.data_mode in ["scenes_only", "rooms_and_scenes"]:
            scene_df = pd.read_csv(scene_manifest)
            # --- Standardize scene schema ---
            scene_df["ROOM_ID"] = ""
            scene_df["POV_TYPE"] = ""
            scene_df["POV_PATH"] = ""
            scene_df["POV_EMBEDDING_PATH"] = ""
            scene_df = scene_df.rename(columns={
                        "SCENE_GRAPH_PATH": "GRAPH_PATH",
                        "SCENE_GRAPH_EMBEDDING_PATH": "GRAPH_EMBEDDING_PATH",
                        "SCENE_LAYOUT_PATH": "LAYOUT_PATH",
                        "SCENE_LAYOUT_EMBEDDING_PATH": "LAYOUT_EMBEDDING_PATH"
                    })
            # --- Common column order ---
            cols = [
                "SCENE_ID", "ROOM_ID", "POV_TYPE", "POV_PATH", "POV_EMBEDDING_PATH",
                "GRAPH_PATH", "GRAPH_EMBEDDING_PATH", "LAYOUT_PATH", "LAYOUT_EMBEDDING_PATH"
            ]
            scene_df = scene_df[cols]
            dfs_to_concat.append(scene_df)

        if not dfs_to_concat:
            raise ValueError(f"Invalid data_mode '{self.data_mode}' or no data loaded.")

        # --- Merge manifests ---
        df = pd.concat(dfs_to_concat, ignore_index=True)

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


