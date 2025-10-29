import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import numpy as np
import json
from pathlib import Path
from common.taxonomy import Taxonomy

from .utils import load_image, load_embedding, load_graph_text, valid_path, compute_sample_weights

# ---------- Layout Dataset ----------

class LayoutDataset(Dataset):
    """
    Loads layout images for either:
      - RGB reconstruction (one_hot=False)
      - Semantic segmentation (one_hot=True) using taxonomy.json
    """

    def __init__(
        self,
        manifest_path,
        transform=None,
        mode="all",
        one_hot=False,
        taxonomy_path=None,
        skip_empty=True,
        return_embeddings=False,
    ):
        self.df = pd.read_csv(manifest_path)
        self.transform = transform
        self.mode = mode
        self.one_hot = one_hot
        self.skip_empty = skip_empty
        self.return_embeddings = return_embeddings
        self.taxonomy_path = taxonomy_path

        # --- taxonomy mapping ---
        if taxonomy_path is not None:
            self.taxonomy = Taxonomy(taxonomy_path)
            self._build_supercategory_mapping()
        else:
            self.COLOR_TO_CLASS = None
            self.NUM_CLASSES = None

        # --- filtering ---

        # scenes/rooms
        if self.mode != "all":
            self.df = self.df[self.df["type"] == self.mode]

        # full/empty
        if self.skip_empty:
            self.df = self.df[self.df["is_empty"] == False]

        self.entries = self.df.to_dict("records")

    # --------------------------------------------------------
    # Taxonomy helpers
    # --------------------------------------------------------

    def _build_supercategory_mapping(self):
        """
        Build mapping from RGB colors (from id2color) to supercategory indices (from id2super).
        Works when taxonomy defines 'id2color' and 'id2super', but not 'color2id'.
        """
        id2color = self.taxonomy.get("id2color")
        id2super = self.taxonomy.get("id2super") or self.taxonomy.get("supercategories")

        if id2color is None:
            raise KeyError("Taxonomy missing 'id2color'. Cannot map colors to classes.")
        if id2super is None:
            raise KeyError("Taxonomy missing 'id2super' or 'supercategories'.")

        # Collect unique supercategory names
        unique_supers = sorted(set(id2super.values()))
        super_to_idx = {name: i for i, name in enumerate(unique_supers)}

        # Build final mapping: color → supercategory_index
        color_to_super = {}
        for cid_str, color in id2color.items():
            cid = int(cid_str)
            super_name = id2super.get(str(cid))
            if super_name is None:
                continue
            color_to_super[tuple(color)] = super_to_idx[super_name]

        self.COLOR_TO_CLASS = color_to_super
        self.NUM_CLASSES = len(super_to_idx)

    # --------------------------------------------------------
    # Dataset methods
    # --------------------------------------------------------

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        row = self.entries[idx]
        path = None  # ensure path exists for error printing
        try:
            if self.return_embeddings:
                # strict access, fail early if missing
                if "layout_emb" not in row:
                    raise KeyError(f"'layout_emb' column missing in row {idx}: keys={list(row.keys())}")
                path = str(row["layout_emb"]).strip()
                if not path.endswith(".pt"):
                    raise ValueError(f"Expected .pt file, got {path}")
                layout = load_embedding(path)
            else:
                path = row["layout_path"]
                layout_rgb = load_image(path, self.transform)
                if self.one_hot and self.COLOR_TO_CLASS is not None:
                    layout = self.rgb_to_class_index(layout_rgb)
                else:
                    layout = layout_rgb
        except Exception as e:
            print(f"[Dataset Error] idx={idx}, path={path}, error={type(e).__name__}: {e}", flush=True)
            raise

        return {
            "scene_id": row["scene_id"],
            "room_id": row["room_id"],
            "type": row["type"],
            "is_empty": row["is_empty"],
            "path": path,
            "layout": layout,
        }

    def rgb_to_class_index(self, tensor_img):
        """Convert RGB tensor (3,H,W) ∈ [0,1] to (H,W) long indices."""
        img = (tensor_img.permute(1, 2, 0) * 255).byte()
        h, w, _ = img.shape
        class_map = torch.zeros((h, w), dtype=torch.long)
        for color, idx in self.COLOR_TO_CLASS.items():
            mask = (img == torch.tensor(color, dtype=torch.uint8)).all(dim=-1)
            class_map[mask] = idx
        return class_map

# ---------- POV Dataset ----------

class PovDataset(Dataset):
    def __init__(self, manifest_path, transform=None, pov_type="seg", skip_empty=True, return_embeddings=False):
        self.df = pd.read_csv(manifest_path)
        self.transform = transform
        self.pov_type = pov_type
        self.skip_empty = skip_empty
        self.return_embeddings = return_embeddings

        self.df = self.df[self.df["type"] == pov_type]

        if self.skip_empty:
            self.df = self.df[self.df["is_empty"] == 0]

        self.entries = self.df.to_dict("records")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        row = self.entries[idx]
        sample = {
            "scene_id": row["scene_id"],
            "room_id": row["room_id"],
            "type": row["type"],
            "is_empty": row["is_empty"],
            "path": row["pov_path"]
        }

        if not row["is_empty"]:
            if self.return_embeddings:
                sample["pov"] = load_embedding(row["pov_path"])
            else:
                sample["pov"] = load_image(row["pov_path"], self.transform)
        else:
            sample["pov"] = None

        return sample

# ---------- Graph Dataset ----------

class GraphDataset(Dataset):
    def __init__(self, manifest_path, return_embeddings=False):
        self.df = pd.read_csv(manifest_path)
        self.return_embeddings = return_embeddings
        self.entries = self.df.to_dict("records")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        row = self.entries[idx]
        sample = {
            "scene_id": row["scene_id"],
            "room_id": row["room_id"],
            "type": row["type"],
            "is_empty": row["is_empty"],
            "path": row["graph_path"]  # assuming your manifest has this
        }

        if not row["is_empty"]:
            if self.return_embeddings:
                sample["graph"] = load_embedding(row["graph_path"])
            else:
                # load raw graph here (JSON, adjacency, etc.)
                with open(row["graph_path"], "r") as f:
                    sample["graph"] = f.read()
        else:
            sample["graph"] = None

        return sample

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

# ---------- Helper Functions ----------

def make_dataloaders(layout_manifest, pov_manifest, graph_manifest, batch_size=32, transform=None):
    layout_ds = LayoutDataset(layout_manifest, transform=transform, mode="all")
    pov_ds = PovDataset(pov_manifest, transform=transform, pov_type="seg")
    graph_ds = GraphDataset(graph_manifest)

    layout_loader = DataLoader(layout_ds, batch_size=batch_size, shuffle=True)
    pov_loader = DataLoader(pov_ds, batch_size=batch_size, shuffle=True)
    graph_loader = DataLoader(graph_ds, batch_size=batch_size, shuffle=True)

    return layout_loader, pov_loader, graph_loader

def main():
    import os
    import torchvision.transforms as T

    layout_manifest = r"C:\Users\Hagai.LAPTOP-QAG9263N\Desktop\Thesis\repositories\ImagiNav\indexes\layouts.csv"
    pov_manifest = r"C:\Users\Hagai.LAPTOP-QAG9263N\Desktop\Thesis\repositories\ImagiNav\indexes\povs.csv"
    graph_manifest = "path/to/graph_manifest.csv"  # not required yet

    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor()
    ])

    # --- Layout Dataset ---
    print("Testing LayoutDataset...")
    layout_ds = LayoutDataset(layout_manifest, transform=transform, mode="all")
    print(f"Loaded {len(layout_ds)} layout samples")
    sample = layout_ds[0]
    for k, v in sample.items():
        if hasattr(v, "shape"):
            print(k, v.shape)
        else:
            print(k, v)

    # --- POV Dataset (seg) ---
    print("\nTesting PovDataset (seg)...")
    pov_ds = PovDataset(pov_manifest, transform=transform, pov_type="seg")
    print(f"Loaded {len(pov_ds)} pov samples (seg)")
    sample = pov_ds[0]
    for k, v in sample.items():
        if hasattr(v, "shape"):
            print(k, v.shape)
        else:
            print(k, v)

    # --- Graph Dataset (optional) ---
    if os.path.exists(graph_manifest):
        print("\nTesting GraphDataset...")
        graph_ds = GraphDataset(graph_manifest, return_embeddings=False)
        print(f"Loaded {len(graph_ds)} graph samples")
        sample = graph_ds[0]
        for k, v in sample.items():
            if isinstance(v, str) and len(v) > 60:
                print(k, v[:60] + "...")
            else:
                print(k, v)
    else:
        print("\nGraph manifest not found, skipping GraphDataset test.")

    # --- Dataloaders (layout + pov only) ---
    print("\nTesting Dataloaders...")
    layout_loader = DataLoader(layout_ds, batch_size=4, shuffle=True)
    pov_loader = DataLoader(pov_ds, batch_size=4, shuffle=True)

    batch = next(iter(layout_loader))
    print("Layout batch keys:", batch.keys())
    if "layout" in batch:
        print("Layout batch tensor shape:", batch["layout"].shape)

if __name__ == "__main__":
    main()
