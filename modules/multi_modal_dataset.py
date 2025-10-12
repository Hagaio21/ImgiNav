import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import numpy as np
from .datasets import load_embedding, load_image

class MultiModalDataset(Dataset):
    """
    Returns triplets: (layout, graph, pov)
    - Scene samples: layout + graph + zero tensor (no POV)
    - Room samples: layout + graph + POV (one sample per POV)
    """
    def __init__(self, layout_manifest, graph_manifest, pov_manifest,
                 transform=None, pov_type="seg", skip_empty=True,
                 return_embeddings=False, pov_shape=(3, 128, 128)):
        
        self.transform = transform
        self.return_embeddings = return_embeddings
        self.pov_shape = pov_shape  # default shape for zero tensor
        
        # Load manifests
        layout_df = pd.read_csv(layout_manifest)
        graph_df = pd.read_csv(graph_manifest)
        pov_df = pd.read_csv(pov_manifest)
        layout_df["scene_id"] = layout_df["scene_id"].astype(str)
        layout_df["room_id"] = layout_df["room_id"].astype(str)
        graph_df["scene_id"] = graph_df["scene_id"].astype(str)
        graph_df["room_id"] = graph_df["room_id"].astype(str)
        pov_df["scene_id"] = pov_df["scene_id"].astype(str)
        pov_df["room_id"] = pov_df["room_id"].astype(str)

        
        # Filter
        pov_df = pov_df[pov_df["type"] == pov_type]
        if skip_empty:
            layout_df = layout_df[layout_df["is_empty"] == False]
            graph_df = graph_df[graph_df["is_empty"] == False]
            pov_df = pov_df[pov_df["is_empty"] == 0]
        
        # Create lookups: (type, scene_id, room_id) -> row
        self.layout_lookup = self._build_lookup(layout_df)
        self.graph_lookup = self._build_lookup(graph_df)
        
        # Build samples list
        self.samples = []
        
        # Add scene samples (zero tensor POV)
        scene_layouts = layout_df[layout_df["type"] == "scene"]
        for _, row in scene_layouts.iterrows():
            key = ("scene", row["scene_id"], None)
            if key in self.graph_lookup:
                self.samples.append({
                    "type": "scene",
                    "scene_id": row["scene_id"],
                    "room_id": None,
                    "pov_path": None
                })
        
        # Add room samples (one per POV)
        for _, pov_row in pov_df.iterrows():
            key = ("room", pov_row["scene_id"], pov_row["room_id"])
            if key in self.layout_lookup and key in self.graph_lookup:
                self.samples.append({
                    "type": "room",
                    "scene_id": pov_row["scene_id"],
                    "room_id": pov_row["room_id"],
                    "pov_path": pov_row["pov_path"]
                })
        
        print(f"Total samples: {len(self.samples)}")
        print(f"  Scenes: {sum(1 for s in self.samples if s['type'] == 'scene')}")
        print(f"  Rooms: {sum(1 for s in self.samples if s['type'] == 'room')}")
    
    def _build_lookup(self, df):
        """Build (type, scene_id, room_id) -> row dict"""
        lookup = {}
        for _, row in df.iterrows():
            if row["type"] == "scene":
                key = ("scene", row["scene_id"], None)
            else:
                key = ("room", row["scene_id"], row["room_id"])
            lookup[key] = row.to_dict()
        return lookup
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        key = (sample["type"], sample["scene_id"], sample["room_id"])
        
        try:
            # Load layout
            layout_row = self.layout_lookup[key]
            if self.return_embeddings:
                layout = load_embedding(layout_row.get("embedding_path", layout_row["layout_path"]))
            else:
                layout = load_image(layout_row["layout_path"], self.transform)
            
            # Load graph
            graph_row = self.graph_lookup[key]
            if self.return_embeddings:
                graph = load_embedding(graph_row["graph_path"])
            else:
                with open(graph_row["graph_path"], "r") as f:
                    graph = f.read()
            
            # Load POV (zero tensor for scenes)
            if sample["pov_path"]:
                if self.return_embeddings:
                    pov = load_embedding(sample["pov_path"])
                else:
                    pov = load_image(sample["pov_path"], self.transform)
            else:
                # Scene-level: return zero tensor
                pov = torch.zeros(self.pov_shape)
            
            return {
                "type": sample["type"],
                "scene_id": sample["scene_id"],
                "room_id": sample["room_id"],
                "layout": layout,
                "graph": graph,
                "pov": pov
            }
        
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return None
