import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import numpy as np

# ---------- Base Utilities ----------

def load_image(path, transform=None):
    img = Image.open(path).convert("RGB")
    if transform:
        img = transform(img)
    return img

def load_embedding(path):
    return torch.load(path) if path.endswith(".pt") else torch.from_numpy(np.load(path))



# ---------- Layout Dataset ----------

class LayoutDataset(Dataset):
    def __init__(self, manifest_path, transform=None, mode="all", skip_empty=True, return_embeddings=False):
        self.df = pd.read_csv(manifest_path)
        self.transform = transform
        self.mode = mode
        self.skip_empty = skip_empty
        self.return_embeddings = return_embeddings

        if self.mode != "all":
            self.df = self.df[self.df["type"] == self.mode]

        if self.skip_empty:
            self.df = self.df[self.df["is_empty"] == False]

        self.entries = self.df.to_dict("records")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        row = self.entries[idx]

        try:
            if self.return_embeddings:
                # Use embedding_path if available, otherwise fall back to layout_path
                path = row.get("embedding_path", row["layout_path"])
                layout = load_embedding(path)
            else:
                path = row["layout_path"]
                layout = load_image(path, self.transform)
        except Exception:
            # skip only broken or unreadable files
            return None

        return {
            "scene_id": row["scene_id"],
            "room_id": row["room_id"],
            "type": row["type"],
            "is_empty": row["is_empty"],
            "path": path,
            "layout": layout,
        }

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


class CombinedMultiModalDataset(Dataset):
    """
    Combines scene-level and room-level samples.
    - Scene samples: layout + graph (no POV)
    - Room samples: layout + graph + POV
    """
    def __init__(self, pov_manifest, layout_manifest, graph_manifest,
                 transform=None, pov_type="seg", skip_empty=True,
                 return_embeddings=False):
        
        self.transform = transform
        self.return_embeddings = return_embeddings
        
        # Load all manifests
        pov_df = pd.read_csv(pov_manifest)
        layout_df = pd.read_csv(layout_manifest)
        graph_df = pd.read_csv(graph_manifest)
        
        # Filter
        pov_df = pov_df[pov_df["type"] == pov_type]
        if skip_empty:
            pov_df = pov_df[pov_df["is_empty"] == 0]
            layout_df = layout_df[layout_df["is_empty"] == False]
            graph_df = graph_df[graph_df["is_empty"] == False]
        
        # Build lookups: key -> row dict
        self.layout_lookup = {}
        for _, row in layout_df.iterrows():
            if row["type"] == "scene":
                key = ("scene", row["scene_id"], None)
            else:  # room
                key = ("room", row["scene_id"], row["room_id"])
            self.layout_lookup[key] = row.to_dict()
        
        self.graph_lookup = {}
        for _, row in graph_df.iterrows():
            if row["type"] == "scene":
                key = ("scene", row["scene_id"], None)
            else:  # room
                key = ("room", row["scene_id"], row["room_id"])
            self.graph_lookup[key] = row.to_dict()
        
        # Build entries list
        self.entries = []
        
        # Add scene entries
        for key in self.layout_lookup:
            if key[0] == "scene" and key in self.graph_lookup:
                self.entries.append({
                    "type": "scene",
                    "scene_id": key[1],
                    "room_id": None,
                    "pov_row": None
                })
        
        # Add room entries (POV-based)
        for _, pov_row in pov_df.iterrows():
            key = ("room", pov_row["scene_id"], pov_row["room_id"])
            if key in self.layout_lookup and key in self.graph_lookup:
                self.entries.append({
                    "type": "room",
                    "scene_id": pov_row["scene_id"],
                    "room_id": pov_row["room_id"],
                    "pov_row": pov_row.to_dict()
                })
        
        print(f"Loaded {len(self.entries)} samples:")
        print(f"  - Scenes: {sum(1 for e in self.entries if e['type'] == 'scene')}")
        print(f"  - Rooms: {sum(1 for e in self.entries if e['type'] == 'room')}")
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        
        # Build lookup key
        key = (entry["type"], entry["scene_id"], entry["room_id"])
        
        layout_row = self.layout_lookup[key]
        graph_row = self.graph_lookup[key]
        
        try:
            # Load layout
            if self.return_embeddings:
                layout = load_embedding(layout_row.get("embedding_path", layout_row["layout_path"]))
            else:
                layout = load_image(layout_row["layout_path"], self.transform)
            
            # Load graph
            if self.return_embeddings:
                graph = load_embedding(graph_row["graph_path"])
            else:
                with open(graph_row["graph_path"], "r") as f:
                    graph = f.read()
            
            # Load POV (only for rooms)
            pov = None
            if entry["type"] == "room":
                pov_row = entry["pov_row"]
                if self.return_embeddings:
                    pov = load_embedding(pov_row["pov_path"])
                else:
                    pov = load_image(pov_row["pov_path"], self.transform)
            
            return {
                "type": entry["type"],
                "scene_id": entry["scene_id"],
                "room_id": entry["room_id"],
                "layout": layout,
                "graph": graph,
                "pov": pov,  # None for scenes
            }
        
        except Exception as e:
            print(f"Error loading {entry['type']} sample {idx}: {e}")
            return None



def make_dataloaders(layout_manifest, pov_manifest, graph_manifest, batch_size=32, transform=None):
    layout_ds = LayoutDataset(layout_manifest, transform=transform, mode="all")
    pov_ds = PovDataset(pov_manifest, transform=transform, pov_type="seg")
    graph_ds = GraphDataset(graph_manifest)

    layout_loader = DataLoader(layout_ds, batch_size=batch_size, shuffle=True)
    pov_loader = DataLoader(pov_ds, batch_size=batch_size, shuffle=True)
    graph_loader = DataLoader(graph_ds, batch_size=batch_size, shuffle=True)

    return layout_loader, pov_loader, graph_loader


from torch.utils.data._utils.collate import default_collate

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None and b.get("layout") is not None]
    if not batch:
        return None
    return default_collate(batch)



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
