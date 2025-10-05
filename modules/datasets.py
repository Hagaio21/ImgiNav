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

# ---------- Example Dataloaders ----------

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
