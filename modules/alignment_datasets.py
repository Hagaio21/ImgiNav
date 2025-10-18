import torch
from torch.utils.data import Dataset
import pandas as pd
import torch.nn.functional as F


class POVEmbeddingDataset(Dataset):
    """
    Dataset for aligning POV embeddings with layout embeddings.
    Uses only the room manifest, since scenes have no POVs.
    """
    def __init__(self, room_manifest_path, subsample=None):
        df = pd.read_csv(room_manifest_path)
        if subsample and subsample < len(df):
            df = df.sample(n=subsample, random_state=42)
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pov = torch.load(row["POV_EMBEDDING_PATH"]).float().squeeze()
        layout = torch.load(row["ROOM_LAYOUT_EMBEDDING_PATH"]).float().squeeze()

        # normalize only the condition
        pov = F.normalize(pov, dim=-1)

        return {"pov": pov, "layout": layout}


class GraphEmbeddingDataset(Dataset):
    """
    Dataset for aligning graph embeddings with layout embeddings.
    Can take one or more manifests (room + scene).
    """
    def __init__(self, manifest_paths, subsample=None):
        if isinstance(manifest_paths, str):
            manifest_paths = [manifest_paths]

        frames = []
        for path in manifest_paths:
            df = pd.read_csv(path)
            df["source"] = path
            frames.append(df)

        df_all = pd.concat(frames, ignore_index=True)
        if subsample and subsample < len(df_all):
            df_all = df_all.sample(n=subsample, random_state=42)
        self.df = df_all.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if "ROOM_GRAPH_EMBEDDING_PATH" in row and pd.notna(row["ROOM_GRAPH_EMBEDDING_PATH"]):
            graph = torch.load(row["ROOM_GRAPH_EMBEDDING_PATH"]).float().squeeze()
            layout = torch.load(row["ROOM_LAYOUT_EMBEDDING_PATH"]).float().squeeze()
        elif "SCENE_GRAPH_EMBEDDING_PATH" in row and pd.notna(row["SCENE_GRAPH_EMBEDDING_PATH"]):
            graph = torch.load(row["SCENE_GRAPH_EMBEDDING_PATH"]).float().squeeze()
            layout = torch.load(row["SCENE_LAYOUT_EMBEDDING_PATH"]).float().squeeze()
        else:
            raise ValueError(f"Row {idx} has no valid graph-layout pair")

        # normalize only the condition
        graph = F.normalize(graph, dim=-1)

        return {"graph": graph, "layout": layout}

