
import json
import torch
from PIL import Image
from pathlib import Path
from datetime import datetime
from torchvision import transforms as T
from torch.utils.data import Dataset



# Reusable image transform
default_transform = T.Compose([
    T.Resize((1024, 1024)),
    T.ToTensor(),
    T.Normalize([0.5] * 3, [0.5] * 3)
])

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=default_transform):
        self.image_paths = list(Path(image_dir).glob("*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)

class ConditioningDataset(Dataset):
    def __init__(self, image_dir, token_dir, encoder, token_embedder, transform=default_transform):
        self.encoder = encoder.eval()
        self.token_embedder = token_embedder.eval()
        self.transform = transform

        self.pairs = self._match_pairs(image_dir, token_dir)

    def _match_pairs(self, image_dir, token_dir):
        image_dir, token_dir = Path(image_dir), Path(token_dir)
        image_paths = {p.stem.replace("_image", ""): p for p in image_dir.glob("*_image.png")}
        token_paths = {p.stem.replace("_tokens", ""): p for p in token_dir.glob("*_tokens.json")}

        common_ids = sorted(set(image_paths) & set(token_paths))
        return [(image_paths[i], token_paths[i]) for i in common_ids]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, token_path = self.pairs[idx]

        # Load and transform image
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0)  # add batch dim

        # Encode image latent
        with torch.no_grad():
            image_latent = self.encoder(img_tensor).squeeze(0)

        # Load and encode tokens
        with open(token_path, "r") as f:
            token_data = json.load(f)

        with torch.no_grad():
            token_embedding = self.token_embedder(token_data)

        return {
            "image_latent": image_latent,
            "token_embedding": token_embedding
        }