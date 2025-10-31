import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path
import numpy as np
from ..components.base_component import BaseComponent

class ManifestDataset(BaseComponent, Dataset):
    def _build(self):
        manifest = Path(self._init_kwargs["manifest"])
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest}")

        self.df = pd.read_csv(manifest)
        self.manifest_dir = manifest.parent  # Store manifest directory for relative path resolution
        self.transform = self._init_kwargs.get("transform", None)
        self.return_path = self._init_kwargs.get("return_path", False)

        # two modes
        self.target_key = self._init_kwargs.get("target_key", None)
        self.outputs = self._init_kwargs.get("outputs", None)

        self.path_col = self._init_kwargs.get("path_col", "path")
        self.label_col = self._init_kwargs.get("label_col", None)

        # Filter out rows with NaN values in required columns before applying filters
        if self.outputs:
            required_cols = list(self.outputs.values())
            if self.label_col:
                required_cols.append(self.label_col)
            # Drop rows where any required column has NaN
            self.df = self.df.dropna(subset=required_cols)

        # optional filters
        filters = self._init_kwargs.get("filters", None)
        if filters:
            self.df = self._apply_filters(self.df, filters)

    # ------------------------
    # Filtering
    # ------------------------
    def _apply_filters(self, df, filters: dict):
        for key, value in filters.items():
            if "__lt" in key:
                col = key.replace("__lt", "")
                df = df[df[col] < value]
            elif "__gt" in key:
                col = key.replace("__gt", "")
                df = df[df[col] > value]
            elif "__le" in key:
                col = key.replace("__le", "")
                df = df[df[col] <= value]
            elif "__ge" in key:
                col = key.replace("__ge", "")
                df = df[df[col] >= value]
            elif "__ne" in key:
                col = key.replace("__ne", "")
                df = df[df[col] != value]
            else:
                if isinstance(value, (list, tuple, set)):
                    df = df[df[key].isin(value)]
                else:
                    df = df[df[key] == value]
        return df.reset_index(drop=True)

    # ------------------------
    # Dataset core
    # ------------------------
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # multi-output mode
        if self.outputs:
            sample = {}
            for key, col in self.outputs.items():
                sample[key] = self._load_value(row[col])
            if self.return_path:
                sample["paths"] = {k: str(row[c]) for k, c in self.outputs.items()}
            return sample

        # simple mode
        sample_path = Path(row[self.path_col])
        data = self._load_value(sample_path)
        label = torch.tensor(row[self.label_col]) if self.label_col and self.label_col in row else None

        sample = {"data": data}
        if label is not None:
            sample["label"] = label
        if self.target_key:
            sample[self.target_key] = data
        if self.return_path:
            sample["path"] = str(sample_path)
        return sample

    # ------------------------
    # Helpers
    # ------------------------
    def _load_value(self, val):
        """Auto-load image, tensor, or numeric."""
        # Handle NaN values from pandas
        if pd.isna(val):
            raise ValueError(f"NaN value in manifest")
        
        # Handle string "scene" for room_id column (maps to room_id 0 -> class index 0)
        if isinstance(val, str) and val.lower() == "scene":
            return torch.tensor(0)
        
        # Handle numeric values (like room_id), return as-is
        if isinstance(val, (int, float)):
            return torch.tensor(val)
        
        if isinstance(val, str):
            # Handle empty strings
            if not val or val.strip() == "":
                raise ValueError(f"Empty path in manifest")
            # Try to convert string to number if it's numeric (for numeric room_ids as strings)
            if val.replace('.', '').replace('-', '').isdigit():
                return torch.tensor(float(val))
            # Otherwise treat as file path
            p = Path(val)
            # Resolve relative paths relative to manifest directory
            if not p.is_absolute():
                p = self.manifest_dir / p
            if not p.exists():
                raise FileNotFoundError(f"File not found: {p}")
            ext = p.suffix.lower()
            if ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                img = Image.open(p).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                else:
                    # Default: convert PIL Image to tensor
                    # Convert to numpy array, then to tensor, and normalize to [0, 1]
                    # Use contiguous array for faster conversion
                    img_array = np.ascontiguousarray(img, dtype=np.float32)
                    img = torch.from_numpy(img_array)
                    # Convert from HWC to CHW format
                    if img.ndim == 3:
                        img = img.permute(2, 0, 1) / 255.0
                    else:
                        raise ValueError(f"Unexpected image shape after conversion: {img.shape}")
                return img
            if ext == ".pt":
                return torch.load(p)
        return torch.tensor(val)

    def make_dataloader(self, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True):
        return DataLoader(
            self, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            pin_memory=pin_memory if torch.cuda.is_available() else False,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )
