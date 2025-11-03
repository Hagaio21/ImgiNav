import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import pandas as pd
from pathlib import Path
import numpy as np
from ..components.base_component import BaseComponent

class ManifestDataset(BaseComponent, Dataset):
    def _build(self):
        # Allow direct DataFrame passing (for split datasets)
        if "_df" in self._init_kwargs:
            self.df = self._init_kwargs["_df"]
            self.manifest_dir = self._init_kwargs.get("_manifest_dir", Path("."))
        else:
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
        label = self._load_value(row[self.label_col]) if self.label_col and self.label_col in row else None

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
        
        # Handle numeric values (like room_id), return as long for consistency
        if isinstance(val, (int, float)):
            return torch.tensor(int(val), dtype=torch.long)
        
        if isinstance(val, str):
            # Handle string labels like "room" or "scene" - return as string for classification head
            val_lower = val.lower().strip()
            if val_lower in ["room", "scene"]:
                # Return as string - will be converted by loss function
                return val
            
            # Handle empty strings
            if not val or val.strip() == "":
                raise ValueError(f"Empty path in manifest")
            # Try to convert string to number if it's numeric (for numeric room_ids as strings)
            if val.replace('.', '').replace('-', '').isdigit():
                # Convert to int (long) for room_id values to maintain dtype consistency
                return torch.tensor(int(float(val)), dtype=torch.long)
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
                    # Convert to numpy array, then to tensor, and normalize to [-1, 1]
                    # This matches tanh output activation (range [-1, 1])
                    # Formula: (pixel / 255.0) * 2.0 - 1.0 = pixel / 127.5 - 1.0
                    # Use contiguous array for faster conversion
                    img_array = np.ascontiguousarray(img, dtype=np.float32)
                    img = torch.from_numpy(img_array)
                    # Convert from HWC to CHW format and normalize to [-1, 1]
                    if img.ndim == 3:
                        img = img.permute(2, 0, 1) / 127.5 - 1.0  # [0, 255] -> [-1, 1]
                    else:
                        raise ValueError(f"Unexpected image shape after conversion: {img.shape}")
                return img
            if ext == ".pt":
                return torch.load(p)
        # Fallback: convert to tensor (will default to appropriate type)
        return torch.tensor(val)

    def split(self, train_split=0.8, random_seed=42):
        """
        Split dataset into train and validation sets.
        
        Args:
            train_split: Fraction of data for training (default: 0.8)
            random_seed: Random seed for reproducibility (default: 42)
            
        Returns:
            (train_dataset, val_dataset) tuple of ManifestDataset instances
        """
        val_split = 1.0 - train_split
        if train_split <= 0 or train_split >= 1.0:
            raise ValueError(f"train_split must be between 0 and 1, got {train_split}")
        
        total_size = len(self)
        train_size = int(train_split * total_size)
        val_size = total_size - train_size
        
        # Set random seed for reproducibility
        generator = torch.Generator()
        generator.manual_seed(random_seed)
        
        # Shuffle indices
        indices = torch.randperm(total_size, generator=generator).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create new datasets with filtered DataFrames
        train_df = self.df.iloc[train_indices].reset_index(drop=True)
        val_df = self.df.iloc[val_indices].reset_index(drop=True)
        
        # Create new dataset instances with filtered data
        train_dataset = ManifestDataset(
            manifest=None,  # Not using manifest path
            outputs=self.outputs,
            filters=None,  # Already filtered
            return_path=self.return_path,
            transform=self.transform,
            target_key=self.target_key,
            label_col=self.label_col,
            path_col=self.path_col,
            _df=train_df,  # Internal: pass DataFrame directly
            _manifest_dir=self.manifest_dir
        )
        
        val_dataset = ManifestDataset(
            manifest=None,
            outputs=self.outputs,
            filters=None,
            return_path=self.return_path,
            transform=self.transform,
            target_key=self.target_key,
            label_col=self.label_col,
            path_col=self.path_col,
            _df=val_df,
            _manifest_dir=self.manifest_dir
        )
        
        return train_dataset, val_dataset
    
    def _compute_room_id_weights(self):
        """Compute inverse frequency weights for room_id balancing."""
        if "room_id" not in self.df.columns:
            return None
        # Convert to string to handle mixed types (int/str)
        room_ids = self.df["room_id"].astype(str).values
        unique_rooms, counts = np.unique(room_ids, return_counts=True)
        max_count = counts.max()
        weight_map = {rid: max_count / count for rid, count in zip(unique_rooms, counts)}
        weights = np.array([weight_map[rid] for rid in room_ids], dtype=np.float32)
        return torch.from_numpy(weights)
    
    def make_dataloader(self, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, use_weighted_sampling=False):
        if use_weighted_sampling:
            weights = self._compute_room_id_weights()
            if weights is None:
                print("Warning: room_id column not found, falling back to regular sampling")
                use_weighted_sampling = False
            else:
                # Convert to string to handle mixed types (int/str)
                room_ids = self.df["room_id"].astype(str).values
                unique_rooms, counts = np.unique(room_ids, return_counts=True)
                max_count = counts.max()
                print(f"Room ID sampling weights:")
                for rid, count in zip(unique_rooms, counts):
                    weight = max_count / count
                    print(f"  {rid}: weight={weight:.2f}, count={count}")
        
        if use_weighted_sampling:
            sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
            return DataLoader(
                self, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                pin_memory=pin_memory if torch.cuda.is_available() else False,
                persistent_workers=persistent_workers if num_workers > 0 else False,
                prefetch_factor=2 if num_workers > 0 else None, drop_last=False
            )
        else:
            return DataLoader(
                self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                pin_memory=pin_memory if torch.cuda.is_available() else False,
                persistent_workers=persistent_workers if num_workers > 0 else False,
                prefetch_factor=2 if num_workers > 0 else None, drop_last=False
            )
