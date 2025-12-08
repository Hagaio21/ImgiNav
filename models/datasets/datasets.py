import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path
import numpy as np
import sys
import operator
from torchvision import transforms

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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

            self.df = pd.read_csv(manifest, low_memory=False)
            # Strip whitespace from column names (common CSV issue)
            self.df.columns = self.df.columns.str.strip()
            self.manifest_dir = manifest.parent
        transform_cfg = self._init_kwargs.get("transform", None)
        self.transform = self._build_transform(transform_cfg) if transform_cfg else None
        self.return_path = self._init_kwargs.get("return_path", False)

        # two modes
        self.target_key = self._init_kwargs.get("target_key", None)
        self.outputs = self._init_kwargs.get("outputs", None)

        self.path_col = self._init_kwargs.get("path_col", "path")
        self.label_col = self._init_kwargs.get("label_col", None)

        # Filter out rows with NaN values in required columns
        if self.outputs:
            required_cols = list(self.outputs.values())
            if self.label_col:
                required_cols.append(self.label_col)
            required_cols = [col.strip() if isinstance(col, str) else col for col in required_cols]
            existing_cols = [col for col in required_cols if col in self.df.columns]
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            
            col_name_mapping = {}
            
            if missing_cols:
                print(f"[WARNING] Required columns not found in manifest: {missing_cols}")
                print(f"[WARNING] Available columns: {list(self.df.columns)}")
                for missing_col in missing_cols[:]:
                    for df_col in self.df.columns:
                        if missing_col.lower().strip() == df_col.lower().strip():
                            print(f"[INFO] Found case/whitespace variation: '{missing_col}' -> '{df_col}'")
                            col_name_mapping[missing_col] = df_col
                            existing_cols.append(df_col)
                            missing_cols.remove(missing_col)
                            break
                
                if existing_cols:
                    print(f"[WARNING] Proceeding with existing columns: {existing_cols}")
                else:
                    raise KeyError(f"None of the required columns found in manifest. Required: {required_cols}, Available: {list(self.df.columns)}")
            
            if col_name_mapping:
                self.outputs = {key: col_name_mapping.get(col, col) for key, col in self.outputs.items()}
                if self.label_col and self.label_col in col_name_mapping:
                    self.label_col = col_name_mapping[self.label_col]
            
            if existing_cols:
                initial_len = len(self.df)
                self.df = self.df.dropna(subset=existing_cols)
                dropped_count = initial_len - len(self.df)
                if dropped_count > 0:
                    print(f"[INFO] Dropped {dropped_count} rows with NaN values in required columns: {existing_cols}")

        # optional filters
        filters = self._init_kwargs.get("filters", None)
        if filters:
            initial_len = len(self.df)
            self.df = self._apply_filters(self.df, filters)
            filtered_len = len(self.df)
            if filtered_len == 0:
                print(f"[ERROR] Dataset is empty after applying filters!")
                print(f"  Initial rows: {initial_len}")
                print(f"  Filters applied: {filters}")
            else:
                print(f"[INFO] Applied filters: {initial_len} -> {filtered_len} rows")

    def _apply_filters(self, df, filters: dict):
        """Apply filters to DataFrame."""
        OPERATOR_MAP = {
            "__lt": operator.lt,
            "__gt": operator.gt,
            "__le": operator.le,
            "__ge": operator.ge,
            "__ne": operator.ne,
        }
        
        missing_columns = []
        filters_to_apply = {}
        
        for key, value in filters.items():
            col = key
            for suffix in OPERATOR_MAP.keys():
                if suffix in key:
                    col = key.replace(suffix, "")
                    break
            
            if col not in df.columns:
                missing_columns.append(f"'{col}' (from filter '{key}')")
            else:
                filters_to_apply[key] = value
        
        if missing_columns:
            print(f"[WARNING] Skipping filter(s) for missing column(s): {', '.join(missing_columns)}")
        
        for key, value in filters_to_apply.items():
            op_func = None
            col = key
            for suffix, op in OPERATOR_MAP.items():
                if suffix in key:
                    col = key.replace(suffix, "")
                    op_func = op
                    break
            
            if op_func is not None:
                df = df[op_func(df[col], value)]
            else:
                if isinstance(value, (list, tuple, set)):
                    if len(value) == 0:
                        continue
                    df = df[df[key].isin(value)]
                else:
                    if col == "rejected" and col in df.columns:
                        def to_bool(val):
                            if isinstance(val, bool):
                                return val
                            if isinstance(val, str):
                                return val.lower() in ("true", "1", "yes")
                            return bool(val)
                        df_bool = df[col].apply(to_bool)
                        filter_bool = to_bool(value)
                        df = df[df_bool == filter_bool]
                    else:
                        df = df[df[key] == value]
        
        return df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.outputs:
            sample = {}
            for key, col in self.outputs.items():
                if col not in row.index:
                    raise KeyError(f"Column '{col}' not found in manifest. Available: {list(self.df.columns)}")
                sample[key] = self._load_value(row[col])
            if self.return_path:
                sample["paths"] = {k: str(row[c]) for k, c in self.outputs.items()}
            return sample

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

    def _build_transform(self, transform_cfg):
        """Build transform from config dict or return callable as-is."""
        if callable(transform_cfg):
            return transform_cfg
        
        if isinstance(transform_cfg, dict):
            if transform_cfg.get("type") == "Compose":
                transform_list = []
                for t_cfg in transform_cfg.get("transforms", []):
                    t_type = t_cfg.get("type")
                    if t_type == "Resize":
                        size = t_cfg.get("size")
                        interpolation = t_cfg.get("interpolation", "nearest")
                        if interpolation == "nearest":
                            transform_list.append(transforms.Resize(size, interpolation=0))
                        elif interpolation == "bilinear":
                            transform_list.append(transforms.Resize(size, interpolation=2))
                        else:
                            transform_list.append(transforms.Resize(size))
                    elif t_type == "ToTensor":
                        transform_list.append(transforms.ToTensor())
                    elif t_type == "Normalize":
                        mean = t_cfg.get("mean", [0.5, 0.5, 0.5])
                        std = t_cfg.get("std", [0.5, 0.5, 0.5])
                        transform_list.append(transforms.Normalize(mean=mean, std=std))
                    elif t_type == "CenterCrop":
                        transform_list.append(transforms.CenterCrop(t_cfg.get("size")))
                    elif t_type == "RandomCrop":
                        transform_list.append(transforms.RandomCrop(t_cfg.get("size")))
                    else:
                        raise ValueError(f"Unknown transform type: {t_type}")
                return transforms.Compose(transform_list)
        return None

    def _resolve_path(self, path_value):
        """Resolve a path value relative to the manifest directory."""
        if pd.isna(path_value) or path_value == "":
            return None
        
        path = Path(path_value)
        
        if path.is_absolute() and path.exists():
            return path
        
        resolved = self.manifest_dir / path
        if resolved.exists():
            return resolved
        
        resolved = self.manifest_dir.parent / path
        if resolved.exists():
            return resolved
        
        return path

    def _load_value(self, value):
        """Load a value based on its type."""
        if pd.isna(value) or value == "":
            return None
        
        path = self._resolve_path(value)
        
        if path is None:
            return value
        
        if not path.exists():
            return value
        
        suffix = path.suffix.lower()
        
        if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img
        elif suffix == ".pt":
            return torch.load(path, map_location="cpu", weights_only=True)
        elif suffix == ".json":
            import json
            with open(path, "r") as f:
                return json.load(f)
        elif suffix == ".txt":
            with open(path, "r") as f:
                return f.read()
        
        return str(value)

    def make_dataloader(self, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True):
        """Create a DataLoader."""
        return DataLoader(
            self, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            pin_memory=pin_memory if torch.cuda.is_available() else False,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None, 
            drop_last=False
        )

    def split(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, stratify_column=None):
        """Split dataset into train/val/test sets."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        np.random.seed(seed)
        n = len(self.df)
        indices = np.random.permutation(n)
        
        if stratify_column and stratify_column in self.df.columns:
            from sklearn.model_selection import train_test_split
            train_idx, temp_idx = train_test_split(
                indices, train_size=train_ratio,
                stratify=self.df.iloc[indices][stratify_column], random_state=seed
            )
            val_size = val_ratio / (val_ratio + test_ratio)
            val_idx, test_idx = train_test_split(
                temp_idx, train_size=val_size,
                stratify=self.df.iloc[temp_idx][stratify_column], random_state=seed
            )
        else:
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            train_idx = indices[:train_end]
            val_idx = indices[train_end:val_end]
            test_idx = indices[val_end:]
        
        def make_split_dataset(idx):
            split_df = self.df.iloc[idx].reset_index(drop=True)
            new_kwargs = self._init_kwargs.copy()
            new_kwargs["_df"] = split_df
            new_kwargs["_manifest_dir"] = self.manifest_dir
            return ManifestDataset(**new_kwargs)
        
        return make_split_dataset(train_idx), make_split_dataset(val_idx), make_split_dataset(test_idx)

    def get_column_values(self, column: str) -> list:
        """Get unique values in a column."""
        if column not in self.df.columns:
            raise KeyError(f"Column '{column}' not found")
        return self.df[column].unique().tolist()
    
    def filter_by(self, **kwargs) -> "ManifestDataset":
        """Create a filtered copy of the dataset."""
        filtered_df = self.df.copy()
        for col, value in kwargs.items():
            if col not in filtered_df.columns:
                print(f"[WARNING] Column '{col}' not found, skipping filter")
                continue
            if isinstance(value, (list, tuple)):
                filtered_df = filtered_df[filtered_df[col].isin(value)]
            else:
                filtered_df = filtered_df[filtered_df[col] == value]
        
        filtered_df = filtered_df.reset_index(drop=True)
        new_kwargs = self._init_kwargs.copy()
        new_kwargs["_df"] = filtered_df
        new_kwargs["_manifest_dir"] = self.manifest_dir
        return ManifestDataset(**new_kwargs)
    
    def __repr__(self):
        return f"ManifestDataset(samples={len(self)}, columns={list(self.df.columns)})"