import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import pandas as pd
from pathlib import Path
import numpy as np
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.weighting import (
    compute_weights_from_counts,
    load_weights_from_stats,
    apply_weights_to_dataframe
)
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
    
    def _compute_column_weights(self, weight_column=None, weights_stats_path=None, 
                                use_grouped_weights=False, weighting_method="inverse_frequency",
                                group_rare_classes=False, class_grouping_path=None, 
                                max_weight=None, exclude_extremely_rare=False, 
                                min_samples_threshold=50, preferred_columns=None):

        # Filter out empty rooms using is_empty column BEFORE computing weights
        # (even though dataset should already be filtered, be safe)
        if "is_empty" in self.df.columns:
            original_size = len(self.df)
            # Handle different representations of False (boolean False, string "false", int 0)
            # Convert to string and check, or use boolean comparison
            is_empty_values = self.df["is_empty"]
            # Check if value is False (boolean), "false" (string), 0 (int), or "False" (string)
            mask = (
                (is_empty_values == False) |  # Boolean False
                (is_empty_values == 0) |  # Integer 0
                (is_empty_values.astype(str).str.lower().isin(['false', '0', 'no']))  # String representations
            )
            self.df = self.df[mask].reset_index(drop=True)
            if len(self.df) < original_size:
                print(f"Filtered out {original_size - len(self.df)} empty rooms from weight computation (using is_empty column)")
        
        # Determine which column to use
        if weight_column is None:
            # Try preferred columns if provided
            if preferred_columns:
                for col in preferred_columns:
                    if col in self.df.columns:
                        weight_column = col
                        break
            
            # If still no column, try to auto-detect any suitable column
            if weight_column is None:
                # Exclude path columns and other non-categorical columns
                exclude_cols = {"path", "layout_path", "image_path", "scene_id", "type", "is_empty"}
                for col in self.df.columns:
                    if col not in exclude_cols and self.df[col].notna().sum() > 0:
                        weight_column = col
                        break
            
            if weight_column is None:
                return None
        elif weight_column not in self.df.columns:
            return None
        
        # Get class IDs from the column (treat all columns the same way)
        class_ids = self.df[weight_column].astype(str).values
        
        # Load or compute class grouping
        class_grouping = None
        if group_rare_classes or (weights_stats_path and use_grouped_weights):
            if weights_stats_path and Path(weights_stats_path).exists():
                import json
                with open(weights_stats_path, 'r') as f:
                    grouping_data = json.load(f)
                    class_grouping = grouping_data.get("class_grouping", {})
                    if class_grouping:
                        print(f"Loaded class grouping from {weights_stats_path}")
            elif class_grouping_path and Path(class_grouping_path).exists():
                import json
                with open(class_grouping_path, 'r') as f:
                    grouping_data = json.load(f)
                    class_grouping = grouping_data.get("class_grouping", {})
                    if class_grouping:
                        print(f"Loaded class grouping from {class_grouping_path}")
            
            # Auto-compute grouping if not loaded and group_rare_classes is True
            if not class_grouping and group_rare_classes:
                unique_classes, counts = np.unique(class_ids, return_counts=True)
                threshold_count = np.percentile(counts, 10)
                class_grouping = {}
                for class_id in unique_classes:
                    count = counts[unique_classes == class_id][0]
                    if count < threshold_count:
                        class_grouping[class_id] = "rare"
                    else:
                        class_grouping[class_id] = class_id
                print(f"Auto-computed class grouping (threshold: {threshold_count:.0f} samples)")
        
        # Apply grouping if enabled
        if class_grouping:
            grouped_class_ids = np.array([class_grouping.get(cid, cid) for cid in class_ids])
        else:
            grouped_class_ids = class_ids
        
        # Exclude extremely rare classes if requested
        if exclude_extremely_rare:
            extremely_rare_class_ids = set()
            if weights_stats_path and Path(weights_stats_path).exists():
                import json
                with open(weights_stats_path, 'r') as f:
                    grouping_data = json.load(f)
                    extremely_rare = grouping_data.get("extremely_rare_classes", [])
                    extremely_rare_class_ids = {c["class_id"] for c in extremely_rare}
            elif class_grouping_path and Path(class_grouping_path).exists():
                import json
                with open(class_grouping_path, 'r') as f:
                    grouping_data = json.load(f)
                    extremely_rare = grouping_data.get("extremely_rare_classes", [])
                    extremely_rare_class_ids = {c["class_id"] for c in extremely_rare}
            
            # Filter out extremely rare classes
            if extremely_rare_class_ids:
                mask = np.array([cid not in extremely_rare_class_ids for cid in class_ids])
                if mask.sum() == 0:
                    print("Warning: All classes would be excluded! Disabling exclusion.")
                else:
                    self.df = self.df[mask].reset_index(drop=True)
                    class_ids = class_ids[mask]
                    grouped_class_ids = grouped_class_ids[mask] if class_grouping else class_ids
                    print(f"Excluded {len(extremely_rare_class_ids)} extremely rare classes from training")
        
        # Compute weights
        unique_groups, group_counts = np.unique(grouped_class_ids, return_counts=True)
        counts_dict = {gid: int(count) for gid, count in zip(unique_groups, group_counts)}
        
        # If grouping is enabled, we need to recompute weights on grouped classes
        # (even if stats file exists, because individual weights don't match grouped structure)
        if class_grouping and group_rare_classes:
            # Grouping is active - recompute weights on grouped counts
            print("Recomputing weights on grouped classes (grouping enabled)...")
            weight_map = compute_weights_from_counts(
                counts_dict,
                method=weighting_method,
                max_weight=max_weight,
                min_weight=1.0
            )
        elif weights_stats_path and Path(weights_stats_path).exists():
            # No grouping or grouping from stats - try to load from stats file
            try:
                weight_map = load_weights_from_stats(weights_stats_path, use_grouped=use_grouped_weights)
                print(f"Loaded {len(weight_map)} weights from {weights_stats_path}")
                # Ensure all groups have weights (fill missing with default)
                for gid in unique_groups:
                    if gid not in weight_map:
                        weight_map[gid] = 1.0
            except Exception as e:
                print(f"Warning: Failed to load weights from stats file: {e}")
                print("Computing weights on-the-fly...")
                weight_map = compute_weights_from_counts(
                    counts_dict,
                    method=weighting_method,
                    max_weight=max_weight,
                    min_weight=1.0
                )
        else:
            # Compute weights on-the-fly
            weight_map = compute_weights_from_counts(
                counts_dict,
                method=weighting_method,
                max_weight=max_weight,
                min_weight=1.0
            )
        
        # Cap weights if requested (additional capping beyond what's in stats)
        if max_weight is not None:
            weight_map = {gid: min(w, max_weight) for gid, w in weight_map.items()}
            if any(w == max_weight for w in weight_map.values()):
                print(f"Capped weights at {max_weight} to prevent over-sampling")
        
        # Create weight tensor
        weights = np.array([weight_map.get(gid, 1.0) for gid in grouped_class_ids], dtype=np.float32)
        
        return torch.from_numpy(weights), class_grouping, weight_map
    
    def make_dataloader(self, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, 
                       use_weighted_sampling=False, weight_column=None, weights_stats_path=None,
                       use_grouped_weights=False, weighting_method="inverse_frequency",
                       group_rare_classes=False, class_grouping_path=None,
                       max_weight=None, exclude_extremely_rare=False, min_samples_threshold=50,
                       preferred_weight_columns=None):

        if use_weighted_sampling:
            result = self._compute_column_weights(
                weight_column=weight_column,
                weights_stats_path=weights_stats_path,
                use_grouped_weights=use_grouped_weights,
                weighting_method=weighting_method,
                group_rare_classes=group_rare_classes,
                class_grouping_path=class_grouping_path,
                max_weight=max_weight,
                exclude_extremely_rare=exclude_extremely_rare,
                min_samples_threshold=min_samples_threshold,
                preferred_columns=preferred_weight_columns
            )
            if result is None:
                print(f"Warning: Weight column not found, falling back to regular sampling")
                use_weighted_sampling = False
            else:
                weights, class_grouping, weight_map = result
                
                # Print weight information using the capped weight_map
                # Get the actual column used for weighting (recompute to get the actual selected column)
                actual_column = weight_column
                if actual_column is None:
                    # Recompute which column was actually selected (same logic as _compute_column_weights)
                    if preferred_weight_columns:
                        for col in preferred_weight_columns:
                            if col in self.df.columns:
                                actual_column = col
                                break
                    if actual_column is None:
                        # Find any suitable column
                        exclude_cols = {"path", "layout_path", "image_path", "scene_id", "type", "is_empty"}
                        for col in self.df.columns:
                            if col not in exclude_cols and self.df[col].notna().sum() > 0:
                                actual_column = col
                                break
                
                if actual_column:
                    # Get current class IDs (after grouping if applied)
                    if class_grouping:
                        # Show grouped class weights
                        class_ids = self.df[actual_column].astype(str).values
                        grouped_class_ids = [class_grouping.get(cid, cid) for cid in class_ids]
                        
                        unique_groups, group_counts = np.unique(grouped_class_ids, return_counts=True)
                        count_map = {gid: int(count) for gid, count in zip(unique_groups, group_counts)}
                        
                        print(f"Class sampling weights (column: {actual_column}, with rare class grouping):")
                        for gid in sorted(weight_map.keys()):
                            weight = weight_map[gid]
                            count = count_map.get(gid, 0)
                            
                            if gid == "rare":
                                rare_classes = [k for k, v in class_grouping.items() if v == "rare"]
                                print(f"  {gid:30s}: weight={weight:.2f}, count={count:6d} (includes {len(rare_classes)} rare classes)")
                            else:
                                print(f"  {gid:30s}: weight={weight:.2f}, count={count:6d}")
                    else:
                        # Show individual class weights
                        class_ids = self.df[actual_column].astype(str).values
                        unique_classes, counts = np.unique(class_ids, return_counts=True)
                        count_map = {cid: int(count) for cid, count in zip(unique_classes, counts)}
                        
                        print(f"Class sampling weights (column: {actual_column}):")
                        for cid in sorted(weight_map.keys()):
                            weight = weight_map[cid]
                            count = count_map.get(cid, 0)
                            print(f"  {cid:30s}: weight={weight:.2f}, count={count:6d}")
        
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
