import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import pandas as pd
from pathlib import Path
import numpy as np
import sys
import operator
from torchvision import transforms

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

            self.df = pd.read_csv(manifest, low_memory=False)
            # Strip whitespace from column names (common CSV issue)
            self.df.columns = self.df.columns.str.strip()
            self.manifest_dir = manifest.parent  # Store manifest directory for relative path resolution
        transform_cfg = self._init_kwargs.get("transform", None)
        # Build transform from config if provided, otherwise use as-is (for callable transforms)
        self.transform = self._build_transform(transform_cfg) if transform_cfg else None
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
            # Strip whitespace from required column names for matching
            required_cols = [col.strip() if isinstance(col, str) else col for col in required_cols]
            # Filter to only columns that exist in the DataFrame
            existing_cols = [col for col in required_cols if col in self.df.columns]
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            
            # Column name mapping for case/whitespace variations
            col_name_mapping = {}
            
            if missing_cols:
                print(f"[WARNING] Required columns not found in manifest: {missing_cols}")
                print(f"[WARNING] Available columns: {list(self.df.columns)}")
                # Check for case-insensitive or whitespace variations
                for missing_col in missing_cols[:]:  # Copy list to modify during iteration
                    for df_col in self.df.columns:
                        if missing_col.lower().strip() == df_col.lower().strip():
                            print(f"[INFO] Found case/whitespace variation: '{missing_col}' -> '{df_col}'")
                            col_name_mapping[missing_col] = df_col
                            existing_cols.append(df_col)
                            missing_cols.remove(missing_col)
                            break
                
                # Raise error if critical columns are still missing
                if existing_cols:
                    print(f"[WARNING] Proceeding with existing columns: {existing_cols}")
                else:
                    raise KeyError(f"None of the required columns found in manifest. Required: {required_cols}, Available: {list(self.df.columns)}")
            
            # Update outputs mapping to use corrected column names
            if col_name_mapping:
                self.outputs = {key: col_name_mapping.get(col, col) for key, col in self.outputs.items()}
                if self.label_col and self.label_col in col_name_mapping:
                    self.label_col = col_name_mapping[self.label_col]
            
            # Drop rows where any required column has NaN (only for existing columns)
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
                print(f"  Available columns: {list(self.df.columns) if len(self.df) > 0 else 'N/A (empty)'}")
                if initial_len > 0:
                    # Show sample of what's in the original data
                    original_df = pd.read_csv(self._init_kwargs.get("manifest", ""), low_memory=False) if "_df" not in self._init_kwargs else self._init_kwargs.get("_df", pd.DataFrame())
                    if len(original_df) > 0:
                        print(f"  Sample of original data:")
                        for key, value in filters.items():
                            if key in original_df.columns:
                                unique_vals = original_df[key].unique()[:10]
                                print(f"    {key}: {unique_vals.tolist()}")
            else:
                print(f"[INFO] Applied filters: {initial_len} -> {filtered_len} rows")

    # ------------------------
    # Filtering
    # ------------------------
    def _apply_filters(self, df, filters: dict):
        """
        Apply filters to DataFrame using operator mapping for cleaner code.
        
        Skips filters for columns that don't exist in the DataFrame (with a warning).
        """
        # Operator mapping for filter suffixes
        OPERATOR_MAP = {
            "__lt": operator.lt,
            "__gt": operator.gt,
            "__le": operator.le,
            "__ge": operator.ge,
            "__ne": operator.ne,
        }
        
        # Track missing columns to warn about
        missing_columns = []
        filters_to_apply = {}
        
        # Validate filter columns and collect valid filters
        for key, value in filters.items():
            # Extract column name (remove operator suffix if present)
            col = key
            for suffix in OPERATOR_MAP.keys():
                if suffix in key:
                    col = key.replace(suffix, "")
                    break
            
            if col not in df.columns:
                missing_columns.append(f"'{col}' (from filter '{key}')")
            else:
                filters_to_apply[key] = value
        
        # Warn about missing columns but continue
        if missing_columns:
            print(f"[WARNING] Skipping filter(s) for missing column(s): {', '.join(missing_columns)}")
            print(f"[WARNING] Available columns: {list(df.columns)}")
        
        # Apply only valid filters
        for key, value in filters_to_apply.items():
            # Check for operator suffix
            op_func = None
            col = key
            for suffix, op in OPERATOR_MAP.items():
                if suffix in key:
                    col = key.replace(suffix, "")
                    op_func = op
                    break
            
            if op_func is not None:
                # Use operator function for comparison
                df = df[op_func(df[col], value)]
            else:
                # Equality or membership check
                if isinstance(value, (list, tuple, set)):
                    # Empty list means "no filter" - include all values
                    if len(value) == 0:
                        print(f"[INFO] Filter '{key}' has empty list - skipping (include all)")
                        continue
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
                if col not in row.index:
                    raise KeyError(
                        f"Column '{col}' (required for output '{key}') not found in manifest. "
                        f"Available columns: {list(self.df.columns)}"
                    )
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
    def _build_transform(self, transform_cfg):
        """
        Build transform from config dict or return callable as-is.
        
        Args:
            transform_cfg: Either a dict with 'type' and 'transforms' keys, or a callable
            
        Returns:
            Transform callable
        """
        # If it's already a callable, return as-is
        if callable(transform_cfg):
            return transform_cfg
        
        # If it's a dict, build transform from config
        if isinstance(transform_cfg, dict):
            if transform_cfg.get("type") == "Compose":
                # Build Compose transform from list of transforms
                transform_list = []
                for t_cfg in transform_cfg.get("transforms", []):
                    t_type = t_cfg.get("type")
                    if t_type == "Resize":
                        size = t_cfg.get("size")
                        # Use nearest-neighbor interpolation to preserve exact RGB values
                        # This is critical for segmentation-based losses that require exact color matching
                        interpolation = t_cfg.get("interpolation", "nearest")
                        if interpolation == "nearest":
                            # PIL.Image.NEAREST = 0
                            transform_list.append(transforms.Resize(size, interpolation=0))
                        elif interpolation == "bilinear":
                            # PIL.Image.BILINEAR = 2
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
                        size = t_cfg.get("size")
                        transform_list.append(transforms.CenterCrop(size))
                    elif t_type == "RandomCrop":
                        size = t_cfg.get("size")
                        transform_list.append(transforms.RandomCrop(size))
                    else:
                        raise ValueError(f"Unknown transform type: {t_type}")
                return transforms.Compose(transform_list)
        
        return None

    def _resolve_path(self, path_value):
        """
        Resolve a path value relative to the manifest directory.
        
        Handles both absolute paths and paths relative to the manifest.
        """
        if pd.isna(path_value) or path_value == "":
            return None
        
        path = Path(path_value)
        
        # If absolute path exists, use it
        if path.is_absolute() and path.exists():
            return path
        
        # Try relative to manifest directory
        resolved = self.manifest_dir / path
        if resolved.exists():
            return resolved
        
        # Try parent of manifest directory (dataset root)
        resolved = self.manifest_dir.parent / path
        if resolved.exists():
            return resolved
        
        # Return original path (may fail later during loading)
        return path

    def _load_value(self, value):
        """
        Load a value based on its type:
        - If it's a path to an image file, load and transform it
        - If it's a path to a .pt file, load tensor
        - If it's a path to a .json file, load JSON
        - Otherwise, return as-is
        """
        if pd.isna(value) or value == "":
            return None
        
        # Try to resolve as path
        path = self._resolve_path(value)
        
        if path is None:
            return value
        
        if not path.exists():
            # Return the raw value if path doesn't exist
            return value
        
        suffix = path.suffix.lower()
        
        # Image files
        if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img
        
        # Tensor files
        elif suffix == ".pt":
            return torch.load(path, map_location="cpu", weights_only=True)
        
        # JSON files
        elif suffix == ".json":
            import json
            with open(path, "r") as f:
                return json.load(f)
        
        # Text files
        elif suffix == ".txt":
            with open(path, "r") as f:
                return f.read()
        
        # Default: return as string
        return str(value)

    # ------------------------
    # Pre-computed weight support
    # ------------------------
    def _use_precomputed_weights(self, weight_column: str = "sample_weight", max_weight: float = None, 
                                 non_empty_multiplier: float = None) -> torch.Tensor:
        """
        Use pre-computed weights from a column in the manifest.
        
        Args:
            weight_column: Name of column containing pre-computed weights
            max_weight: Optional cap on maximum weight
            non_empty_multiplier: Optional multiplier to apply to non-empty rooms (e.g., 2.0 doubles their weight)
            
        Returns:
            Tensor of weights for each sample
        """
        if weight_column not in self.df.columns:
            raise KeyError(f"Weight column '{weight_column}' not found. Available: {list(self.df.columns)}")
        
        # Check if dataset is empty
        if len(self.df) == 0:
            raise ValueError(f"Dataset is empty after filtering. Cannot create weights for zero samples.")
        
        weights = self.df[weight_column].values.astype(np.float32)
        
        # Handle NaN values
        nan_mask = np.isnan(weights)
        if nan_mask.any():
            print(f"[WARNING] {nan_mask.sum()} NaN values in weight column, replacing with 1.0")
            weights[nan_mask] = 1.0
        
        # Apply multiplier for non-empty rooms if requested
        if non_empty_multiplier is not None and non_empty_multiplier != 1.0:
            if "is_empty" in self.df.columns:
                non_empty_mask = ~self.df["is_empty"].fillna(True).astype(bool)
                non_empty_count = non_empty_mask.sum()
                if non_empty_count > 0:
                    weights[non_empty_mask] *= non_empty_multiplier
                    print(f"[INFO] Applied {non_empty_multiplier}x multiplier to {non_empty_count} non-empty room samples")
            else:
                print(f"[WARNING] 'is_empty' column not found, cannot apply non_empty_multiplier")
        
        # Cap weights if requested
        if max_weight is not None:
            capped = (weights > max_weight).sum()
            if capped > 0:
                print(f"[INFO] Capping {capped} weights at {max_weight}")
                weights = np.clip(weights, None, max_weight)
        
        # Check if weights array is empty
        if len(weights) == 0:
            raise ValueError(f"No valid weights found in column '{weight_column}'. Dataset may be empty after filtering.")
        
        # Print weight statistics
        print(f"[INFO] Using pre-computed weights from '{weight_column}':")
        print(f"  min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}, std={weights.std():.4f}")
        
        return torch.from_numpy(weights)

    # ------------------------
    # Weighted sampling statistics
    # ------------------------
    def _print_weight_statistics(self, weights: np.ndarray, group_columns: list = None):
        """Print detailed statistics about sample weights."""
        print(f"\n{'='*60}")
        print("Sample Weight Statistics")
        print(f"{'='*60}")
        print(f"Total samples: {len(weights)}")
        print(f"Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
        print(f"Weight mean: {weights.mean():.4f}, std: {weights.std():.4f}")
        
        if group_columns:
            for col in group_columns:
                if col in self.df.columns:
                    print(f"\nWeights by '{col}':")
                    for value in self.df[col].unique():
                        mask = self.df[col] == value
                        group_weights = weights[mask]
                        effective_samples = group_weights.sum()
                        print(f"  {value}: count={mask.sum()}, "
                              f"mean_weight={group_weights.mean():.4f}, "
                              f"effective_samples={effective_samples:.1f}")
        
        print(f"{'='*60}\n")

    # ------------------------
    # Column-based weighting (existing functionality)
    # ------------------------
    def _compute_column_weights(self, weight_column=None, weights_stats_path=None, 
                                use_grouped_weights=False, weighting_method="inverse_frequency",
                                group_rare_classes=False, class_grouping_path=None,
                                max_weight=None, exclude_extremely_rare=False,
                                min_samples_threshold=50, preferred_columns=None):
        """
        Compute sampling weights based on a categorical column.
        
        This implements inverse frequency weighting to balance class distributions.
        """
        # Find weight column
        if weight_column is None:
            # Try preferred columns first
            if preferred_columns:
                for col in preferred_columns:
                    if col in self.df.columns:
                        weight_column = col
                        break
            
            # Fall back to finding any suitable column
            if weight_column is None:
                exclude_cols = {"path", "layout_path", "image_path", "scene_id", "type", "is_empty",
                               "pov_weight", "type_weight", "empty_weight", "sample_weight"}
                for col in self.df.columns:
                    if col not in exclude_cols and self.df[col].notna().sum() > 0:
                        weight_column = col
                        break
        
        if weight_column is None or weight_column not in self.df.columns:
            return None
        
        print(f"[INFO] Computing weights based on column: {weight_column}")
        
        # Get class IDs
        class_ids = self.df[weight_column].astype(str).values
        unique_classes, counts = np.unique(class_ids, return_counts=True)
        num_classes = len(unique_classes)
        
        print(f"[INFO] Found {num_classes} unique classes in '{weight_column}'")
        
        # Build class grouping if requested
        class_grouping = None
        if class_grouping_path and Path(class_grouping_path).exists():
            import json
            with open(class_grouping_path, 'r') as f:
                grouping_data = json.load(f)
                class_grouping = grouping_data.get("class_grouping", {})
                print(f"[INFO] Loaded class grouping with {len(class_grouping)} mappings")
        elif group_rare_classes:
            # Auto-compute grouping based on percentiles
            count_map = {cid: int(cnt) for cid, cnt in zip(unique_classes, counts)}
            sorted_counts = sorted(count_map.values())
            
            # Create percentile bands
            percentiles = [
                0,
                np.percentile(sorted_counts, 10),
                np.percentile(sorted_counts, 20),
                np.percentile(sorted_counts, 30),
                np.percentile(sorted_counts, 40),
                np.percentile(sorted_counts, 50),
            ]
            band_names = ["rare_0_10", "rare_10_20", "rare_20_30", "rare_30_40", "rare_40_50"]
            
            class_grouping = {}
            band_counts = {name: 0 for name in band_names}
            individual_count = 0
            
            for class_id, count in count_map.items():
                assigned = False
                for i, band_name in enumerate(band_names):
                    if percentiles[i] <= count < percentiles[i + 1]:
                        class_grouping[class_id] = band_name
                        band_counts[band_name] += 1
                        assigned = True
                        break
                
                if not assigned:
                    class_grouping[class_id] = class_id
                    individual_count += 1
            
            band_summary = ", ".join([f"{name}: {cnt}" for name, cnt in band_counts.items() if cnt > 0])
            print(f"[INFO] Auto-computed grouping: {band_summary}, individual: {individual_count}")
        
        # Apply grouping
        if class_grouping:
            grouped_class_ids = np.array([class_grouping.get(cid, cid) for cid in class_ids])
        else:
            grouped_class_ids = class_ids
        
        # Compute weights
        unique_groups, group_counts = np.unique(grouped_class_ids, return_counts=True)
        counts_dict = {gid: int(count) for gid, count in zip(unique_groups, group_counts)}
        
        weight_map = compute_weights_from_counts(
            counts_dict,
            method=weighting_method,
            max_weight=max_weight,
            min_weight=1.0
        )
        
        # Cap weights
        if max_weight is not None:
            weight_map = {gid: min(w, max_weight) for gid, w in weight_map.items()}
        
        # Create weight tensor
        weights = np.array([weight_map.get(gid, 1.0) for gid in grouped_class_ids], dtype=np.float32)
        
        return torch.from_numpy(weights), class_grouping, weight_map
    
    # ------------------------
    # DataLoader creation
    # ------------------------
    def make_dataloader(self, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, 
                       use_weighted_sampling=False, weight_column=None, weights_stats_path=None,
                       use_grouped_weights=False, weighting_method="inverse_frequency",
                       group_rare_classes=False, class_grouping_path=None,
                       max_weight=None, exclude_extremely_rare=False, min_samples_threshold=50,
                       preferred_weight_columns=None,
                       # New options for pre-computed weights
                       use_precomputed_weights=False, precomputed_weight_column="sample_weight",
                       print_weight_stats=True, weight_stats_columns=None,
                       non_empty_multiplier=None):
        """
        Create a DataLoader with optional weighted sampling.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle (ignored if using weighted sampling)
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory
            persistent_workers: Whether to keep workers alive between epochs
            
            # Weighted sampling options (existing)
            use_weighted_sampling: Enable weighted sampling based on class frequencies
            weight_column: Column to use for class-based weighting
            weights_stats_path: Path to pre-computed weight statistics
            use_grouped_weights: Use grouped class weights
            weighting_method: Method for computing weights ("inverse_frequency", etc.)
            group_rare_classes: Auto-group rare classes
            class_grouping_path: Path to class grouping JSON
            max_weight: Maximum weight cap
            exclude_extremely_rare: Exclude very rare classes
            min_samples_threshold: Minimum samples threshold
            preferred_weight_columns: Preferred columns for weight computation
            
            # Pre-computed weight options (new)
            use_precomputed_weights: Use pre-computed weights from manifest column
            precomputed_weight_column: Column name for pre-computed weights (default: "sample_weight")
            print_weight_stats: Print weight statistics
            weight_stats_columns: Columns to group by when printing statistics
            non_empty_multiplier: Optional multiplier for non-empty rooms (e.g., 2.0 doubles their weight)
            
        Returns:
            DataLoader instance
        """
        weights = None
        
        # Option 1: Use pre-computed weights from manifest
        if use_precomputed_weights:
            weights = self._use_precomputed_weights(
                weight_column=precomputed_weight_column,
                max_weight=max_weight,
                non_empty_multiplier=non_empty_multiplier
            )
            
            if print_weight_stats:
                self._print_weight_statistics(
                    weights.numpy(),
                    group_columns=weight_stats_columns or ["type", "is_empty"]
                )
        
        # Option 2: Compute weights based on class column (existing functionality)
        elif use_weighted_sampling:
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
                print(f"[WARNING] Weight column not found, falling back to regular sampling")
            else:
                weights, class_grouping, weight_map = result
                
                if print_weight_stats:
                    # Print class-based weight info
                    actual_column = weight_column or "auto-detected"
                    print(f"\nClass sampling weights (column: {actual_column}):")
                    for cid in sorted(weight_map.keys()):
                        print(f"  {cid:30s}: weight={weight_map[cid]:.2f}")
        
        # Create DataLoader
        if weights is not None:
            sampler = WeightedRandomSampler(
                weights=weights, 
                num_samples=len(weights), 
                replacement=True
            )
            return DataLoader(
                self, 
                batch_size=batch_size, 
                sampler=sampler, 
                num_workers=num_workers,
                pin_memory=pin_memory if torch.cuda.is_available() else False,
                persistent_workers=persistent_workers if num_workers > 0 else False,
                prefetch_factor=2 if num_workers > 0 else None, 
                drop_last=False
            )
        else:
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

    # ------------------------
    # Dataset splitting
    # ------------------------
    def split(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, stratify_column=None):
        """
        Split dataset into train/val/test sets.
        
        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            seed: Random seed
            stratify_column: Column to stratify by (optional)
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        np.random.seed(seed)
        n = len(self.df)
        indices = np.random.permutation(n)
        
        if stratify_column and stratify_column in self.df.columns:
            # Stratified split
            from sklearn.model_selection import train_test_split
            
            # First split: train vs (val + test)
            train_idx, temp_idx = train_test_split(
                indices, 
                train_size=train_ratio,
                stratify=self.df.iloc[indices][stratify_column],
                random_state=seed
            )
            
            # Second split: val vs test
            val_size = val_ratio / (val_ratio + test_ratio)
            val_idx, test_idx = train_test_split(
                temp_idx,
                train_size=val_size,
                stratify=self.df.iloc[temp_idx][stratify_column],
                random_state=seed
            )
        else:
            # Random split
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            train_idx = indices[:train_end]
            val_idx = indices[train_end:val_end]
            test_idx = indices[val_end:]
        
        # Create new dataset instances with split DataFrames
        def make_split_dataset(idx):
            split_df = self.df.iloc[idx].reset_index(drop=True)
            # Create new instance with the split DataFrame
            new_kwargs = self._init_kwargs.copy()
            new_kwargs["_df"] = split_df
            new_kwargs["_manifest_dir"] = self.manifest_dir
            return ManifestDataset(**new_kwargs)
        
        return (
            make_split_dataset(train_idx),
            make_split_dataset(val_idx),
            make_split_dataset(test_idx)
        )

    # ------------------------
    # Utility methods
    # ------------------------
    def get_column_values(self, column: str) -> list:
        """Get unique values in a column."""
        if column not in self.df.columns:
            raise KeyError(f"Column '{column}' not found")
        return self.df[column].unique().tolist()
    
    def filter_by(self, **kwargs) -> "ManifestDataset":
        """
        Create a filtered copy of the dataset.
        
        Example:
            rooms_only = dataset.filter_by(type="room")
            empty_rooms = dataset.filter_by(type="room", is_empty=True)
        """
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