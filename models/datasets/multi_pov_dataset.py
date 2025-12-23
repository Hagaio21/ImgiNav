#!/usr/bin/env python3
"""
Multi-POV Dataset wrapper for iterative refinement experiments.

Wraps the standard ManifestDataset to provide:
1. All POV embeddings for a given room (not just one)
2. Support for accumulation experiments (progressively more POVs)
3. Support for refinement experiments (prior + new POV)
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.datasets.datasets import ManifestDataset


class MultiPOVDataset(Dataset):
    """
    Dataset that provides all POV embeddings for each room.
    
    Groups samples by room_id and returns all POVs for refinement experiments.
    
    Usage:
        dataset = MultiPOVDataset(
            manifest_path="manifest.csv",
            outputs={
                "latent": "latent_embedding_path",
                "text_emb": "graph_embedding_path",
                "pov_emb": "pov_embedding_path"
            },
            group_by="room_id"  # or "scene_id"
        )
        
        sample = dataset[0]
        # sample["latent"]: target latent
        # sample["text_emb"]: graph embedding (single)
        # sample["pov_emb"]: primary POV embedding
        # sample["all_pov_embs"]: list of all POV embeddings for this room
        # sample["num_povs"]: number of POVs available
    """
    
    def __init__(
        self,
        manifest_path: Path,
        outputs: Dict[str, str],
        group_by: str = "room_id",
        filters: Optional[Dict] = None,
        max_povs_per_room: int = 10,
        return_paths: bool = False,
        pov_columns: Optional[List[str]] = None
    ):
        """
        Args:
            manifest_path: Path to manifest CSV
            outputs: Mapping of output keys to column names
            group_by: Column to group POVs by (room_id or scene_id)
            filters: Optional filters to apply
            max_povs_per_room: Maximum POVs to return per room
            return_paths: Whether to return file paths
            pov_columns: List of column names for POV embeddings (for column-based format).
                         If provided, POVs are read from these columns instead of grouping rows.
        """
        self.manifest_path = Path(manifest_path)
        self.outputs = outputs
        self.group_by = group_by
        self.max_povs_per_room = max_povs_per_room
        self.return_paths = return_paths
        self.pov_columns = pov_columns
        
        # Load manifest
        self.df = pd.read_csv(self.manifest_path, low_memory=False)
        self.df.columns = self.df.columns.str.strip()
        self.manifest_dir = self.manifest_path.parent
        
        # Apply filters
        if filters:
            self.df = self._apply_filters(self.df, filters)
        
        # Check if using column-based POV format
        if self.pov_columns:
            # Each row is a sample, POVs come from columns
            self.group_ids = list(range(len(self.df)))
            self.groups = {i: [i] for i in range(len(self.df))}
            print(f"[MultiPOVDataset] Using column-based POV format with {len(self.pov_columns)} POV columns")
            print(f"[MultiPOVDataset] Found {len(self.group_ids)} samples")
        else:
            # Validate group_by column exists
            if group_by not in self.df.columns:
                raise ValueError(f"Group column '{group_by}' not found. Available: {list(self.df.columns)}")
            # Group by room/scene
            self._build_groups()
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply filters to DataFrame."""
        for key, value in filters.items():
            if key not in df.columns:
                print(f"[WARNING] Filter column '{key}' not found, skipping")
                continue
            
            if key == "rejected":
                def to_bool(val):
                    if isinstance(val, bool):
                        return val
                    if isinstance(val, str):
                        return val.lower() in ("true", "1", "yes")
                    return bool(val)
                df_bool = df[key].apply(to_bool)
                filter_bool = to_bool(value)
                df = df[df_bool == filter_bool]
            elif isinstance(value, (list, tuple)):
                df = df[df[key].isin(value)]
            else:
                df = df[df[key] == value]
        
        return df.reset_index(drop=True)
    
    def _build_groups(self):
        """Build mapping from group ID to row indices."""
        self.groups = defaultdict(list)
        
        for idx, row in self.df.iterrows():
            group_id = row[self.group_by]
            self.groups[group_id].append(idx)
        
        # Convert to list for indexing
        self.group_ids = list(self.groups.keys())
        
        print(f"[MultiPOVDataset] Found {len(self.group_ids)} unique groups")
        print(f"[MultiPOVDataset] Avg POVs per group: {len(self.df) / len(self.group_ids):.1f}")
    
    def __len__(self):
        return len(self.group_ids)
    
    def _resolve_path(self, path_value) -> Optional[Path]:
        """Resolve path relative to manifest directory."""
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
        
        return path if path.exists() else None
    
    def _load_tensor(self, path_value) -> Optional[torch.Tensor]:
        """Load tensor from path."""
        path = self._resolve_path(path_value)
        if path is None or not path.exists():
            return None
        
        if path.suffix == ".pt":
            return torch.load(path, map_location="cpu", weights_only=True)
        return None
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample with all POVs for the group.
        
        Returns:
            Dictionary with:
                - latent: target latent (from first row)
                - text_emb: graph embedding (from first row)
                - pov_emb: primary POV embedding (from first row)
                - all_pov_embs: list of all POV embeddings
                - num_povs: number of POVs
                - group_id: the group identifier
        """
        group_id = self.group_ids[idx]
        row_indices = self.groups[group_id]
        
        # Use first row as primary
        primary_row = self.df.iloc[row_indices[0]]
        
        sample = {
            "group_id": group_id,
        }
        
        # Load outputs from primary row
        for key, col in self.outputs.items():
            if col in primary_row.index:
                value = self._load_tensor(primary_row[col])
                if value is not None:
                    sample[key] = value
        
        # Load all POV embeddings
        all_pov_embs = []
        
        if self.pov_columns:
            # Column-based format: load POVs from specified columns
            for pov_col in self.pov_columns[:self.max_povs_per_room]:
                if pov_col in primary_row.index:
                    pov_emb = self._load_tensor(primary_row[pov_col])
                    if pov_emb is not None:
                        all_pov_embs.append(pov_emb)
        else:
            # Row-based format: load POVs from multiple rows
            pov_col = self.outputs.get("pov_emb")
            if pov_col:
                for row_idx in row_indices[:self.max_povs_per_room]:
                    row = self.df.iloc[row_idx]
                    pov_emb = self._load_tensor(row[pov_col])
                    if pov_emb is not None:
                        all_pov_embs.append(pov_emb)
        
        sample["all_pov_embs"] = all_pov_embs
        sample["num_povs"] = len(all_pov_embs)
        
        if self.return_paths:
            sample["paths"] = {
                key: str(primary_row[col]) 
                for key, col in self.outputs.items() 
                if col in primary_row.index
            }
        
        return sample
    
    def get_sample_with_n_povs(
        self,
        idx: int,
        n_povs: int,
        combine_method: str = "average"
    ) -> Dict[str, Any]:
        """
        Get a sample with exactly N POVs combined.
        
        Args:
            idx: Sample index
            n_povs: Number of POVs to include (0 = no POV conditioning)
            combine_method: "average", "first", or "concat"
        
        Returns:
            Sample with combined POV embedding
        """
        sample = self[idx]
        all_povs = sample.get("all_pov_embs", [])
        
        if n_povs == 0 or len(all_povs) == 0:
            # No POV conditioning
            sample["combined_pov_emb"] = None
        elif n_povs >= len(all_povs):
            # Use all available
            if combine_method == "average":
                sample["combined_pov_emb"] = torch.stack(all_povs).mean(dim=0)
            elif combine_method == "first":
                sample["combined_pov_emb"] = all_povs[0]
            else:  # concat
                sample["combined_pov_emb"] = torch.cat(all_povs, dim=-1)
        else:
            # Use first N POVs
            povs_to_use = all_povs[:n_povs]
            if combine_method == "average":
                sample["combined_pov_emb"] = torch.stack(povs_to_use).mean(dim=0)
            elif combine_method == "first":
                sample["combined_pov_emb"] = povs_to_use[0]
            else:  # concat
                sample["combined_pov_emb"] = torch.cat(povs_to_use, dim=-1)
        
        sample["n_povs_used"] = min(n_povs, len(all_povs))
        return sample


def create_refinement_dataloader(
    manifest_path: Path,
    outputs: Dict[str, str],
    group_by: str = "room_id",
    filters: Optional[Dict] = None,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0
):
    """
    Create a DataLoader for refinement experiments.
    
    Note: batch_size should typically be 1 for refinement experiments
    since each sample has variable number of POVs.
    """
    from torch.utils.data import DataLoader
    
    dataset = MultiPOVDataset(
        manifest_path=manifest_path,
        outputs=outputs,
        group_by=group_by,
        filters=filters
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda x: x[0] if len(x) == 1 else x  # Don't stack for batch_size=1
    )


if __name__ == "__main__":
    # Quick test
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    
    dataset = MultiPOVDataset(
        manifest_path=args.manifest,
        outputs={
            "latent": "latent_embedding_path",
            "text_emb": "graph_embedding_path",
            "pov_emb": "pov_embedding_path"
        },
        filters={"rejected": False}
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test first sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Group ID: {sample['group_id']}")
    print(f"Num POVs: {sample['num_povs']}")
    
    if "latent" in sample:
        print(f"Latent shape: {sample['latent'].shape}")
    if "text_emb" in sample:
        print(f"Text emb shape: {sample['text_emb'].shape}")
    if sample["all_pov_embs"]:
        print(f"POV emb shape: {sample['all_pov_embs'][0].shape}")