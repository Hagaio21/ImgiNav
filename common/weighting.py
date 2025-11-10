#!/usr/bin/env python3
"""
Generic weighting utilities for manifest columns.

This module provides functions to compute weights from any manifest column
and apply them for balanced sampling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
import json


def compute_weights_from_counts(
    counts_dict: Dict[str, int],
    method: str = "inverse_frequency",
    max_weight: Optional[float] = None,
    min_weight: float = 1.0
) -> Dict[str, float]:
    """
    Compute weights from class counts using various methods.
    
    Args:
        counts_dict: Dict mapping class_id to count
        method: Weighting method ("inverse_frequency", "sqrt", "log", "balanced")
        max_weight: Maximum weight to cap (None = no capping)
        min_weight: Minimum weight (default: 1.0)
    
    Returns:
        Dict mapping class_id to weight
    """
    if not counts_dict:
        return {}
    
    counts = np.array(list(counts_dict.values()))
    max_count = max(counts)
    total_samples = sum(counts)
    num_classes = len(counts_dict)
    
    weights = {}
    
    if method == "inverse_frequency":
        # Standard inverse frequency: weight = total_samples / (num_classes * count)
        for class_id, count in counts_dict.items():
            weight = total_samples / (num_classes * count)
            weights[class_id] = weight
    
    elif method == "sqrt":
        # Square root weighting: weight = sqrt(max_count / count)
        for class_id, count in counts_dict.items():
            weight = np.sqrt(max_count / count)
            weights[class_id] = weight
    
    elif method == "log":
        # Logarithmic weighting: weight = log(max_count / count + 1)
        for class_id, count in counts_dict.items():
            weight = np.log(max_count / count + 1)
            weights[class_id] = weight
    
    elif method == "balanced":
        # Balanced: weight = max_count / count
        for class_id, count in counts_dict.items():
            weight = max_count / count
            weights[class_id] = weight
    
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Apply min/max constraints
    if min_weight is not None:
        weights = {k: max(v, min_weight) for k, v in weights.items()}
    
    if max_weight is not None:
        weights = {k: min(v, max_weight) for k, v in weights.items()}
    
    return weights


def load_weights_from_stats(stats_path: Union[str, Path], use_grouped: bool = False) -> Dict[str, float]:
    """
    Load weights from a distribution stats JSON file.
    
    Args:
        stats_path: Path to distribution stats JSON
        use_grouped: If True, use grouped_weights; otherwise use weights
    
    Returns:
        Dict mapping class_id to weight
    """
    stats_path = Path(stats_path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    if use_grouped:
        weights = stats.get("grouped_weights", {})
    else:
        weights = stats.get("weights", {})
    
    return weights


def apply_weights_to_dataframe(
    df: pd.DataFrame,
    column_name: str,
    weights: Optional[Dict[str, float]] = None,
    stats_path: Optional[Union[str, Path]] = None,
    use_grouped: bool = False,
    default_weight: float = 1.0
) -> pd.DataFrame:
    """
    Apply weights to a DataFrame based on a column.
    
    Args:
        df: DataFrame to weight
        column_name: Column name to use for weighting
        weights: Optional dict of weights (if None, will load from stats_path)
        stats_path: Optional path to stats JSON to load weights from
        use_grouped: If True, use grouped weights from stats
        default_weight: Default weight for missing values
    
    Returns:
        DataFrame with added "sample_weight" column
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    # Load weights if not provided
    if weights is None:
        if stats_path is None:
            raise ValueError("Either weights or stats_path must be provided")
        weights = load_weights_from_stats(stats_path, use_grouped=use_grouped)
    
    # Apply grouping if using grouped weights
    if use_grouped and stats_path:
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        class_grouping = stats.get("class_grouping", {})
        
        # Map original values to grouped values
        def get_grouped_value(value):
            if pd.isna(value):
                return None
            value_str = str(value)
            return class_grouping.get(value_str, value_str)
        
        df["_grouped_value"] = df[column_name].apply(get_grouped_value)
        weight_column = "_grouped_value"
    else:
        weight_column = column_name
    
    # Map weights
    df["sample_weight"] = df[weight_column].map(weights)
    df["sample_weight"] = df["sample_weight"].fillna(default_weight)
    
    # Clean up temporary column
    if "_grouped_value" in df.columns:
        df = df.drop(columns=["_grouped_value"])
    
    return df


def weighted_sample(
    df: pd.DataFrame,
    n: int,
    weight_column: str = "sample_weight",
    random_state: int = 42,
    replace: bool = False
) -> pd.DataFrame:
    """
    Perform weighted sampling from a DataFrame.
    
    Args:
        df: DataFrame to sample from
        n: Number of samples to draw
        weight_column: Column name containing weights
        random_state: Random seed
        replace: Whether to sample with replacement
    
    Returns:
        Sampled DataFrame
    """
    if weight_column not in df.columns:
        raise ValueError(f"Weight column '{weight_column}' not found in DataFrame")
    
    if len(df) < n and not replace:
        print(f"Warning: Requested {n} samples but only {len(df)} available. Using all samples.")
        n = len(df)
    
    return df.sample(n=n, weights=weight_column, replace=replace, random_state=random_state).reset_index(drop=True)

