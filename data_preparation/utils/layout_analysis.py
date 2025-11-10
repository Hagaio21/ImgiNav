#!/usr/bin/env python3
"""
Common utilities for layout image analysis.

This module provides shared functions for analyzing layout images:
- Building color-to-category mappings
- Analyzing layout colors
- Categorizing rooms by contents
"""

from pathlib import Path
from typing import Dict, Set, Tuple
from PIL import Image

from common.taxonomy import Taxonomy


def build_color_to_category_mapping(taxonomy: Taxonomy) -> Dict[Tuple[int, int, int], str]:
    """
    Build a mapping from RGB colors to object category names.
    
    Args:
        taxonomy: Taxonomy instance
    
    Returns:
        Dict mapping RGB tuples to category names
    """
    color_to_category = {}
    id2color = taxonomy.data.get("id2color", {})
    id2category = taxonomy.data.get("id2category", {})
    id2super = taxonomy.data.get("id2super", {})
    
    # Map category IDs to colors
    for cat_id_str, cat_name in id2category.items():
        color = id2color.get(cat_id_str)
        if color:
            rgb_tuple = tuple(color) if isinstance(color, list) else tuple(color)
            color_to_category[rgb_tuple] = cat_name
    
    # Also map super-category IDs to colors (for broader categories)
    for super_id_str, super_name in id2super.items():
        color = id2color.get(super_id_str)
        if color:
            rgb_tuple = tuple(color) if isinstance(color, list) else tuple(color)
            # Only add if not already present (category takes priority)
            if rgb_tuple not in color_to_category:
                color_to_category[rgb_tuple] = super_name
    
    return color_to_category


def analyze_layout_colors(layout_path: Path, color_to_category: Dict[Tuple[int, int, int], str], 
                          min_pixel_threshold: int = 50) -> Set[str]:
    """
    Analyze layout image to identify which object categories are present.
    
    Args:
        layout_path: Path to layout image
        color_to_category: Dict mapping RGB tuples to category names
        min_pixel_threshold: Minimum number of pixels for a color to be considered present
    
    Returns:
        Set of category names present in the layout
    """
    try:
        img = Image.open(layout_path).convert("RGB")
        # Get color counts
        color_counts = img.getcolors(maxcolors=1_000_000)
        if color_counts is None:
            return set()
        
        present_categories = set()
        white_vals = {(240, 240, 240), (255, 255, 255), (200, 200, 200), (211, 211, 211)}
        
        for count, rgb in color_counts:
            # Skip background/structural colors
            if rgb in white_vals:
                continue
            
            # Check if color represents a significant object
            if count < min_pixel_threshold:
                continue
            
            rgb_tuple = tuple(rgb) if isinstance(rgb, (list, tuple)) else rgb
            category = color_to_category.get(rgb_tuple)
            if category:
                present_categories.add(category)
        
        return present_categories
    except Exception as e:
        return set()


def categorize_room_by_contents(categories_present: Set[str]) -> str:
    """
    Categorize a room based on the object categories present.
    
    Args:
        categories_present: Set of category names found in the layout
    
    Returns:
        String category name for the room
    """
    # Define priority order for categorization (most specific first)
    category_priority = [
        ("Bed", "bedroom"),
        ("Table", "dining"),
        ("Chair", "seating"),
        ("Sofa", "living"),
        ("Cabinet", "storage"),
        ("Kitchen", "kitchen"),
        ("Bath", "bathroom"),
        ("Desk", "office"),
    ]
    
    # Check for specific categories
    for cat_name, room_type in category_priority:
        if any(cat_name.lower() in cat.lower() for cat in categories_present):
            return room_type
    
    # If no specific category found, create a combined category
    if len(categories_present) > 0:
        # Sort for consistency
        sorted_cats = sorted(categories_present)
        return "_".join(sorted_cats[:3])  # Use first 3 categories
    else:
        return "empty_or_unknown"


def count_distinct_colors(layout_path: Path, exclude_background: bool = True, 
                         min_pixel_threshold: int = 0) -> int:
    """
    Count distinct colors in a layout image.
    
    Args:
        layout_path: Path to layout image
        exclude_background: If True, exclude white/gray background colors
        min_pixel_threshold: Minimum number of pixels for a color to be counted
    
    Returns:
        Number of distinct colors
    """
    try:
        img = Image.open(layout_path).convert("RGB")
        # Get color counts
        color_counts = img.getcolors(maxcolors=1_000_000)
        if color_counts is None:
            return 0
        
        distinct_colors = 0
        white_vals = {(240, 240, 240), (255, 255, 255), (200, 200, 200), (211, 211, 211)}
        
        for count, rgb in color_counts:
            # Skip background colors if requested
            if exclude_background:
                if rgb in white_vals:
                    continue
            
            # Check minimum pixel threshold
            if count < min_pixel_threshold:
                continue
            
            distinct_colors += 1
        
        return distinct_colors
    except Exception as e:
        return 0


def get_object_class_combination(layout_path: Path, color_to_category: Dict[Tuple[int, int, int], str],
                                min_pixel_threshold: int = 50) -> str:
    """
    Get a string representation of the combination of object classes present.
    
    Returns a sorted, comma-separated string of category names.
    This can be used to identify common vs rare layout combinations.
    
    Args:
        layout_path: Path to layout image
        color_to_category: Dict mapping RGB tuples to category names
        min_pixel_threshold: Minimum number of pixels for a color to be counted
    
    Returns:
        Sorted comma-separated string of category names (e.g., "Bed,Chair,Table")
        or "empty" if no objects found
    """
    try:
        categories_present = analyze_layout_colors(layout_path, color_to_category, min_pixel_threshold)
        if not categories_present:
            return "empty"
        
        # Sort for consistency (same combination always produces same string)
        sorted_categories = sorted(categories_present)
        return ",".join(sorted_categories)
    except Exception as e:
        return "unknown"


def count_distinct_object_classes(layout_path: Path, color_to_category: Dict[Tuple[int, int, int], str],
                                 min_pixel_threshold: int = 50) -> int:
    """
    Count distinct object classes in a layout image by counting unique categories.
    
    Only counts colors that map to actual object categories in the taxonomy,
    not all RGB colors (which could include gradients, artifacts, etc.).
    
    Args:
        layout_path: Path to layout image
        color_to_category: Dict mapping RGB tuples to category names
        min_pixel_threshold: Minimum number of pixels for a color to be counted
    
    Returns:
        Number of distinct object classes (categories)
    """
    try:
        img = Image.open(layout_path).convert("RGB")
        # Get color counts
        color_counts = img.getcolors(maxcolors=1_000_000)
        if color_counts is None:
            return 0
        
        distinct_categories = set()
        white_vals = {(240, 240, 240), (255, 255, 255), (200, 200, 200), (211, 211, 211)}
        
        for count, rgb in color_counts:
            # Skip background/structural colors
            if rgb in white_vals:
                continue
            
            # Check if color represents a significant object
            if count < min_pixel_threshold:
                continue
            
            rgb_tuple = tuple(rgb) if isinstance(rgb, (list, tuple)) else rgb
            category = color_to_category.get(rgb_tuple)
            if category:
                distinct_categories.add(category)
        
        num_classes = len(distinct_categories)
        # Sanity check: if we somehow get an unreasonably high number, something is wrong
        if num_classes > 1000:
            # This shouldn't happen - log a warning
            import warnings
            warnings.warn(f"Unusually high class count ({num_classes}) for {layout_path}. "
                        f"Taxonomy has {len(color_to_category)} color mappings.")
        
        return num_classes
    except Exception as e:
        return 0

