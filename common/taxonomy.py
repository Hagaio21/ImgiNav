#!/usr/bin/env python3
"""
Taxonomy module for ImgiNav.

Provides the Taxonomy class for loading and querying taxonomy JSON files.
The taxonomy JSON should be built using data_preparation_v2/stage0_build_taxonomy.py.

Taxonomy JSON format (name-based):
{
    "version": "2.1",
    "categories": ["Chair", "Table", ...],
    "supercategories": ["Seating", "Storage", ...],
    "room_types": ["Bedroom", "Kitchen", ...],
    "category_to_super": {"Chair": "Seating", ...},
    "category_to_color": {"Chair": [255, 0, 0], ...},
    "supercategory_to_color": {"Seating": [200, 0, 0], ...},
    "title_to_category": {"Modern Chair": "Chair", ...},
    "title_to_super": {"Modern Chair": "Seating", ...}
}
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class Taxonomy:
    """
    Taxonomy class for category/supercategory lookups and color mappings.
    
    Uses a simple name-based format for all lookups.
    """

    def __init__(self, taxonomy_path: str | Path):
        with open(Path(taxonomy_path), "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        # Precompute reverse mappings for fast lookups
        self._init_lookups()
    
    def _init_lookups(self):
        """Initialize lookup dictionaries."""
        # Color lookups (RGB tuple -> name)
        self.color_to_category = {}
        self.color_to_super = {}
        
        for name, rgb in self.data.get("category_to_color", {}).items():
            rgb_tuple = tuple(rgb) if isinstance(rgb, list) else rgb
            self.color_to_category[rgb_tuple] = name
        
        for name, rgb in self.data.get("supercategory_to_color", {}).items():
            rgb_tuple = tuple(rgb) if isinstance(rgb, list) else rgb
            self.color_to_super[rgb_tuple] = name
        
        # Structural categories for special handling
        self.structural_categories = {'wall', 'floor', 'ceiling', 'door', 'window'}

    # =========================================================================
    # Basic lookups
    # =========================================================================
    
    def get_categories(self) -> List[str]:
        """Get list of all categories."""
        return self.data.get("categories", [])
    
    def get_supercategories(self) -> List[str]:
        """Get list of all supercategories."""
        return self.data.get("supercategories", [])
    
    def get_room_types(self) -> List[str]:
        """Get list of all room types."""
        return self.data.get("room_types", [])
    
    def get_super(self, category: str) -> str:
        """Get supercategory for a category."""
        return self.data.get("category_to_super", {}).get(category, "Others")
    
    def get_category_from_title(self, title: str) -> str:
        """Get category for a furniture title."""
        return self.data.get("title_to_category", {}).get(title, "Unknown")
    
    def get_super_from_title(self, title: str) -> str:
        """Get supercategory for a furniture title."""
        return self.data.get("title_to_super", {}).get(title, "Others")

    # =========================================================================
    # Color lookups
    # =========================================================================
    
    def get_color(self, name: str, mode: str = "category") -> Tuple[int, int, int]:
        """
        Get RGB color for a category or supercategory name.
        
        Args:
            name: Category or supercategory name
            mode: "category" or "super" to specify which mapping to use
        
        Returns:
            RGB tuple (0-255)
        """
        default_color = (127, 127, 127)
        
        if mode == "category":
            color = self.data.get("category_to_color", {}).get(name)
            if color:
                return tuple(color)
            # Try case-insensitive match
            for cat, col in self.data.get("category_to_color", {}).items():
                if cat.lower() == name.lower():
                    return tuple(col)
        else:  # super
            color = self.data.get("supercategory_to_color", {}).get(name)
            if color:
                return tuple(color)
            # Try case-insensitive match
            for sup, col in self.data.get("supercategory_to_color", {}).items():
                if sup.lower() == name.lower():
                    return tuple(col)
        
        return default_color
    
    def get_category_color(self, category: str) -> Tuple[int, int, int]:
        """Get RGB color for a category."""
        return self.get_color(category, mode="category")
    
    def get_super_color(self, supercategory: str) -> Tuple[int, int, int]:
        """Get RGB color for a supercategory."""
        return self.get_color(supercategory, mode="super")
    
    def get_name_from_color(self, rgb: Tuple[int, int, int], mode: str = "category") -> Optional[str]:
        """
        Get category or supercategory name from RGB color.
        
        Args:
            rgb: RGB tuple (0-255)
            mode: "category" or "super"
        
        Returns:
            Name or None if not found
        """
        if mode == "category":
            return self.color_to_category.get(rgb)
        else:
            return self.color_to_super.get(rgb)
    
    def match_color_to_category(self, rgb: Tuple[int, int, int], 
                                 threshold: float = 30.0) -> Tuple[Optional[str], float]:
        """
        Find the closest matching category for an RGB color.
        
        Args:
            rgb: RGB tuple (0-255)
            threshold: Maximum Euclidean distance to consider a match
        
        Returns:
            Tuple of (category_name, distance) or (None, inf) if no match
        """
        import math
        
        best_match = None
        best_dist = float('inf')
        
        for cat, color in self.data.get("category_to_color", {}).items():
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb, color)))
            if dist < best_dist:
                best_dist = dist
                best_match = cat
        
        if best_dist <= threshold:
            return best_match, best_dist
        return None, float('inf')

    # =========================================================================
    # Utility methods
    # =========================================================================
    
    def is_structural(self, name: str) -> bool:
        """Check if a category is structural (wall, floor, ceiling, door, window)."""
        return name.lower() in self.structural_categories
