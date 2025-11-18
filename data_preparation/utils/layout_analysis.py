#!/usr/bin/env python3
"""
Common utilities for layout image analysis.

This module provides shared functions for analyzing layout images:
- Building color-to-category mappings
- Analyzing layout colors
- Categorizing rooms by contents
- Segmenting layout images to category maps
"""

from pathlib import Path
from typing import Dict, Set, Tuple, Optional, Union
import numpy as np
from PIL import Image

from common.taxonomy import Taxonomy


def build_color_to_category_mapping(taxonomy: Taxonomy, super_categories_only: bool = True) -> Dict[Tuple[int, int, int], str]:
    """
    Build a mapping from RGB colors to object category names.
    
    Args:
        taxonomy: Taxonomy instance
        super_categories_only: If True, only map super-categories; if False, map both categories and super-categories
    
    Returns:
        Dict mapping RGB tuples to category/super-category names
    """
    color_to_category = {}
    id2color = taxonomy.data.get("id2color", {})
    id2super = taxonomy.data.get("id2super", {})
    
    if super_categories_only:
        # Only map super-category IDs to colors
        for super_id_str, super_name in id2super.items():
            color = id2color.get(super_id_str)
            if color:
                rgb_tuple = tuple(color) if isinstance(color, list) else tuple(color)
                color_to_category[rgb_tuple] = super_name
    else:
        # Map both categories and super-categories (original behavior)
        id2category = taxonomy.data.get("id2category", {})
        
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


def analyze_layout_colors(layout_path: Path, color_to_category: Dict[Tuple[int, int, int], str]) -> Set[str]:
    """
    Analyze layout image to identify which object categories are present.
    
    Args:
        layout_path: Path to layout image
        color_to_category: Dict mapping RGB tuples to category names
    
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


def get_object_class_combination(layout_path: Path, color_to_category: Dict[Tuple[int, int, int], str]) -> str:
    """
    Get a string representation of the combination of object classes present.
    
    Returns a sorted, comma-separated string of category names.
    This can be used to identify common vs rare layout combinations.
    
    Args:
        layout_path: Path to layout image
        color_to_category: Dict mapping RGB tuples to category names
    
    Returns:
        Sorted comma-separated string of category names (e.g., "Bed,Chair,Table")
        or "empty" if no objects found
    """
    try:
        categories_present = analyze_layout_colors(layout_path, color_to_category)
        if not categories_present:
            return "empty"
        
        # Sort for consistency (same combination always produces same string)
        sorted_categories = sorted(categories_present)
        return ",".join(sorted_categories)
    except Exception as e:
        return "unknown"


def count_distinct_object_classes(layout_path: Path, color_to_category: Dict[Tuple[int, int, int], str] = None) -> int:
    """
    Count distinct colors in a layout image (excluding background colors).
    
    This counts all distinct RGB colors present, not just those mapped to categories.
    Used for content_category column.
    
    Args:
        layout_path: Path to layout image
        color_to_category: Not used, kept for compatibility
    
    Returns:
        Number of distinct colors (excluding white/background)
    """
    try:
        img = Image.open(layout_path).convert("RGB")
        # Get color counts
        color_counts = img.getcolors(maxcolors=1_000_000)
        if color_counts is None:
            return 0
        
        distinct_colors = set()
        white_vals = {(240, 240, 240), (255, 255, 255), (200, 200, 200), (211, 211, 211)}
        
        for count, rgb in color_counts:
            # Skip background/structural colors
            if rgb in white_vals:
                continue

            rgb_tuple = tuple(rgb) if isinstance(rgb, (list, tuple)) else rgb
            distinct_colors.add(rgb_tuple)

        return len(distinct_colors)
    except Exception as e:
        return 0


class LayoutSegmentor:
    """
    Segmentor that maps each pixel in a layout image to the closest category color.
    
    For generated layouts where colors may not exactly match taxonomy colors,
    this finds the nearest color in RGB space and assigns the corresponding category.
    """
    
    def __init__(self, taxonomy: Taxonomy, mode: str = "category"):
        """
        Initialize the segmentor.
        
        Args:
            taxonomy: Taxonomy instance for color-to-category mapping
            mode: "category" or "super" - whether to segment by category or super-category
        """
        self.taxonomy = taxonomy
        self.mode = mode
        
        # Build color to ID mapping
        self.color_to_id = {}
        self.colors_array = []  # List of RGB tuples
        self.ids_array = []     # List of corresponding IDs
        
        id2color = taxonomy.data.get("id2color", {})
        
        if mode == "category":
            # Map category IDs to colors
            id2category = taxonomy.data.get("id2category", {})
            for cat_id_str, cat_name in id2category.items():
                color = id2color.get(cat_id_str)
                if color:
                    rgb_tuple = tuple(color) if isinstance(color, list) else tuple(color)
                    cat_id = int(cat_id_str)
                    if rgb_tuple not in self.color_to_id:  # Avoid duplicates
                        self.color_to_id[rgb_tuple] = cat_id
                        self.colors_array.append(rgb_tuple)
                        self.ids_array.append(cat_id)
            
            # Also include structural categories (floor, wall, ceiling)
            for struct_id in [2051, 2052, 2053]:  # ceiling, floor, wall
                struct_id_str = str(struct_id)
                color = id2color.get(struct_id_str)
                if color:
                    rgb_tuple = tuple(color) if isinstance(color, list) else tuple(color)
                    if rgb_tuple not in self.color_to_id:
                        self.color_to_id[rgb_tuple] = struct_id
                        self.colors_array.append(rgb_tuple)
                        self.ids_array.append(struct_id)
        else:
            # Map super-category IDs to colors
            id2super = taxonomy.data.get("id2super", {})
            for super_id_str, super_name in id2super.items():
                color = id2color.get(super_id_str)
                if color:
                    rgb_tuple = tuple(color) if isinstance(color, list) else tuple(color)
                    super_id = int(super_id_str)
                    if rgb_tuple not in self.color_to_id:
                        self.color_to_id[rgb_tuple] = super_id
                        self.colors_array.append(rgb_tuple)
                        self.ids_array.append(super_id)
            
            # Also include wall (2053) which is a category but commonly used
            wall_color = id2color.get("2053")
            if wall_color:
                rgb_tuple = tuple(wall_color) if isinstance(wall_color, list) else tuple(wall_color)
                if rgb_tuple not in self.color_to_id:
                    # Resolve wall to its super-category
                    wall_super_id = taxonomy.resolve_super(2053)
                    if wall_super_id:
                        self.color_to_id[rgb_tuple] = wall_super_id
                        self.colors_array.append(rgb_tuple)
                        self.ids_array.append(wall_super_id)
        
        # Convert to numpy arrays for efficient distance computation
        self.colors_array = np.array(self.colors_array, dtype=np.float32)  # (N, 3)
        self.ids_array = np.array(self.ids_array, dtype=np.int32)  # (N,)
        
        if len(self.colors_array) == 0:
            raise ValueError("No valid colors found in taxonomy")
    
    def segment(self, image: Union[Image.Image, np.ndarray], 
                return_names: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Segment a layout image by finding the closest color for each pixel.
        
        Args:
            image: PIL Image or numpy array (H, W, 3) in RGB format, uint8 [0, 255]
            return_names: If True, also return category/super-category names array
        
        Returns:
            If return_names=False: segmentation map as (H, W) array of category/super-category IDs
            If return_names=True: tuple of (segmentation_map, names_array) where names_array is (H, W) of strings
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"), dtype=np.uint8)
        
        # Ensure image is in correct format
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Handle different input shapes
        if image.ndim == 2:
            # Grayscale - convert to RGB by repeating channels
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3:
            if image.shape[2] != 3:
                raise ValueError(f"Expected 3 channels (RGB), got {image.shape[2]}")
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        
        H, W, C = image.shape
        image_flat = image.reshape(-1, 3).astype(np.float32)  # (H*W, 3)
        
        # Compute Euclidean distances to all colors
        # Broadcasting: (H*W, 1, 3) - (1, N, 3) = (H*W, N, 3)
        diff = image_flat[:, np.newaxis, :] - self.colors_array[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))  # (H*W, N)
        
        # Find closest color for each pixel
        closest_indices = np.argmin(distances, axis=1)  # (H*W,)
        
        # Map to category/super-category IDs
        seg_map = self.ids_array[closest_indices].reshape(H, W)
        
        if return_names:
            # Map IDs to names
            names_flat = np.array([self.taxonomy.id_to_name(int(id_val)) for id_val in seg_map.flatten()])
            names_map = names_flat.reshape(H, W)
            return seg_map, names_map
        else:
            return seg_map
    
    def segment_to_image(self, image: Union[Image.Image, np.ndarray], 
                        colormap: Optional[Dict[int, Tuple[int, int, int]]] = None,
                        background_color: Tuple[int, int, int] = (255, 255, 255),
                        background_threshold: int = 30) -> Image.Image:
        """
        Segment image and return as a color-coded visualization.
        Each pixel is "shifted" to its closest category color.
        Background pixels (white/light gray) are set to white.
        
        Args:
            image: PIL Image or numpy array to segment
            colormap: Optional dict mapping category/super-category ID to RGB color.
                     If None, uses original taxonomy colors.
            background_color: RGB color to use for background pixels (default: white)
            background_threshold: Threshold for detecting background pixels.
                                Pixels with average RGB value > (255 - threshold) are considered background.
        
        Returns:
            PIL Image with each pixel colored according to its assigned category,
            with background pixels set to white
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image_array = np.array(image.convert("RGB"), dtype=np.uint8)
        else:
            image_array = image.copy()
            if image_array.dtype != np.uint8:
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = image_array.astype(np.uint8)
        
        # Handle different input shapes
        if image_array.ndim == 2:
            image_array = np.stack([image_array, image_array, image_array], axis=-1)
        elif image_array.ndim == 3 and image_array.shape[2] != 3:
            raise ValueError(f"Expected 3 channels (RGB), got {image_array.shape[2]}")
        
        H, W, C = image_array.shape
        
        # Segment to get category IDs
        seg_map = self.segment(image_array, return_names=False)
        
        # Create output image - start with category colors
        output = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Build ID to color mapping
        if colormap is None:
            # Use original taxonomy colors
            id2color = self.taxonomy.data.get("id2color", {})
            id_to_color = {}
            for cat_id in np.unique(seg_map):
                color = id2color.get(str(int(cat_id)))
                if color:
                    rgb = tuple(color) if isinstance(color, list) else tuple(color)
                    id_to_color[int(cat_id)] = rgb
        else:
            id_to_color = colormap
        
        # Map each pixel to its closest category color
        for cat_id in np.unique(seg_map):
            mask = (seg_map == cat_id)
            if int(cat_id) in id_to_color:
                output[mask] = id_to_color[int(cat_id)]
        
        # Detect and set background pixels to white
        # Background: pixels that are very light (close to white)
        # Check if original pixel is close to white/light gray
        avg_intensity = image_array.mean(axis=2)  # (H, W) - average of R, G, B
        is_background = avg_intensity > (255 - background_threshold)
        
        # Set background pixels to white
        output[is_background] = background_color
        
        return Image.fromarray(output)


def segment_layout(layout_path: Union[Path, str], 
                  taxonomy_path: Union[Path, str] = "config/taxonomy.json",
                  mode: str = "category",
                  output_path: Optional[Union[Path, str]] = None,
                  return_seg_map: bool = False) -> Optional[np.ndarray]:
    """
    Convenience function to segment a layout image file.
    
    Args:
        layout_path: Path to layout image
        taxonomy_path: Path to taxonomy.json file
        mode: "category" or "super" - segment by category or super-category
        output_path: Optional path to save segmentation visualization
        return_seg_map: If True, return the segmentation map array
    
    Returns:
        If return_seg_map=True: segmentation map as (H, W) array of IDs
        If return_seg_map=False: None
    """
    layout_path = Path(layout_path)
    taxonomy_path = Path(taxonomy_path)
    
    if not layout_path.exists():
        raise FileNotFoundError(f"Layout image not found: {layout_path}")
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_path}")
    
    # Load taxonomy and create segmentor
    taxonomy = Taxonomy(taxonomy_path)
    segmentor = LayoutSegmentor(taxonomy, mode=mode)
    
    # Load and segment image
    image = Image.open(layout_path).convert("RGB")
    seg_map = segmentor.segment(image)
    
    # Save visualization if requested
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vis_image = segmentor.segment_to_image(image)
        vis_image.save(output_path)
    
    if return_seg_map:
        return seg_map
    return None

