#!/usr/bin/env python3
"""
Visualize Taxonomy Colors

Displays the color mappings from taxonomy.json using matplotlib.
Shows both category colors and supercategory colors.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_taxonomy(taxonomy_path: Path) -> Dict:
    """Load taxonomy JSON file."""
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        return json.load(f)


def rgb_to_hex(rgb: List[int]) -> str:
    """Convert RGB [0-255] to hex color string."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def visualize_categories(taxonomy: Dict, output_path: Path = None, figsize: Tuple[int, int] = (14, 20)):
    """Visualize category colors."""
    category_to_color = taxonomy.get("category_to_color", {})
    
    if not category_to_color:
        print("No category_to_color found in taxonomy")
        return
    
    # Sort categories alphabetically for consistent display
    categories = sorted(category_to_color.keys())
    n_categories = len(categories)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_categories)
    ax.axis('off')
    ax.set_title('Taxonomy Category Colors', fontsize=16, fontweight='bold', pad=20)
    
    # Display each category with its color
    for i, category in enumerate(categories):
        rgb = category_to_color[category]
        hex_color = rgb_to_hex(rgb)
        
        # Color patch
        y_pos = n_categories - i - 0.5
        rect = mpatches.Rectangle((0.05, y_pos - 0.4), 0.15, 0.8, 
                                 facecolor=hex_color, edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        
        # Category name
        ax.text(0.25, y_pos, category, fontsize=9, verticalalignment='center',
               fontfamily='monospace')
        
        # RGB values
        ax.text(0.65, y_pos, f"RGB({rgb[0]}, {rgb[1]}, {rgb[2]})", 
               fontsize=8, verticalalignment='center', color='gray')
        
        # Hex color
        ax.text(0.85, y_pos, hex_color, fontsize=8, verticalalignment='center',
               color='gray', fontfamily='monospace')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved category visualization to: {output_path}")
    else:
        plt.show()


def visualize_supercategories(taxonomy: Dict, output_path: Path = None, figsize: Tuple[int, int] = (12, 8)):
    """Visualize supercategory colors."""
    supercategory_to_color = taxonomy.get("supercategory_to_color", {})
    
    if not supercategory_to_color:
        print("No supercategory_to_color found in taxonomy")
        return
    
    # Sort supercategories alphabetically
    supercategories = sorted(supercategory_to_color.keys())
    n_super = len(supercategories)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_super)
    ax.axis('off')
    ax.set_title('Taxonomy Supercategory Colors', fontsize=16, fontweight='bold', pad=20)
    
    # Display each supercategory with its color
    for i, supercat in enumerate(supercategories):
        rgb = supercategory_to_color[supercat]
        hex_color = rgb_to_hex(rgb)
        
        # Color patch (larger for supercategories)
        y_pos = n_super - i - 0.5
        rect = mpatches.Rectangle((0.1, y_pos - 0.4), 0.2, 0.8, 
                                 facecolor=hex_color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        
        # Supercategory name
        ax.text(0.35, y_pos, supercat, fontsize=11, verticalalignment='center',
               fontweight='bold', fontfamily='monospace')
        
        # RGB values
        ax.text(0.65, y_pos, f"RGB({rgb[0]}, {rgb[1]}, {rgb[2]})", 
               fontsize=10, verticalalignment='center', color='gray')
        
        # Hex color
        ax.text(0.85, y_pos, hex_color, fontsize=10, verticalalignment='center',
               color='gray', fontfamily='monospace')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved supercategory visualization to: {output_path}")
    else:
        plt.show()


def visualize_color_grid(taxonomy: Dict, output_path: Path = None, figsize: Tuple[int, int] = (16, 12)):
    """Visualize all colors in a grid layout."""
    category_to_color = taxonomy.get("category_to_color", {})
    supercategory_to_color = taxonomy.get("supercategory_to_color", {})
    
    # Combine all colors
    all_colors = {}
    all_colors.update({f"CAT: {k}": v for k, v in category_to_color.items()})
    all_colors.update({f"SUP: {k}": v for k, v in supercategory_to_color.items()})
    
    if not all_colors:
        print("No colors found in taxonomy")
        return
    
    # Sort by name
    items = sorted(all_colors.items())
    n_items = len(items)
    
    # Calculate grid dimensions
    cols = 8
    rows = (n_items + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle('Taxonomy Color Grid (Categories and Supercategories)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Flatten axes array for easier indexing
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    for idx, (name, rgb) in enumerate(items):
        ax = axes_flat[idx]
        hex_color = rgb_to_hex(rgb)
        
        # Display color
        ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, 
                                       facecolor=hex_color, edgecolor='black', linewidth=1))
        
        # Add label
        label = name.replace("CAT: ", "").replace("SUP: ", "")
        if len(label) > 15:
            label = label[:12] + "..."
        
        # Determine text color (white or black based on brightness)
        brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
        text_color = 'white' if brightness < 128 else 'black'
        
        ax.text(0.5, 0.5, label, fontsize=7, ha='center', va='center',
               color=text_color, fontweight='bold', wrap=True)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_items, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved color grid to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize taxonomy colors")
    parser.add_argument(
        "--taxonomy",
        type=str,
        default="data_preparation_v2/taxonomy.json",
        help="Path to taxonomy.json file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["categories", "supercategories", "grid", "all"],
        default="all",
        help="Visualization mode: categories, supercategories, grid, or all"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for saving images (if not provided, displays interactively)"
    )
    
    args = parser.parse_args()
    
    taxonomy_path = Path(args.taxonomy)
    if not taxonomy_path.exists():
        print(f"ERROR: Taxonomy file not found: {taxonomy_path}")
        return
    
    # Load taxonomy
    print(f"Loading taxonomy from: {taxonomy_path}")
    taxonomy = load_taxonomy(taxonomy_path)
    
    # Determine output paths
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Visualize based on mode
    if args.mode in ["categories", "all"]:
        output_path = output_dir / "taxonomy_categories.png" if output_dir else None
        visualize_categories(taxonomy, output_path)
    
    if args.mode in ["supercategories", "all"]:
        output_path = output_dir / "taxonomy_supercategories.png" if output_dir else None
        visualize_supercategories(taxonomy, output_path)
    
    if args.mode in ["grid", "all"]:
        output_path = output_dir / "taxonomy_color_grid.png" if output_dir else None
        visualize_color_grid(taxonomy, output_path)
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

