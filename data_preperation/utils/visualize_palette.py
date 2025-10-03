#!/usr/bin/env python3
"""
visualize_palette.py

Generates one PNG per super-category showing its categories and assigned colors,
plus an overview PNG showing all super-categories and special colors (wall/floor/ceiling),
plus a "chosen labels/colors" PNG showing the actual used palette.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np


# Set seaborn style for better aesthetics
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)


def visualize_chosen_palette(taxonomy_path: str, out_dir: str):
    """Generate a plot showing the actual used palette: all super-categories + wall."""
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    super2id = taxonomy["super2id"]
    id2color = taxonomy["id2color"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get all super-categories
    sorted_supers = sorted(super2id.items(), key=lambda x: x[1])
    
    # Start with super-categories
    chosen_items = list(sorted_supers)
    
    # Add wall - try multiple possible keys
    wall_id = None
    if "wall_id" in taxonomy:
        wall_id = taxonomy["wall_id"]
    elif "wall" in taxonomy:
        wall_id = taxonomy["wall"]
    
    # Also check if wall is in category2id
    if wall_id is None and "category2id" in taxonomy:
        category2id = taxonomy["category2id"]
        if "wall" in category2id:
            wall_id = category2id["wall"]
    
    # ALWAYS add wall to the list
    if wall_id is not None:
        chosen_items.append(("wall", wall_id))
        print(f"[DEBUG] Added wall with ID: {wall_id}")
    else:
        print(f"[WARNING] Wall ID not found in taxonomy!")
        print(f"[DEBUG] Available taxonomy keys: {list(taxonomy.keys())}")
    
    n_items = len(chosen_items)
    print(f"[INFO] Chosen palette has {n_items} items (should be {len(sorted_supers)} super-categories + 1 wall)")
    
    # Calculate grid dimensions
    cols = min(4, n_items)
    rows = int(np.ceil(n_items / cols))
    
    # Create figure with seaborn styling
    fig, ax = plt.subplots(figsize=(14, max(8, rows * 2.5)))
    ax.axis("off")
    ax.set_aspect('equal')
    
    # Patch dimensions and spacing
    patch_width = 2.5
    patch_height = 1.8
    h_spacing = 3.5
    v_spacing = 3.0
    
    for idx, (name, item_id) in enumerate(chosen_items):
        row, col = divmod(idx, cols)
        x = col * h_spacing
        y = -(row * v_spacing)
        
        # Get color
        item_color = tuple(c / 255 for c in id2color.get(str(item_id), [127, 127, 127]))
        
        # Check if this is wall
        is_wall = name == "wall"
        
        # Draw patch with shadow effect (no black outline)
        shadow = mpatches.Rectangle(
            (x + 0.05, y - 0.05), patch_width, patch_height,
            facecolor='gray', alpha=0.3, edgecolor='none'
        )
        ax.add_patch(shadow)
        
        rect = mpatches.Rectangle(
            (x, y), patch_width, patch_height,
            facecolor=item_color, edgecolor='none'
        )
        ax.add_patch(rect)
        
        # Add text with better positioning
        display_name = name.upper() if is_wall else name
        fontsize = 10 if is_wall else 11
        
        ax.text(
            x + patch_width / 2, y + patch_height / 2,
            display_name,
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='none')
        )
        
        # Add ID label
        label_text = f"ID: {item_id}"
        if is_wall:
            label_text = f"(Wall) {label_text}"
        
        ax.text(
            x + patch_width / 2, y - 0.4,
            label_text,
            ha="center", va="top",
            fontsize=8 if is_wall else 9, 
            style='italic', color='#333333'
        )
    
    # Set limits with padding
    ax.set_xlim(-0.5, cols * h_spacing - 0.5)
    ax.set_ylim(-(rows * v_spacing + 0.5), patch_height + 1.0)
    
    plt.title("Chosen Labels/Colors (Used Palette)", fontsize=16, fontweight='bold', pad=20)
    
    out_path = out_dir / "palette_chosen_labels_colors.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()
    print(f"[INFO] Saved chosen palette: {out_path}")


def visualize_super_overview(taxonomy_path: str, out_dir: str):
    """Generate an overview plot showing all super-categories and special colors."""
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    super2id = taxonomy["super2id"]
    id2color = taxonomy["id2color"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sort super-categories by ID for consistent ordering
    sorted_supers = sorted(super2id.items(), key=lambda x: x[1])
    
    # Add special categories (wall/floor/ceiling) if they exist
    special_categories = []
    special_mapping = {
        "wall": taxonomy.get("wall_id"),
        "floor": taxonomy.get("floor_id"),
        "ceiling": taxonomy.get("ceiling_id")
    }
    
    # Also check alternative keys
    if special_mapping["wall"] is None:
        special_mapping["wall"] = taxonomy.get("wall")
    if special_mapping["floor"] is None:
        special_mapping["floor"] = taxonomy.get("floor")
    if special_mapping["ceiling"] is None:
        special_mapping["ceiling"] = taxonomy.get("ceiling")
    
    # Check category2id as fallback
    if "category2id" in taxonomy:
        category2id = taxonomy["category2id"]
        for name in ["wall", "floor", "ceiling"]:
            if special_mapping[name] is None and name in category2id:
                special_mapping[name] = category2id[name]
    
    for name, cat_id in special_mapping.items():
        if cat_id is not None:
            special_categories.append((name, cat_id))
    
    # Combine all items to display
    all_items = sorted_supers + special_categories
    n_items = len(all_items)
    
    # Calculate grid dimensions
    cols = min(4, n_items)
    rows = int(np.ceil(n_items / cols))
    
    # Create figure with seaborn styling
    fig, ax = plt.subplots(figsize=(14, max(8, rows * 2.5)))
    ax.axis("off")
    ax.set_aspect('equal')
    
    # Patch dimensions and spacing
    patch_width = 2.5
    patch_height = 1.8
    h_spacing = 3.5
    v_spacing = 3.0
    
    for idx, (name, item_id) in enumerate(all_items):
        row, col = divmod(idx, cols)
        x = col * h_spacing
        y = -(row * v_spacing)
        
        # Get color
        item_color = tuple(c / 255 for c in id2color.get(str(item_id), [127, 127, 127]))
        
        # Check if this is a special category
        is_special = name in ["wall", "floor", "ceiling"]
        
        # Draw patch with shadow effect (no black outline)
        shadow = mpatches.Rectangle(
            (x + 0.05, y - 0.05), patch_width, patch_height,
            facecolor='gray', alpha=0.3, edgecolor='none'
        )
        ax.add_patch(shadow)
        
        rect = mpatches.Rectangle(
            (x, y), patch_width, patch_height,
            facecolor=item_color, edgecolor='none'
        )
        ax.add_patch(rect)
        
        # Add text with better positioning
        # Add special indicator for wall/floor/ceiling
        display_name = name.upper() if is_special else name
        fontsize = 11 if not is_special else 10
        
        ax.text(
            x + patch_width / 2, y + patch_height / 2,
            display_name,
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='none')
        )
        
        # Add label for special categories
        label_text = f"ID: {item_id}"
        if is_special:
            label_text = f"(Special) {label_text}"
        
        ax.text(
            x + patch_width / 2, y - 0.4,
            label_text,
            ha="center", va="top",
            fontsize=8 if is_special else 9, 
            style='italic', color='#333333'
        )
    
    # Set limits with padding - added top padding to prevent cropping
    ax.set_xlim(-0.5, cols * h_spacing - 0.5)
    ax.set_ylim(-(rows * v_spacing + 0.5), patch_height + 1.0)
    
    plt.title("Super-Categories & Special Colors Overview", fontsize=16, fontweight='bold', pad=20)
    
    out_path = out_dir / "palette_super_categories_overview.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()
    print(f"[INFO] Saved super-categories overview: {out_path}")


def visualize_per_super(taxonomy_path: str, out_dir: str):
    """Generate one visualization per super-category."""
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    super2id = taxonomy["super2id"]
    category2id = taxonomy["category2id"]
    category2super = taxonomy["category2super"]
    id2color = taxonomy["id2color"]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for supercat, sid in sorted(super2id.items(), key=lambda x: x[1]):
        # Categories belonging to this super
        categories = sorted([c for c, s in category2super.items() if s == supercat])
        
        if not categories:
            continue
        
        # Calculate dynamic figure size based on number of categories
        n_cats = len(categories)
        cols = 5
        rows = int(np.ceil(n_cats / cols))
        fig_height = max(8, 4 + rows * 2.5)
        
        fig, ax = plt.subplots(figsize=(14, fig_height))
        ax.axis("off")
        ax.set_aspect('equal')

        # Draw super-category block with enhanced styling
        scol = tuple(c / 255 for c in id2color.get(str(sid), [127, 127, 127]))
        
        # Shadow for super block
        super_shadow = mpatches.Rectangle(
            (0.05, 0.05), 3.5, 1.8,
            facecolor='gray', alpha=0.3, edgecolor='none'
        )
        ax.add_patch(super_shadow)
        
        # Main super block (no black outline)
        super_rect = mpatches.Rectangle(
            (0, 0.1), 3.5, 1.8,
            facecolor=scol, edgecolor='none'
        )
        ax.add_patch(super_rect)
        
        ax.text(
            1.75, 1.0,
            f"{supercat}",
            ha="center", va="center",
            fontsize=14, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9, edgecolor='none')
        )
        ax.text(
            1.75, -0.3,
            f"Super-Category ID: {sid}",
            ha="center", va="top",
            fontsize=10, style='italic', color='#444444'
        )

        # Draw categories in a grid with better spacing
        patch_width = 2.0
        patch_height = 1.5
        h_spacing = 2.8
        v_spacing = 2.5
        
        for i, cat in enumerate(categories):
            cid = category2id[cat]
            ccol = tuple(c / 255 for c in id2color.get(str(cid), [127, 127, 127]))
            
            row, col = divmod(i, cols)
            x = col * h_spacing
            y = -(row + 2) * v_spacing
            
            # Shadow effect
            shadow = mpatches.Rectangle(
                (x + 0.05, y - 0.05), patch_width, patch_height,
                facecolor='gray', alpha=0.2, edgecolor='none'
            )
            ax.add_patch(shadow)
            
            # Category patch
            rect = mpatches.Rectangle(
                (x, y), patch_width, patch_height,
                facecolor=ccol, edgecolor='black', linewidth=1.5
            )
            ax.add_patch(rect)
            
            # Category name (with wrapping for long names)
            cat_display = cat if len(cat) <= 20 else cat[:17] + "..."
            ax.text(
                x + patch_width / 2, y + patch_height / 2,
                cat_display,
                ha="center", va="center",
                fontsize=9, fontweight="semibold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7, edgecolor='none')
            )
            
            # Category ID below patch
            ax.text(
                x + patch_width / 2, y - 0.3,
                f"ID: {cid}",
                ha="center", va="top",
                fontsize=8, color='#555555'
            )

        # Set limits with proper padding
        ax.set_xlim(-0.5, cols * h_spacing - 0.5)
        ax.set_ylim(-(rows + 2) * v_spacing - 1, 2.5)
        
        plt.title(
            f"Category Palette for: {supercat}",
            fontsize=15, fontweight='bold', pad=20
        )

        out_path = out_dir / f"palette_{supercat.replace('/', '_').replace(' ', '_')}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor='white')
        plt.close()
        print(f"[INFO] Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize taxonomy color palette (one PNG per super + overview)."
    )
    parser.add_argument("taxonomy", help="Path to taxonomy.json")
    parser.add_argument("--out-dir", required=True, help="Directory to save palette visualizations")
    args = parser.parse_args()

    # Generate chosen labels/colors (actual used palette)
    visualize_chosen_palette(args.taxonomy, args.out_dir)
    
    # Generate super-categories overview
    visualize_super_overview(args.taxonomy, args.out_dir)
    
    # Generate individual super-category visualizations
    visualize_per_super(args.taxonomy, args.out_dir)