#!/usr/bin/env python3
"""
Visualize 3D-FRONT scenes with isometric plots: textured and semantic side-by-side.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from utils.semantic_utils import Taxonomy
except ImportError:
    # Try absolute import if relative doesn't work
    import importlib.util
    spec = importlib.util.spec_from_file_location("semantic_utils", Path(__file__).parent / "utils" / "semantic_utils.py")
    semantic_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(semantic_utils)
    Taxonomy = semantic_utils.Taxonomy


def load_point_cloud(scene_dir: Path, scene_id: str, format: str = "parquet"):
    """Load point cloud from parquet or csv."""
    if format == "parquet":
        file_path = scene_dir / f"{scene_id}_sem_pointcloud.parquet"
        return pd.read_parquet(file_path)
    else:
        file_path = scene_dir / f"{scene_id}_sem_pointcloud.csv"
        return pd.read_csv(file_path)


def create_category_colormap(taxonomy: Taxonomy, categories: List[str]) -> Dict[str, np.ndarray]:
    """Create a consistent color mapping for categories."""
    unique_cats = sorted(set(categories))
    
    # Use a colorblind-friendly palette
    base_colors = plt.cm.tab20c(np.linspace(0, 1, 20))
    extended_colors = plt.cm.Set3(np.linspace(0, 1, 12))
    all_colors = np.vstack([base_colors, extended_colors])
    
    color_map = {}
    for i, cat in enumerate(unique_cats):
        color_map[cat] = all_colors[i % len(all_colors)][:3]  # RGB only
    
    # Special colors for architectural elements
    if 'floor' in color_map:
        color_map['floor'] = np.array([0.8, 0.8, 0.8])
    if 'wall' in color_map:
        color_map['wall'] = np.array([0.9, 0.9, 0.85])
    
    return color_map


def setup_isometric_view(ax: Axes3D, bounds: Dict):
    """Configure axis for isometric view."""
    # Set equal aspect ratio
    x_range = bounds['max'][0] - bounds['min'][0]
    y_range = bounds['max'][1] - bounds['min'][1]
    z_range = bounds['max'][2] - bounds['min'][2]
    
    max_range = max(x_range, y_range, z_range)
    mid_x = (bounds['max'][0] + bounds['min'][0]) / 2
    mid_y = (bounds['max'][1] + bounds['min'][1]) / 2
    mid_z = (bounds['max'][2] + bounds['min'][2]) / 2
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    # Isometric viewing angle (35.264° elevation, 45° azimuth)
    ax.view_init(elev=25, azim=45)
    
    # Clean up axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.grid(True, alpha=0.3)


def downsample_points(df: pd.DataFrame, max_points: int = 50000) -> pd.DataFrame:
    """Downsample point cloud for faster rendering."""
    if len(df) <= max_points:
        return df
    
    indices = np.random.choice(len(df), max_points, replace=False)
    return df.iloc[indices]


def plot_textured_view(ax: Axes3D, df: pd.DataFrame, bounds: Dict, point_size: float = 5.0):
    """Plot textured (RGB) view."""
    # Normalize RGB values
    colors = np.column_stack([df['r'], df['g'], df['b']]) / 255.0
    
    ax.scatter(df['x'], df['y'], df['z'], 
               c=colors, 
               s=point_size, 
               alpha=0.8,
               edgecolors='none')
    
    setup_isometric_view(ax, bounds)
    ax.set_title('Textured View', fontsize=14, fontweight='bold')


def plot_semantic_view(ax: Axes3D, df: pd.DataFrame, bounds: Dict, 
                       taxonomy: Taxonomy, point_size: float = 1.0):
    """Plot semantic (category-colored) view."""
    # Create color mapping
    color_map = create_category_colormap(taxonomy, df['category'].tolist())
    
    # Assign colors to each point
    colors = np.array([color_map[cat] for cat in df['category']])
    
    ax.scatter(df['x'], df['y'], df['z'], 
               c=colors, 
               s=point_size, 
               alpha=0.8,
               edgecolors='none')
    
    setup_isometric_view(ax, bounds)
    ax.set_title('Semantic View', fontsize=14, fontweight='bold')
    
    # Add legend for top categories
    unique_cats = df['category'].value_counts().head(10).index.tolist()
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor=color_map[cat], 
                                   markersize=8, label=cat)
                       for cat in unique_cats if cat in color_map]
    
    ax.legend(handles=legend_elements, loc='upper left', 
              bbox_to_anchor=(1.05, 1), fontsize=8)


def visualize_scene(scene_dir: Path, scene_id: str, taxonomy: Taxonomy,
                   output_path: Path = None, max_points: int = 50000,
                   point_size: float = 1.0, format: str = "parquet",
                   figsize: Tuple[int, int] = (16, 8), dpi: int = 150):
    """Create side-by-side isometric visualization."""
    
    # Load data
    print(f"Loading scene {scene_id}...")
    df = load_point_cloud(scene_dir, scene_id, format)
    
    # Load scene info for bounds
    scene_info_path = scene_dir / f"{scene_id}_scene_info.json"
    with open(scene_info_path, 'r') as f:
        scene_info = json.load(f)
    
    # Downsample if needed
    if len(df) > max_points:
        print(f"Downsampling from {len(df)} to {max_points} points...")
        df = downsample_points(df, max_points)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Textured view (left)
    ax1 = fig.add_subplot(121, projection='3d')
    plot_textured_view(ax1, df, scene_info['bounds'], point_size)
    
    # Semantic view (right)
    ax2 = fig.add_subplot(122, projection='3d')
    plot_semantic_view(ax2, df, scene_info['bounds'], taxonomy, point_size)
    
    # Add main title
    fig.suptitle(f'Scene: {scene_id}', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate isometric visualizations of 3D-FRONT scenes"
    )
    parser.add_argument("--scene_dir", type=str, required=True,
                       help="Directory containing scene point clouds and metadata")
    parser.add_argument("--scene_id", type=str, required=True,
                       help="Scene ID to visualize")
    parser.add_argument("--taxonomy", type=str, required=True,
                       help="Path to taxonomy file")
    parser.add_argument("--output", type=str, default=None,
                       help="Output image path (if not specified, displays interactively)")
    parser.add_argument("--format", type=str, default="parquet",
                       choices=["parquet", "csv"],
                       help="Point cloud format")
    parser.add_argument("--max_points", type=int, default=50000,
                       help="Maximum points to render (for performance)")
    parser.add_argument("--point_size", type=float, default=1.0,
                       help="Size of points in plot")
    parser.add_argument("--figsize", type=int, nargs=2, default=[16, 8],
                       help="Figure size (width height)")
    parser.add_argument("--dpi", type=int, default=150,
                       help="Output resolution")
    
    args = parser.parse_args()
    
    # Load taxonomy
    taxonomy = Taxonomy(Path(args.taxonomy))
    
    # Visualize scene
    output_path = Path(args.output) if args.output else None
    
    visualize_scene(
        scene_dir=Path(args.scene_dir),
        scene_id=args.scene_id,
        taxonomy=taxonomy,
        output_path=output_path,
        max_points=args.max_points,
        point_size=args.point_size,
        format=args.format,
        figsize=tuple(args.figsize),
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()