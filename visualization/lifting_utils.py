#!/usr/bin/env python3

import json
import re
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Union, Tuple, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

# --- Constants from your scripts ---
# Default height slice - use much wider range to capture real height variation
DEFAULT_HEIGHT_RANGE = (-2.0, 5.0) 
# Default background color
DEFAULT_BG_COLOR = (240, 240, 240) 
# "wall" category ID from taxonomy - actually 4003 in the parquet files
WALL_CATEGORY_ID = 4003 


def create_zmap(
    taxonomy_path: Union[str, Path],
    parquet_files: List[Union[str, Path]],
    output_path: Union[str, Path],
    verbose: bool = True
):
    taxonomy_path = Path(taxonomy_path)
    output_path = Path(output_path)
    
    if verbose:
        print(f"Loading taxonomy from {taxonomy_path}...")
    with open(taxonomy_path, 'r') as f:
        taxonomy = json.load(f)
    
    # Get the map of SEMANTIC ID -> LAYOUT COLOR
    id_to_color = taxonomy.get("id2color")
    if not id_to_color:
        raise ValueError("Taxonomy file is missing 'id2color' map.")

    # This dict will store { "semantic_id_str": [z1, z2, ...] }
    z_data_by_id = {str(id_val): [] for id_val in id_to_color.keys()}
    
    # Also add wall category 2053 to ensure it's included
    z_data_by_id["2053"] = []
    
    # Track min/max heights from actual data (for fallback)
    all_heights = []
    
    if verbose:
        print(f"Processing {len(parquet_files)} parquet files...")
        
    iterator = parquet_files
    if verbose:
        iterator = tqdm(parquet_files, desc="Analyzing point clouds")

    for file_path in iterator:
        try:
            # Load the required semantic IDs and y value (y is actually the height/z-axis)
            df = pd.read_parquet(file_path, columns=['y', 'category_id', 'super_id'])
            
            # Collect all heights for min/max calculation (for fallback)
            # Note: y-axis is actually the vertical height in this coordinate system
            all_heights.extend(df['y'].tolist())
            
            # --- Apply "super categories + wall" logic ---
            wall_count = 0
            total_points = 0
            unique_cats = set()
            
            for row in df.itertuples(index=False):
                z = row.y  # y is actually the height/z-axis
                cat_id = row.category_id
                super_id = row.super_id
                total_points += 1
                unique_cats.add(cat_id)

                semantic_id_str = None
                if cat_id == 4003:  # 4003 is wall in parquet files
                    semantic_id_str = "2053"  # Map to 2053 for color lookup
                    wall_count += 1
                    # Wall point found
                else:
                    semantic_id_str = str(super_id)
                
                # Store the z-value if the ID is one we have a color for
                if semantic_id_str in z_data_by_id:
                    z_data_by_id[semantic_id_str].append(z)
            
            if verbose and wall_count > 0:
                print(f"Found {wall_count} wall points in {file_path}")
                        
        except Exception as e:
            if verbose:
                print(f"Warning: Could not process {file_path}. Error: {e}")

    # --- Compute Stats --- [MODIFIED SECTION]
    
    # Find floor and ceiling height based on the 'wall' category (ID 2053)
    # Walls define the min/max of the entire scene, handling outliers robustly
    floor_z_values = z_data_by_id.get("2053")

    if not floor_z_values:
        if verbose:
            print("Error: No 'wall' (ID 2053) points found. Cannot determine floor height.")
            print("Using global min as a fallback, but this may be incorrect.")
        # Fallback to original (problematic) logic if no wall points exist
        floor_height = min(all_heights) if all_heights else 0.0
        scene_max_height = max(all_heights) if all_heights else 5.0
    else:
        # Use walls to define scene bounds, handling outliers with percentiles
        wall_array = np.array(floor_z_values)
        # Use 1st percentile for floor (robust to outliers) and 99th percentile for ceiling
        floor_height = float(np.percentile(wall_array, 1))
        scene_max_height = float(np.percentile(wall_array, 99))
        
        if verbose:
            wall_min = float(np.min(wall_array))
            wall_max = float(np.max(wall_array))
            print(f"Wall heights: min={wall_min:.2f}, max={wall_max:.2f}")
            print(f"Using robust floor height (1st percentile of walls): {floor_height:.2f}")
            print(f"Using robust ceiling height (99th percentile of walls): {scene_max_height:.2f}")
    
    # Ensure all heights are within wall-defined bounds
    # Clip any outliers that go beyond the scene bounds defined by walls
    scene_min_height = floor_height  # Floor is the minimum reference
    
    zmap = {}
    missing_colors = []
    for semantic_id_str, z_list in z_data_by_id.items():
        if not z_list:
            if verbose:
                print(f"  Warning: Semantic ID {semantic_id_str} has no data points")
            continue
            
        # Get the corresponding layout color first - skip if no color mapping
        color_rgb = id_to_color.get(semantic_id_str)
        if not color_rgb:
            if verbose:
                missing_colors.append(semantic_id_str)
            continue
            
        z_array = np.array(z_list)
        # Clip heights to wall-defined scene bounds (handle outliers)
        z_array_clipped = np.clip(z_array, scene_min_height, scene_max_height)
        
        # Make heights relative to floor (floor becomes 0)
        z_array_relative = z_array_clipped - floor_height
        
        # Check for invalid/empty array after processing
        if len(z_array_relative) == 0:
            if verbose:
                print(f"  Warning: Semantic ID {semantic_id_str} has no valid points after processing")
            continue
        
        # Use percentiles instead of min/max to handle outliers robustly
        # This prevents categories from "exploding" due to a few extreme outliers
        z_p5 = float(np.percentile(z_array_relative, 5))   # 5th percentile instead of min
        z_p95 = float(np.percentile(z_array_relative, 95))  # 95th percentile instead of max
        z_mean = float(np.mean(z_array_relative))
        z_std = float(np.std(z_array_relative))
        
        # Check for NaN or invalid values
        if np.isnan(z_p5) or np.isnan(z_p95) or np.isnan(z_mean) or np.isnan(z_std):
            if verbose:
                print(f"  Warning: Semantic ID {semantic_id_str} produced NaN values (array shape: {z_array_relative.shape})")
            # Fall back to min/max if percentiles fail
            z_p5 = float(np.min(z_array_relative))
            z_p95 = float(np.max(z_array_relative))
            if np.isnan(z_p5) or np.isnan(z_p95):
                if verbose:
                    print(f"  Error: Cannot compute stats for semantic ID {semantic_id_str}, skipping")
                continue
        
        # Clamp min/max to be reasonable relative to mean (within 1.5 std for tighter bounds)
        # This ensures the range isn't too wide due to outliers
        # Use 1.5 std for conservative bounds to prevent exploding
        z_min = max(z_p5, z_mean - 1.5 * z_std)  # At least mean - 1.5*std, but respect 5th percentile
        z_max = min(z_p95, z_mean + 1.5 * z_std)  # At most mean + 1.5*std, but respect 95th percentile
        
        # Additional safeguard: use even tighter bounds if range is still too wide
        # For categories that span full height, use tighter percentiles
        if z_max - z_min > 1.0:  # If range > 1.0, it's probably too wide
            z_p10 = float(np.percentile(z_array_relative, 10))
            z_p90 = float(np.percentile(z_array_relative, 90))
            z_min = max(z_p10, z_mean - 1.0 * z_std)
            z_max = min(z_p90, z_mean + 1.0 * z_std)
            
            # Final check: if still too wide, use even tighter bounds
            if z_max - z_min > 1.0:
                z_min = max(z_p10, z_mean - 0.75 * z_std)
                z_max = min(z_p90, z_mean + 0.75 * z_std)
        
        # Ensure min <= max
        if z_min > z_max:
            z_min, z_max = min(z_p5, z_p95), max(z_p5, z_p95)
        
        # Final NaN check
        if np.isnan(z_min) or np.isnan(z_max) or np.isnan(z_mean) or np.isnan(z_std):
            if verbose:
                print(f"  Error: Semantic ID {semantic_id_str} still has NaN values after processing, skipping")
            continue
        
        # Log if outliers were clipped
        if verbose and len(z_array) > 0:
            n_clipped = np.sum((z_array < scene_min_height) | (z_array > scene_max_height))
            original_min = float(np.min(z_array_relative))
            original_max = float(np.max(z_array_relative))
            if n_clipped > 0 or abs(original_min - z_min) > 0.01 or abs(original_max - z_max) > 0.01:
                print(f"  Semantic ID {semantic_id_str}: clipped {n_clipped} outliers, "
                      f"bounds: [{original_min:.2f}, {original_max:.2f}] -> [{z_min:.2f}, {z_max:.2f}]")
        
        # Use a "rgb(r,g,b)" string as the key for the JSON
        color_key = f"rgb({color_rgb[0]},{color_rgb[1]},{color_rgb[2]})"
        zmap[color_key] = {
            "min": z_min,
            "max": z_max,
            "mean": z_mean,
            "std": z_std,
            "semantic_id": int(semantic_id_str),
            "samples": len(z_list)
        }
    
    # Report missing color mappings
    if verbose and missing_colors:
        print(f"Warning: {len(missing_colors)} semantic IDs have data but no color mapping: {missing_colors}")

    # --- Save Z-Map ---
    output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists
    with open(output_path, 'w') as f:
        json.dump(zmap, f, indent=2)


def lift_layout(
    layout_path: Union[str, Path],
    zmap_path: Union[str, Path],
    bg_color: Tuple[int, int, int] = DEFAULT_BG_COLOR,
    point_density: float = 1.0,
    height_samples: int = 5,
    force_scale: str = None  # "room" or "scene", None for auto-detection
) -> np.ndarray:

    layout_path = Path(layout_path)
    zmap_path = Path(zmap_path)

    # 1. Load layout image
    img = Image.open(layout_path).convert('RGB')
    layout_np = np.array(img)
    H, W, _ = layout_np.shape

    # 2. Load z-map
    with open(zmap_path, 'r') as f:
        zmap = json.load(f)

    # 3. Find all non-background pixel coordinates
    bg_mask = np.all(layout_np == bg_color, axis=-1)
    fg_coords_y, fg_coords_x = np.where(~bg_mask)

    num_fg_pixels = len(fg_coords_y)
    if num_fg_pixels == 0:
        print("Warning: Layout image appears to be empty (all background).")
        return np.empty((0, 6))

    # 4. Calculate coordinate scale based on force_scale parameter
    # Room footprint: 3m x 3m, Scene footprint: 30m x 30m
    # Image is 512x512 pixels, so we normalize to footprint size
    if force_scale == "scene":
        coord_scale = 30.0 / 512  # Scene: 512 pixels = 30m
    elif force_scale == "room":
        coord_scale = 3.0 / 512   # Room: 512 pixels = 3m
    else:
        # Default to room scale if not specified
        coord_scale = 3.0 / 512   # Room: 512 pixels = 3m

    # 5. Handle point density sampling
    if point_density < 1.0:
        num_samples = int(num_fg_pixels * point_density)
        sample_indices = np.random.choice(num_fg_pixels, num_samples, replace=False)
        fg_coords_y = fg_coords_y[sample_indices]
        fg_coords_x = fg_coords_x[sample_indices]

    lifted_points = []
    
    # 5. Create EXTRUDED point clouds by sampling multiple heights per pixel
    for y, x in zip(fg_coords_y, fg_coords_x):
        # Get pixel color
        r, g, b = layout_np[y, x]
        
        # Find Z statistics for this color
        color_key = f"rgb({r},{g},{b})"
        stats = zmap.get(color_key)
        
        if stats:
            # Coordinate scale already determined based on room/scene detection above
            
            # We reverse the Y-axis flip from stage3
            x_world = x * coord_scale
            y_world = ((H - 1) - y) * coord_scale
            
            # Create extrusion by randomly sampling heights from the distribution
            z_min = stats["min"]
            z_max = stats["max"]
            z_mean = stats["mean"]
            z_std = stats.get("std", 0.1)  # Use std if available, otherwise small default
            semantic_id = stats.get("semantic_id")
            
            # Special handling for walls (semantic_id 2053): sample uniformly from min to max
            # to properly represent floor-to-ceiling structure
            if semantic_id == 2053:
                # Walls should span from floor to ceiling uniformly
                heights = np.linspace(z_min, z_max, height_samples)
            else:
                # For other objects, sample around the mean with normal distribution
                # Safety clamp: Ensure bounds are reasonable relative to mean
                # This prevents "exploding" even if zmap has extreme outliers
                # Use tighter bounds: mean ± 1.5*std, but respect the provided min/max
                safe_min = max(z_min, z_mean - 1.5 * z_std)
                safe_max = min(z_max, z_mean + 1.5 * z_std)
                
                # If the range is still too wide, use even tighter bounds
                if safe_max - safe_min > 1.0:  # If range > 1.0, it's probably too wide
                    safe_min = max(z_min, z_mean - 1.0 * z_std)
                    safe_max = min(z_max, z_mean + 1.0 * z_std)
                
                # Ensure safe bounds are valid
                if safe_max <= safe_min:
                    safe_min, safe_max = z_min, z_max
                
                # Sample heights randomly to create actual point cloud (not layers)
                # Use normal distribution centered at mean with std, clamped to safe range
                if safe_max > safe_min:
                    # Sample from normal distribution to match the actual distribution
                    # Clamp to safe bounds to prevent exploding
                    heights = np.random.normal(z_mean, z_std, height_samples)
                    heights = np.clip(heights, safe_min, safe_max)
                else:
                    # If min == max (flat surface), use that single value
                    heights = np.full(height_samples, z_mean)
            
            # Create a point for each height sample to create the extrusion
            for height in heights:
                lifted_points.append([x_world, y_world, height, r, g, b])

    return np.array(lifted_points, dtype=np.float64)


def load_ply_as_points(ply_path: Union[str, Path]) -> np.ndarray:
    """Load a PLY file and return as numpy array of points [x,y,z,r,g,b]."""
    points = []
    with open(ply_path, 'r') as f:
        lines = f.readlines()
    
    # Find the start of vertex data
    vertex_start = None
    vertex_count = 0
    for i, line in enumerate(lines):
        if line.startswith('element vertex'):
            vertex_count = int(line.split()[-1])
        elif line.startswith('end_header'):
            vertex_start = i + 1
            break
    
    if vertex_start is None:
        raise ValueError("Could not find vertex data in PLY file")
    
    # Read vertex data
    for i in range(vertex_start, vertex_start + vertex_count):
        parts = lines[i].strip().split()
        if len(parts) >= 6:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
            points.append([x, y, z, r, g, b])
    
    return np.array(points, dtype=np.float64)


def plot_point_cloud_3d(points: np.ndarray, title: str = "3D Point Cloud"):
    """Plot the 3D point cloud using matplotlib (interactive display)."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates and colors
    x = points[:, 0]
    y = points[:, 1] 
    z = points[:, 2]
    colors = points[:, 3:6] / 255.0  # Normalize RGB to [0,1]
    
    # Plot points with their colors
    ax.scatter(x, y, z, c=colors, s=1, alpha=0.6)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()


def plot_point_cloud_3d_to_image(
    points: np.ndarray, 
    output_path: Union[str, Path],
    title: str = "3D Point Cloud",
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 100,
    axis_limits: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = None,
    layout_image: np.ndarray = None
) -> None:
    """Create 3D point cloud plot and save as image file."""
    if layout_image is not None:
        fig = plt.figure(figsize=(18, 8), dpi=dpi)
        ax_3d = fig.add_subplot(121, projection='3d')
        ax_2d = fig.add_subplot(122)
    else:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax_3d = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates and colors
    x = points[:, 0]
    y = points[:, 1] 
    z = points[:, 2]
    colors = points[:, 3:6] / 255.0  # Normalize RGB to [0,1]
    
    # Plot points with their colors
    ax_3d.scatter(x, y, z, c=colors, s=1, alpha=0.6)
    
    # Set labels and title
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title(title)
    
    # Set axis limits (fixed if provided, otherwise calculate)
    if axis_limits is not None:
        ax_3d.set_xlim(axis_limits[0])
        ax_3d.set_ylim(axis_limits[1])
        ax_3d.set_zlim(axis_limits[2])
    else:
        # Set equal aspect ratio
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
        ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
        ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add 2D layout image if provided
    if layout_image is not None:
        ax_2d.imshow(layout_image)
        ax_2d.set_title("2D Layout")
        ax_2d.axis('off')
    
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def save_point_cloud_as_ply(
    points: np.ndarray,
    output_ply_path: Union[str, Path]
):

    output_ply_path = Path(output_ply_path)
    output_ply_path.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists
    num_points = points.shape[0]
    
    # Create the PLY header
    header = f"""ply
                format ascii 1.0
                element vertex {num_points}
                property float x
                property float y
                property float z
                property uchar red
                property uchar green
                property uchar blue
                end_header
                """
                    
    # Format the data
    data = np.zeros((num_points, 6))
    data[:, :3] = points[:, :3] # x, y, z
    data[:, 3:] = points[:, 3:].astype(np.uint8) # r, g, b
    
    # Save the file
    with open(output_ply_path, 'w') as f:
        f.write(header)
        np.savetxt(f, data, fmt="%.6f %.6f %.6f %d %d %d")
    
    print(f"Saved lifted point cloud to {output_ply_path}")


# --- Main execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Z-Map and Layout Lifting Utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- create-zmap command ---
    parser_map = subparsers.add_parser(
        "create-zmap", 
        help="Analyze parquet files to create the zmap.json."
    )
    parser_map.add_argument(
        "--taxonomy", required=True, help="Path to taxonomy.json"
    )
    parser_map.add_argument(
        "--parquets", required=True, nargs="+", 
        help="List of parquet files (or glob pattern like 'data/*.parquet')"
    )
    parser_map.add_argument(
        "--output", required=True, help="Path to save the zmap.json"
    )

    # --- lift-layout command ---
    parser_lift = subparsers.add_parser(
        "lift-layout", 
        help="Lift a 2D layout.png to a 3D .ply point cloud."
    )
    
    # --- plot-3d command ---
    parser_plot = subparsers.add_parser(
        "plot-3d",
        help="Plot a 3D point cloud from a .ply file"
    )
    parser_lift.add_argument(
        "--layout", required=True, help="Path to the 2D layout.png"
    )
    parser_lift.add_argument(
        "--zmap", required=True, help="Path to the zmap.json"
    )
    parser_lift.add_argument(
        "--output", required=True, help="Path to the output .ply file"
    )
    parser_lift.add_argument(
        "--density", type=float, default=1.0, 
        help="Point density (1.0 = all pixels, 0.5 = 50% random sample)"
    )
    parser_lift.add_argument(
        "--height-samples", type=int, default=5,
        help="Number of height samples per pixel for extrusion (default: 5)"
    )
    
    parser_plot.add_argument(
        "--ply", required=True, help="Path to the .ply file to plot"
    )
    parser_plot.add_argument(
        "--title", default="3D Point Cloud", help="Title for the plot"
    )

    args = parser.parse_args()

    if args.command == "create-zmap":
        # Handle glob pattern for parquet files
        if len(args.parquets) == 1 and '*' in args.parquets[0]:
            import glob
            parquet_files = glob.glob(args.parquets[0], recursive=True)
            if not parquet_files:
                print(f"Error: No files found matching glob pattern '{args.parquets[0]}'")
                exit(1)
        else:
            parquet_files = args.parquets
            
        print(f"Found {len(parquet_files)} parquet files to process.")

        create_zmap(
            taxonomy_path=args.taxonomy,
            parquet_files=parquet_files,
            output_path=args.output
        )

    elif args.command == "lift-layout":
        print(f"Lifting {args.layout} using {args.zmap}...")
        lifted_points = lift_layout(
            layout_path=args.layout,
            zmap_path=args.zmap,
            point_density=args.density,
            height_samples=args.height_samples
        )
        
        if lifted_points.shape[0] > 0:
            save_point_cloud_as_ply(
                points=lifted_points,
                output_ply_path=args.output
            )
        else:
            print("No points were generated. Output .ply file not saved.")

    elif args.command == "plot-3d":
        print(f"Loading point cloud from {args.ply}...")
        # Load PLY file and convert to numpy array
        points = load_ply_as_points(args.ply)
        print(f"Loaded {len(points)} points")
        plot_point_cloud_3d(points, args.title)