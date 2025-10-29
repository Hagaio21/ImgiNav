#!/usr/bin/env python3
"""
Utility script to:
1. Create a "z-map" (z-distribution) from semantic point clouds.
2. "Lift" a 2D segmented layout image into a 3D point cloud using the z-map.

Based on the processing pipeline from stage1, stage2, and stage3 scripts.
"""

import json
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Union, Tuple, Dict
from tqdm import tqdm
import argparse

# --- Constants from your scripts ---
# Default height slice
DEFAULT_HEIGHT_RANGE = (0.1, 1.8) 
# Default background color
DEFAULT_BG_COLOR = (240, 240, 240) 
# "wall" category ID from taxonomy
WALL_CATEGORY_ID = 2053 


def create_zmap(
    taxonomy_path: Union[str, Path],
    parquet_files: List[Union[str, Path]],
    output_path: Union[str, Path],
    height_range: Tuple[float, float] = DEFAULT_HEIGHT_RANGE,
    verbose: bool = True
):
    """
    Analyzes Z distributions for each semantic class from parquet files.
    
    Creates a zmap.json mapping semantic layout colors (from taxonomy)
    to the mean and std of their z-values.
    
    This function implements the "super categories + wall" logic.
    
    Args:
        taxonomy_path: Path to your taxonomy.json file.
        parquet_files: A list of paths to your .parquet point cloud files.
        output_path: Path to save the resulting zmap.json.
        height_range: The (min_z, max_z) tuple to filter points, same as your
                      projection slice.
        verbose: Print progress.
    """
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
    
    if verbose:
        print(f"Processing {len(parquet_files)} parquet files...")
        
    iterator = parquet_files
    if verbose:
        iterator = tqdm(parquet_files, desc="Analyzing point clouds")

    for file_path in iterator:
        try:
            # Load the required semantic IDs and z value
            df = pd.read_parquet(file_path, columns=['z', 'category_id', 'super_id'])
            
            # Filter by the same height range used for 2D projection
            df = df[
                (df['z'] >= height_range[0]) & (df['z'] <= height_range[1])
            ]
            
            # --- Apply "super categories + wall" logic ---
            for row in df.itertuples(index=False):
                z = row.z
                cat_id = row.category_id
                super_id = row.super_id

                semantic_id_str = None
                if cat_id == WALL_CATEGORY_ID:
                    semantic_id_str = str(WALL_CATEGORY_ID)
                else:
                    semantic_id_str = str(super_id)
                
                # Store the z-value if the ID is one we have a color for
                if semantic_id_str in z_data_by_id:
                    z_data_by_id[semantic_id_str].append(z)
                    
        except Exception as e:
            if verbose:
                print(f"Warning: Could not process {file_path}. Error: {e}")

    # --- Compute Stats ---
    if verbose:
        print("Computing Z-statistics for each class...")
        
    zmap = {}
    for semantic_id_str, z_list in z_data_by_id.items():
        if not z_list:
            continue
            
        z_array = np.array(z_list)
        z_mean = float(np.mean(z_array))
        z_std = float(np.std(z_array))
        
        # Get the corresponding layout color
        color_rgb = id_to_color.get(semantic_id_str)
        if color_rgb:
            # Use a "rgb(r,g,b)" string as the key for the JSON
            color_key = f"rgb({color_rgb[0]},{color_rgb[1]},{color_rgb[2]})"
            zmap[color_key] = {
                "mean": z_mean,
                "std": z_std,
                "semantic_id": int(semantic_id_str),
                "samples": len(z_list)
            }

    # --- Save Z-Map ---
    if verbose:
        print(f"Saving z-map with {len(zmap)} classes to {output_path}")
    from common.utils import write_json
    write_json(zmap, output_path)

    if verbose:
        print("Z-Map creation complete.")


def lift_layout(
    layout_path: Union[str, Path],
    zmap_path: Union[str, Path],
    bg_color: Tuple[int, int, int] = DEFAULT_BG_COLOR,
    point_density: float = 1.0
) -> np.ndarray:
    """
    Lifts a 2D segmented layout PNG to a 3D point cloud using a z-map.
    (This function is unchanged, as it was already correct)
    
    Args:
        layout_path: Path to the layout.png to lift.
        zmap_path: Path to the zmap.json created by create_zmap.
        bg_color: The RGB tuple of the background color to ignore.
        point_density: Fraction of pixels to sample (1.0 = all, 0.5 = 50%).
    
    Returns:
        A NumPy array of shape (N, 6) with columns [x, y, z, r, g, b].
    """
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

    # 4. Handle point density sampling
    if point_density < 1.0:
        num_samples = int(num_fg_pixels * point_density)
        sample_indices = np.random.choice(num_fg_pixels, num_samples, replace=False)
        fg_coords_y = fg_coords_y[sample_indices]
        fg_coords_x = fg_coords_x[sample_indices]

    lifted_points = []
    
    # 5. Iterate through pixels and "lift" them
    for y, x in zip(fg_coords_y, fg_coords_x):
        # Get pixel color
        r, g, b = layout_np[y, x]
        
        # Find Z statistics for this color
        color_key = f"rgb({r},{g},{b})"
        stats = zmap.get(color_key)
        
        if stats:
            # Sample Z value from the learned distribution
            z_mean = stats["mean"]
            z_std = stats["std"]
            # Set a minimum std deviation to avoid all points landing on one plane
            z_std = max(z_std, 0.01) 
            z_world = np.random.normal(z_mean, z_std)
            
            # We reverse the Y-axis flip from stage3
            x_world = x
            y_world = (H - 1) - y 
            
            lifted_points.append([x_world, y_world, z_world, r, g, b])

    return np.array(lifted_points, dtype=np.float64)


def save_point_cloud_as_ply(
    points: np.ndarray,
    output_ply_path: Union[str, Path]
):
    """
    Saves an (N, 6) [x,y,z,r,g,b] point cloud to a standard .ply file.
    (This function is unchanged)
    
    Args:
        points: The (N, 6) NumPy array from lift_layout.
        output_ply_path: Path to save the .ply file.
    """
    output_ply_path = Path(output_ply_path)
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
    parser_map.add_argument(
        "--hmin", type=float, default=DEFAULT_HEIGHT_RANGE[0], 
        help="Minimum Z height to analyze"
    )
    parser_map.add_argument(
        "--hmax", type=float, default=DEFAULT_HEIGHT_RANGE[1], 
        help="Maximum Z height to analyze"
    )

    # --- lift-layout command ---
    parser_lift = subparsers.add_parser(
        "lift-layout", 
        help="Lift a 2D layout.png to a 3D .ply point cloud."
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
            output_path=args.output,
            height_range=(args.hmin, args.hmax)
        )

    elif args.command == "lift-layout":
        print(f"Lifting {args.layout} using {args.zmap}...")
        lifted_points = lift_layout(
            layout_path=args.layout,
            zmap_path=args.zmap,
            point_density=args.density
        )
        
        if lifted_points.shape[0] > 0:
            save_point_cloud_as_ply(
                points=lifted_points,
                output_ply_path=args.output
            )
        else:
            print("No points were generated. Output .ply file not saved.")