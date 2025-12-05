#!/usr/bin/env python3
"""Compute A* paths from camera to object bounding boxes."""
import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.taxonomy import Taxonomy


def astar_path_to_bbox(grid, start, bbox, target_object_mask, taxonomy):
    """
    A* pathfinding from start to object bounding box.
    Walkable: floor, background, doors, windows, and the specific target object pixels.
    NOT walkable: walls and other objects.
    Goal: any point on the bounding box of the target object.
    
    Args:
        grid: 2D numpy array with class IDs
        start: (x, y) start position (camera)
        bbox: dict with min_x, min_y, max_x, max_y
        target_object_mask: 2D boolean array - True for pixels of the specific target object
        taxonomy: Taxonomy object to get category IDs
        other_objects_mask: 2D boolean array - True for pixels/bboxes of OTHER objects (to block)
    
    Returns:
        List of (x, y) positions forming the path, or None if no path found
    """
    # Get category IDs
    categories = taxonomy.data.get("categories", [])
    category2id = {cat: idx + 1 for idx, cat in enumerate(categories)}
    category2id["Unknown"] = 0
    
    floor_id = category2id.get("Floor", None)
    door_id = category2id.get("Door", None)
    window_id = category2id.get("Window", None)
    wall_id = category2id.get("Wall", None)
    
    h, w = grid.shape
    start_x, start_y = int(start[0]), int(start[1])
    
    # Check bounds
    if not (0 <= start_x < w and 0 <= start_y < h):
        return None
    
    # Create walkable mask
    walkable = np.zeros((h, w), dtype=bool)
    
    # Walkable: background, floor, doors, windows (not obstacles)
    walkable |= (grid == 0)  # Background
    if floor_id is not None:
        walkable |= (grid == floor_id)
    if door_id is not None:
        walkable |= (grid == door_id)
    if window_id is not None:
        walkable |= (grid == window_id)
    
    # NOT walkable: obstacles (marked as -1) and walls
    # ALL objects are already marked as -1 in the occupancy grid
    walkable &= (grid != -1)  # Obstacles - this blocks ALL object bounding boxes
    if wall_id is not None:
        walkable &= (grid != wall_id)
    
    if not walkable[start_y, start_x]:
        return None
    
    # Find goal: nearest walkable point OUTSIDE the bounding box (on perimeter)
    bbox_min_x, bbox_min_y = bbox['min_x'], bbox['min_y']
    bbox_max_x, bbox_max_y = bbox['max_x'], bbox['max_y']
    
    # Sample points just OUTSIDE bbox perimeter (adjacent to bbox, not inside)
    # We want points that are walkable and adjacent to the bbox
    bbox_points = []
    step = 1  # Check every point for better coverage
    
    # Points just above top edge
    for x in range(bbox_min_x, bbox_max_x + 1, step):
        y = bbox_min_y - 1
        if 0 <= y < h and 0 <= x < w:
            bbox_points.append((x, y))
    # Points just below bottom edge
    for x in range(bbox_min_x, bbox_max_x + 1, step):
        y = bbox_max_y + 1
        if 0 <= y < h and 0 <= x < w:
            bbox_points.append((x, y))
    # Points just left of left edge
    for y in range(bbox_min_y, bbox_max_y + 1, step):
        x = bbox_min_x - 1
        if 0 <= y < h and 0 <= x < w:
            bbox_points.append((x, y))
    # Points just right of right edge
    for y in range(bbox_min_y, bbox_max_y + 1, step):
        x = bbox_max_x + 1
        if 0 <= y < h and 0 <= x < w:
            bbox_points.append((x, y))
    
    # Also try corners (diagonal neighbors)
    corner_neighbors = [
        (bbox_min_x - 1, bbox_min_y - 1), (bbox_max_x + 1, bbox_min_y - 1),
        (bbox_min_x - 1, bbox_max_y + 1), (bbox_max_x + 1, bbox_max_y + 1)
    ]
    for cx, cy in corner_neighbors:
        if 0 <= cy < h and 0 <= cx < w:
            bbox_points.append((cx, cy))
    
    # Find closest walkable point OUTSIDE bbox to start
    goal_x, goal_y = None, None
    min_dist = float('inf')
    for bx, by in bbox_points:
        if 0 <= by < h and 0 <= bx < w:
            if walkable[by, bx]:  # Must be walkable (not inside any object)
                dist = abs(bx - start_x) + abs(by - start_y)
                if dist < min_dist:
                    min_dist = dist
                    goal_x, goal_y = bx, by
    
    if goal_x is None or goal_y is None:
        # No walkable point directly adjacent - try small expansion outward
        # Search in expanding rings around the bbox
        for radius in range(1, 10):  # Search up to 10 cells away
            for x in range(bbox_min_x - radius, bbox_max_x + radius + 1):
                for y in range(bbox_min_y - radius, bbox_max_y + radius + 1):
                    # Must be outside the bbox
                    if (x < bbox_min_x or x > bbox_max_x or 
                        y < bbox_min_y or y > bbox_max_y):
                        # Must be within radius distance from bbox
                        dist_to_bbox = min(
                            abs(x - bbox_min_x) if x < bbox_min_x else (x - bbox_max_x) if x > bbox_max_x else 0,
                            abs(y - bbox_min_y) if y < bbox_min_y else (y - bbox_max_y) if y > bbox_max_y else 0
                        )
                        if dist_to_bbox <= radius:
                            if 0 <= y < h and 0 <= x < w:
                                if walkable[y, x]:
                                    dist = abs(x - start_x) + abs(y - start_y)
                                    if dist < min_dist:
                                        min_dist = dist
                                        goal_x, goal_y = x, y
            if goal_x is not None:
                break
    
    if goal_x is None or goal_y is None:
        return None
    
    # Goal must be walkable (already checked above)
    if not walkable[goal_y, goal_x]:
        return None
    
    # A* algorithm - path to goal (optimized)
    from heapq import heappush, heappop
    
    def heuristic(x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)  # Manhattan distance
    
    # Limit search space to reasonable area around start and goal
    max_search_dist = max(abs(goal_x - start_x), abs(goal_y - start_y)) * 2 + 100
    
    open_set = [(0, start_x, start_y)]
    came_from = {}
    g_score = np.full((h, w), np.inf, dtype=np.float32)
    g_score[start_y, start_x] = 0
    f_score = np.full((h, w), np.inf, dtype=np.float32)
    f_score[start_y, start_x] = heuristic(start_x, start_y, goal_x, goal_y)
    visited = np.zeros((h, w), dtype=bool)
    
    # 4-directional movement (faster than 8-directional)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    iterations = 0
    max_iterations = h * w  # Safety limit
    
    while open_set and iterations < max_iterations:
        iterations += 1
        current_f, current_x, current_y = heappop(open_set)
        
        if visited[current_y, current_x]:
            continue
        visited[current_y, current_x] = True
        
        if current_x == goal_x and current_y == goal_y:
            # Reconstruct path
            path = []
            current = (current_x, current_y)
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(current)
            path.reverse()
            return path
        
        for dx, dy in neighbors:
            nx, ny = current_x + dx, current_y + dy
            
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if visited[ny, nx]:
                continue
            if not walkable[ny, nx]:
                continue
            
            # Check if within reasonable search distance
            dist_from_start = abs(nx - start_x) + abs(ny - start_y)
            if dist_from_start > max_search_dist:
                continue
            
            tentative_g = g_score[current_y, current_x] + 1
            
            if tentative_g < g_score[ny, nx]:
                came_from[(nx, ny)] = (current_x, current_y)
                g_score[ny, nx] = tentative_g
                f_score[ny, nx] = tentative_g + heuristic(nx, ny, goal_x, goal_y)
                heappush(open_set, (f_score[ny, nx], nx, ny))
    
    return None  # No path found


def main():
    parser = argparse.ArgumentParser(
        description="Compute A* paths from camera to object bounding boxes"
    )
    parser.add_argument(
        "occupancy_grid_path",
        type=Path,
        help="Path to occupancy grid .npy file"
    )
    parser.add_argument(
        "objects_path",
        type=Path,
        help="Path to objects JSON file"
    )
    parser.add_argument(
        "cleaned_layout_path",
        type=Path,
        help="Path to cleaned layout image"
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("../data_preparation_v2/taxonomy.json"),
        help="Path to taxonomy JSON file (default: ../data_preparation_v2/taxonomy.json)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as occupancy grid file)"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Base name for output files (default: occupancy grid filename without extension)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.occupancy_grid_path.exists():
        print(f"Error: Occupancy grid file not found: {args.occupancy_grid_path}")
        sys.exit(1)
    
    if not args.objects_path.exists():
        print(f"Error: Objects file not found: {args.objects_path}")
        sys.exit(1)
    
    if not args.cleaned_layout_path.exists():
        print(f"Error: Cleaned layout file not found: {args.cleaned_layout_path}")
        sys.exit(1)
    
    if not args.taxonomy.exists():
        print(f"Error: Taxonomy file not found: {args.taxonomy}")
        sys.exit(1)
    
    # Load data
    print(f"Loading occupancy grid from {args.occupancy_grid_path}...")
    occupancy_grid = np.load(args.occupancy_grid_path)
    
    print(f"Loading objects from {args.objects_path}...")
    with open(args.objects_path, 'r') as f:
        objects_data = json.load(f)
    
    camera_pos = tuple(objects_data['camera_pos'])
    objects = objects_data['objects']
    grid_resolution = objects_data.get('grid_resolution', occupancy_grid.shape[0])
    scale = objects_data.get('scale', (1.0, 1.0))
    
    print(f"Loading taxonomy from {args.taxonomy}...")
    taxonomy = Taxonomy(args.taxonomy)
    
    print(f"Loading cleaned layout from {args.cleaned_layout_path}...")
    cleaned_layout = Image.open(args.cleaned_layout_path).convert("RGB")
    
    # Reconstruct object masks from stored coordinates (already in grid space)
    print("Reconstructing object masks...")
    h, w = occupancy_grid.shape
    objects_with_masks = []
    for obj in objects:
        if 'mask_coords' not in obj:
            continue
        
        # Reconstruct mask from coordinates (in grid coordinates)
        obj_mask = np.zeros((h, w), dtype=bool)
        for gy, gx in obj['mask_coords']:
            if 0 <= gy < h and 0 <= gx < w:
                obj_mask[gy, gx] = True
        
        # Use grid coordinates for pathfinding
        bbox = obj.get('bbox_grid', obj['bbox'])
        centroid = obj.get('centroid_grid', obj['centroid'])
        
        objects_with_masks.append({
            'class_id': obj['class_id'],
            'centroid': tuple(centroid),  # Grid coordinates
            'bbox': bbox,  # Grid coordinates
            'mask': obj_mask
        })
    
    print(f"  Found {len(objects_with_masks)} objects with valid masks")
    
    # Compute A* paths
    print("Computing A* paths from camera to object bounding boxes...")
    paths = []
    for obj in objects_with_masks:
        # All objects are already marked as -1 in occupancy grid, no need for extra masks
        path = astar_path_to_bbox(
            occupancy_grid, 
            camera_pos, 
            obj['bbox'], 
            obj['mask'], 
            taxonomy
        )
        if path:
            paths.append({
                'class_id': obj['class_id'],
                'centroid': obj['centroid'],
                'bbox': obj['bbox'],
                'path': path,
                'path_length': len(path)
            })
        else:
            print(f"  Warning: No path found to object {obj['class_id']} at {obj['centroid']}")
    
    print(f"  Found paths to {len(paths)}/{len(objects_with_masks)} objects")
    
    # Draw paths on cleaned layout
    print("Drawing paths on layout...")
    draw = ImageDraw.Draw(cleaned_layout)
    
    # Get category names and colors
    categories = taxonomy.data.get("categories", [])
    category2id = {cat: idx + 1 for idx, cat in enumerate(categories)}
    category2id["Unknown"] = 0
    id2category = {v: k for k, v in category2id.items()}
    category_to_color = taxonomy.data.get("category_to_color", {})
    
    # Convert camera position from grid to image coordinates
    scale_x, scale_y = scale
    img_cam_x = int(camera_pos[0] * scale_x)
    img_cam_y = int(camera_pos[1] * scale_y)
    
    # Draw camera marker (red circle)
    radius = 5
    draw.ellipse([img_cam_x - radius, img_cam_y - radius, 
                  img_cam_x + radius, img_cam_y + radius], 
                 fill=(255, 0, 0), outline=(255, 0, 0))
    
    # Draw paths as dashed lines
    for path_info in paths:
        class_id = path_info['class_id']
        path = path_info['path']  # Path is in grid coordinates
        
        # Get category name and color
        category_name = id2category.get(class_id, "Unknown")
        if category_name in category_to_color:
            path_color = tuple(category_to_color[category_name])
        else:
            path_color = (127, 127, 127)  # Default gray
        
        # Convert grid coordinates to image coordinates for drawing
        scale_x, scale_y = scale
        
        # Draw object centroid marker (convert from grid to image coords)
        grid_centroid_x, grid_centroid_y = path_info['centroid']
        centroid_x = int(grid_centroid_x * scale_x)
        centroid_y = int(grid_centroid_y * scale_y)
        draw.ellipse([centroid_x - 3, centroid_y - 3, 
                      centroid_x + 3, centroid_y + 3], 
                     fill=path_color, outline=(0, 0, 0))
        
        # Draw bounding box (convert from grid to image coords)
        bbox = path_info['bbox']
        img_bbox = {
            'min_x': int(bbox['min_x'] * scale_x),
            'min_y': int(bbox['min_y'] * scale_y),
            'max_x': int(bbox['max_x'] * scale_x),
            'max_y': int(bbox['max_y'] * scale_y)
        }
        draw.rectangle([img_bbox['min_x'], img_bbox['min_y'], 
                       img_bbox['max_x'], img_bbox['max_y']], 
                       outline=path_color, width=1)
        
        # Draw path (convert from grid to image coords)
        for i in range(len(path) - 1):
            gx1, gy1 = path[i]
            gx2, gy2 = path[i + 1]
            x1 = int(gx1 * scale_x)
            y1 = int(gy1 * scale_y)
            x2 = int(gx2 * scale_x)
            y2 = int(gy2 * scale_y)
            
            # Calculate distance and direction
            dx = x2 - x1
            dy = y2 - y1
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist > 0:
                # Normalize direction
                dx /= dist
                dy /= dist
                
                # Draw dashed segments
                dash_length = 8
                gap_length = 4
                current_dist = 0
                draw_dash = True
                while current_dist < dist:
                    if draw_dash:
                        # Draw dash
                        dash_end = min(current_dist + dash_length, dist)
                        start_x = int(x1 + dx * current_dist)
                        start_y = int(y1 + dy * current_dist)
                        end_x = int(x1 + dx * dash_end)
                        end_y = int(y1 + dy * dash_end)
                        draw.line([(start_x, start_y), (end_x, end_y)], fill=path_color, width=2)
                        current_dist += dash_length
                    else:
                        # Skip gap
                        current_dist += gap_length
                    draw_dash = not draw_dash
        
        # Draw marker at end of path (where it touches bbox)
        if len(path_info['path']) > 0:
            end_gx, end_gy = path_info['path'][-1]
            end_x = int(end_gx * scale_x)
            end_y = int(end_gy * scale_y)
            draw.ellipse([end_x - 2, end_y - 2, 
                          end_x + 2, end_y + 2], 
                         fill=(255, 255, 0), outline=(0, 0, 0))  # Yellow marker at path end
    
    # Determine output paths
    if args.output_dir is None:
        output_dir = args.occupancy_grid_path.parent
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.output_name is None:
        base_name = args.occupancy_grid_path.stem.replace("_occupancy_grid", "")
    else:
        base_name = args.output_name
    
    paths_path = output_dir / f"{base_name}_paths.json"
    layout_with_paths_path = output_dir / f"{base_name}_layout_with_paths.png"
    
    # Save paths
    paths_data = {
        'camera_pos': camera_pos,
        'objects': paths
    }
    print(f"Saving paths to {paths_path}...")
    with open(paths_path, 'w') as f:
        json.dump(paths_data, f, indent=2)
    
    # Save layout with paths
    print(f"Saving layout with paths to {layout_with_paths_path}...")
    cleaned_layout.save(layout_with_paths_path)
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Objects found: {len(objects_with_masks)}")
    print(f"  Paths computed: {len(paths)}")
    print(f"  Camera position: (x={camera_pos[0]}, y={camera_pos[1]})")
    print(f"\nOutput files:")
    print(f"  Paths: {paths_path}")
    print(f"  Layout with paths: {layout_with_paths_path}")


if __name__ == "__main__":
    main()

