#!/usr/bin/env python3
"""
Stage 0: Scene Validation

Validates 3D-FRONT scenes before processing and outputs a list of valid scene IDs.

Checks:
1. Scene JSON exists and is valid
2. Has rooms with valid structure
3. Has furniture with valid JIDs
4. Referenced models exist in 3D-FUTURE
5. Has doors and/or windows
6. Rooms have floor geometry (for bbox computation)
7. No outdoor/garden rooms (optional)
8. Scene is spatially connected (no disjoint parts)
9. Rooms are not split into multiple parts

Output: valid_scenes.txt - one scene ID per line
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Room types that indicate outdoor/garden areas
OUTDOOR_ROOM_TYPES = {
    "balcony", "garden", "terrace", "outdoor", "patio", 
    "yard", "deck", "porch", "veranda", "loggia"
}


class ValidationResult:
    def __init__(self, scene_id: str):
        self.scene_id = scene_id
        self.is_valid = True
        self.errors = []
        self.warnings = []
        self.stats = {}
    
    def add_error(self, msg: str):
        self.is_valid = False
        self.errors.append(msg)
    
    def add_warning(self, msg: str):
        self.warnings.append(msg)
    
    def __str__(self):
        status = "✓ VALID" if self.is_valid else "✗ INVALID"
        lines = [f"{self.scene_id}: {status}"]
        for err in self.errors:
            lines.append(f"  ERROR: {err}")
        for warn in self.warnings:
            lines.append(f"  WARNING: {warn}")
        return "\n".join(lines)


def get_available_models(model_dir: Path) -> Set[str]:
    """Get set of available model JIDs."""
    models = set()
    if model_dir.exists():
        for subdir in model_dir.iterdir():
            if subdir.is_dir():
                if (subdir / "raw_model.obj").exists() or (subdir / "raw_model.glb").exists():
                    models.add(subdir.name)
    return models


def get_floor_vertices(room: Dict, mesh_map: Dict) -> Optional[np.ndarray]:
    """Extract floor vertices for a room."""
    all_vertices = []
    
    for child in room.get("children", []):
        ref_id = child.get("ref")
        if ref_id and ref_id in mesh_map:
            mesh = mesh_map[ref_id]
            if "Floor" in mesh.get("type", ""):
                xyz = mesh.get("xyz", [])
                if xyz:
                    vertices = np.array(xyz, dtype=np.float64).reshape(-1, 3)
                    all_vertices.append(vertices)
    
    if all_vertices:
        return np.vstack(all_vertices)
    return None


def compute_room_bbox(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 2D bounding box (X-Z plane) from vertices."""
    if len(vertices) == 0:
        return np.array([0, 0]), np.array([1, 1])
    
    min_xz = np.array([vertices[:, 0].min(), vertices[:, 2].min()])
    max_xz = np.array([vertices[:, 0].max(), vertices[:, 2].max()])
    return min_xz, max_xz


def boxes_overlap(box1_min: np.ndarray, box1_max: np.ndarray, 
                  box2_min: np.ndarray, box2_max: np.ndarray,
                  margin: float = 0.5) -> bool:
    """Check if two 2D boxes overlap or are adjacent (within margin)."""
    # Expand boxes by margin
    b1_min = box1_min - margin
    b1_max = box1_max + margin
    b2_min = box2_min - margin
    b2_max = box2_max + margin
    
    # Check overlap
    return not (b1_max[0] < b2_min[0] or b2_max[0] < b1_min[0] or
                b1_max[1] < b2_min[1] or b2_max[1] < b1_min[1])


def find_connected_components(adjacency: Dict[int, Set[int]], n_nodes: int) -> List[Set[int]]:
    """Find connected components using BFS."""
    visited = set()
    components = []
    
    for start in range(n_nodes):
        if start in visited:
            continue
        
        # BFS from start
        component = set()
        queue = [start]
        
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            
            visited.add(node)
            component.add(node)
            
            for neighbor in adjacency.get(node, set()):
                if neighbor not in visited:
                    queue.append(neighbor)
        
        if component:
            components.append(component)
    
    return components


def check_scene_connectivity(rooms: List[Dict], mesh_map: Dict) -> Tuple[bool, int, List[str]]:
    """
    Check if all rooms in a scene are spatially connected.
    
    Returns:
        Tuple of (is_connected, num_components, list of disjoint room groups)
    """
    if len(rooms) <= 1:
        return True, 1, []
    
    # Get bounding boxes for all rooms
    room_boxes = []
    room_names = []
    
    for room in rooms:
        room_type = room.get("type", "Unknown")
        room_names.append(room_type)
        
        vertices = get_floor_vertices(room, mesh_map)
        if vertices is not None and len(vertices) > 0:
            bbox_min, bbox_max = compute_room_bbox(vertices)
            room_boxes.append((bbox_min, bbox_max))
        else:
            # No floor geometry - use a dummy box
            room_boxes.append((np.array([0, 0]), np.array([0, 0])))
    
    # Build adjacency graph
    n_rooms = len(rooms)
    adjacency = defaultdict(set)
    
    for i in range(n_rooms):
        for j in range(i + 1, n_rooms):
            if boxes_overlap(room_boxes[i][0], room_boxes[i][1],
                           room_boxes[j][0], room_boxes[j][1], margin=0.5):
                adjacency[i].add(j)
                adjacency[j].add(i)
    
    # Find connected components
    components = find_connected_components(adjacency, n_rooms)
    
    is_connected = len(components) == 1
    
    # Build list of room groups for reporting
    disjoint_groups = []
    if len(components) > 1:
        for comp in components:
            group_names = [room_names[i] for i in comp]
            disjoint_groups.append(", ".join(group_names))
    
    return is_connected, len(components), disjoint_groups


def check_room_connectivity(room: Dict, mesh_map: Dict) -> Tuple[bool, int]:
    """
    Check if a room's floor geometry is connected (not split into parts).
    
    Uses simple grid-based connectivity analysis.
    
    Returns:
        Tuple of (is_connected, num_components)
    """
    vertices = get_floor_vertices(room, mesh_map)
    if vertices is None or len(vertices) == 0:
        return True, 1  # No floor = assume connected
    
    # Project to 2D (X-Z plane)
    points_2d = vertices[:, [0, 2]]
    
    if len(points_2d) < 3:
        return True, 1
    
    # Create a simple grid and mark cells with floor vertices
    grid_resolution = 0.5  # 50cm grid
    
    min_pt = points_2d.min(axis=0)
    max_pt = points_2d.max(axis=0)
    
    # Grid dimensions
    grid_size = ((max_pt - min_pt) / grid_resolution + 1).astype(int)
    grid_size = np.clip(grid_size, 1, 200)  # Limit grid size
    
    if grid_size[0] * grid_size[1] > 10000:
        # Too large, skip detailed check
        return True, 1
    
    # Mark cells with floor vertices
    grid = np.zeros(grid_size, dtype=bool)
    
    for pt in points_2d:
        cell = ((pt - min_pt) / grid_resolution).astype(int)
        cell = np.clip(cell, 0, grid_size - 1)
        grid[cell[0], cell[1]] = True
    
    # Find connected components in grid using flood fill
    visited = np.zeros_like(grid, dtype=bool)
    components = 0
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if grid[i, j] and not visited[i, j]:
                # Flood fill from this cell
                components += 1
                stack = [(i, j)]
                
                while stack:
                    ci, cj = stack.pop()
                    if ci < 0 or ci >= grid_size[0] or cj < 0 or cj >= grid_size[1]:
                        continue
                    if visited[ci, cj] or not grid[ci, cj]:
                        continue
                    
                    visited[ci, cj] = True
                    
                    # Add 4-connected neighbors
                    stack.extend([(ci+1, cj), (ci-1, cj), (ci, cj+1), (ci, cj-1)])
    
    return components <= 1, components


def is_outdoor_room(room_type: str) -> bool:
    """Check if room type indicates an outdoor/garden area."""
    room_lower = room_type.lower()
    return any(outdoor in room_lower for outdoor in OUTDOOR_ROOM_TYPES)


def validate_scene(
    scene_path: Path,
    available_models: Set[str],
    min_rooms: int = 1,
    min_furniture: int = 0,
    require_doors: bool = False,
    require_windows: bool = False,
    exclude_outdoor: bool = False,
    require_connected: bool = False,
    check_room_splits: bool = False,
) -> ValidationResult:
    """
    Validate a single scene.
    """
    scene_id = scene_path.stem
    result = ValidationResult(scene_id)
    
    # Check 1: JSON is valid
    try:
        with open(scene_path, "r", encoding="utf-8") as f:
            scene_data = json.load(f)
    except json.JSONDecodeError as e:
        result.add_error(f"Invalid JSON: {e}")
        return result
    except Exception as e:
        result.add_error(f"Cannot read file: {e}")
        return result
    
    # Check 2: Has scene structure
    scene_struct = scene_data.get("scene", {})
    if not scene_struct:
        result.add_error("Missing 'scene' structure")
        return result
    
    # Check 3: Has rooms
    rooms = scene_struct.get("room", [])
    if len(rooms) < min_rooms:
        result.add_error(f"Not enough rooms: {len(rooms)} < {min_rooms}")
        return result
    
    result.stats["num_rooms"] = len(rooms)
    
    # Build mesh map for geometry checks
    mesh_map = {m.get("uid"): m for m in scene_data.get("mesh", [])}
    
    # Check 4: Outdoor rooms
    outdoor_rooms = []
    indoor_rooms = []
    
    for room in rooms:
        room_type = room.get("type", "Unknown")
        if is_outdoor_room(room_type):
            outdoor_rooms.append(room_type)
        else:
            indoor_rooms.append(room)
    
    result.stats["outdoor_rooms"] = len(outdoor_rooms)
    result.stats["indoor_rooms"] = len(indoor_rooms)
    
    if outdoor_rooms:
        if exclude_outdoor:
            result.add_warning(f"Has outdoor rooms: {', '.join(outdoor_rooms)}")
        else:
            result.add_warning(f"Has outdoor rooms (not excluded): {', '.join(outdoor_rooms)}")
    
    # Check 5: Scene connectivity (using indoor rooms only)
    if require_connected and len(indoor_rooms) > 1:
        is_connected, num_components, disjoint_groups = check_scene_connectivity(
            indoor_rooms, mesh_map
        )
        result.stats["num_components"] = num_components
        
        if not is_connected:
            result.add_error(f"Scene has {num_components} disjoint parts: {disjoint_groups}")
            return result
    
    # Check 6: Room splits (each room should be one connected piece)
    if check_room_splits:
        split_rooms = []
        for room in indoor_rooms:
            room_type = room.get("type", "Unknown")
            is_connected, num_parts = check_room_connectivity(room, mesh_map)
            if not is_connected:
                split_rooms.append(f"{room_type} ({num_parts} parts)")
        
        result.stats["split_rooms"] = len(split_rooms)
        
        if split_rooms:
            result.add_error(f"Rooms with disjoint floor geometry: {', '.join(split_rooms)}")
            return result
    
    # Check 7: Rooms have valid structure
    valid_rooms = 0
    rooms_with_floor = 0
    
    for room in rooms:
        room_type = room.get("type", "Unknown")
        children = room.get("children", [])
        
        if not children:
            result.add_warning(f"Room '{room_type}' has no children")
            continue
        
        # Check if room has floor geometry
        has_floor = False
        for child in children:
            ref_id = child.get("ref")
            if ref_id and ref_id in mesh_map:
                mesh = mesh_map[ref_id]
                if "Floor" in mesh.get("type", ""):
                    has_floor = True
                    break
        
        if has_floor:
            rooms_with_floor += 1
        
        valid_rooms += 1
    
    if valid_rooms == 0:
        result.add_error("No valid rooms found")
        return result
    
    if rooms_with_floor == 0:
        result.add_warning("No rooms have floor geometry (bbox computation may fail)")
    
    result.stats["valid_rooms"] = valid_rooms
    result.stats["rooms_with_floor"] = rooms_with_floor
    
    # Check 8: Has furniture
    furniture_list = scene_data.get("furniture", [])
    result.stats["num_furniture"] = len(furniture_list)
    
    if len(furniture_list) < min_furniture:
        result.add_error(f"Not enough furniture: {len(furniture_list)} < {min_furniture}")
        return result
    
    # Check 9: Furniture models exist
    missing_models = []
    valid_furniture = 0
    
    for item in furniture_list:
        jid = item.get("jid")
        if not jid:
            continue
        
        if jid in available_models:
            valid_furniture += 1
        else:
            missing_models.append(jid)
    
    result.stats["valid_furniture"] = valid_furniture
    result.stats["missing_models"] = len(missing_models)
    
    if missing_models:
        if len(missing_models) > 5:
            result.add_warning(f"{len(missing_models)} furniture models not found in 3D-FUTURE")
        else:
            result.add_warning(f"Missing models: {missing_models[:5]}")
    
    if valid_furniture == 0 and len(furniture_list) > 0:
        result.add_error("No furniture models found in 3D-FUTURE")
        return result
    
    # Check 10: Has doors/windows in mesh array
    mesh_list = scene_data.get("mesh", [])
    mesh_doors = [m for m in mesh_list if "door" in m.get("type", "").lower()]
    mesh_windows = [m for m in mesh_list if "window" in m.get("type", "").lower()]
    
    # Also count furniture-based doors/windows
    furniture_doors = 0
    furniture_windows = 0
    for furn in furniture_list:
        title = furn.get("title", "").lower()
        if title.startswith("door/"):
            furniture_doors += 1
        elif title.startswith("window/"):
            furniture_windows += 1
    
    result.stats["mesh_doors"] = len(mesh_doors)
    result.stats["mesh_windows"] = len(mesh_windows)
    result.stats["furniture_doors"] = furniture_doors
    result.stats["furniture_windows"] = furniture_windows
    result.stats["num_doors"] = len(mesh_doors) + furniture_doors
    result.stats["num_windows"] = len(mesh_windows) + furniture_windows
    
    # Check if doors/windows are only in furniture (problematic for orientation)
    if len(mesh_doors) == 0 and furniture_doors > 0:
        result.add_warning(f"Doors only in furniture ({furniture_doors}), may have orientation issues")
    
    if len(mesh_windows) == 0 and furniture_windows > 0:
        result.add_warning(f"Windows only in furniture ({furniture_windows}), may have orientation issues")
    
    if require_doors and len(mesh_doors) == 0 and furniture_doors == 0:
        result.add_error("No doors found")
        return result
    
    if require_windows and len(mesh_windows) == 0 and furniture_windows == 0:
        result.add_error("No windows found")
        return result
    
    if len(mesh_doors) == 0 and len(mesh_windows) == 0 and furniture_doors == 0 and furniture_windows == 0:
        result.add_warning("No doors or windows found in scene")
    
    # Check 11: Doors/windows have geometry (only relevant for mesh-based ones)
    doors_with_geometry = sum(1 for d in mesh_doors if d.get("xyz") and d.get("faces"))
    windows_with_geometry = sum(1 for w in mesh_windows if w.get("xyz") and w.get("faces"))
    
    result.stats["doors_with_geometry"] = doors_with_geometry
    result.stats["windows_with_geometry"] = windows_with_geometry
    
    if len(mesh_doors) > 0 and doors_with_geometry == 0:
        result.add_warning("Mesh doors have no geometry data")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate 3D-FRONT scenes")
    parser.add_argument("--scenes-dir", required=True, help="3D-FRONT scenes directory")
    parser.add_argument("--model-dir", required=True, help="3D-FUTURE models directory")
    parser.add_argument("--output", default="valid_scenes.txt", help="Output file for valid scene IDs")
    parser.add_argument("--min-rooms", type=int, default=1, help="Minimum number of rooms")
    parser.add_argument("--min-furniture", type=int, default=1, help="Minimum furniture items")
    parser.add_argument("--require-doors", action="store_true", help="Require at least one door")
    parser.add_argument("--require-windows", action="store_true", help="Require at least one window")
    parser.add_argument("--exclude-outdoor", action="store_true", help="Warn about outdoor/garden rooms")
    parser.add_argument("--require-connected", action="store_true", help="Require all rooms to be spatially connected")
    parser.add_argument("--check-room-splits", action="store_true", help="Check for rooms with disjoint floor geometry")
    parser.add_argument("--verbose", action="store_true", help="Show details for each scene")
    parser.add_argument("--stats", action="store_true", help="Show statistics summary")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of scenes to check")
    args = parser.parse_args()
    
    scenes_dir = Path(args.scenes_dir)
    model_dir = Path(args.model_dir)
    output_path = Path(args.output)
    
    # Find scene files
    scene_files = list(scenes_dir.glob("*.json"))
    if not scene_files:
        logger.error(f"No scene files found in {scenes_dir}")
        return
    
    if args.limit:
        scene_files = scene_files[:args.limit]
    
    logger.info(f"Found {len(scene_files)} scene files")
    
    # Get available models
    logger.info(f"Scanning 3D-FUTURE models in {model_dir}...")
    available_models = get_available_models(model_dir)
    logger.info(f"Found {len(available_models)} available models")
    
    # Validate scenes
    logger.info(f"Validating scenes...")
    
    results = []
    for i, scene_path in enumerate(scene_files):
        result = validate_scene(
            scene_path,
            available_models,
            min_rooms=args.min_rooms,
            min_furniture=args.min_furniture,
            require_doors=args.require_doors,
            require_windows=args.require_windows,
            exclude_outdoor=args.exclude_outdoor,
            require_connected=args.require_connected,
            check_room_splits=args.check_room_splits,
        )
        results.append(result)
        
        if args.verbose:
            print(result)
        
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(scene_files)} scenes...")
    
    # Separate valid and invalid
    valid_results = [r for r in results if r.is_valid]
    invalid_results = [r for r in results if not r.is_valid]
    
    # Write valid scene IDs
    with open(output_path, "w") as f:
        for result in valid_results:
            f.write(f"{result.scene_id}\n")
    
    logger.info(f"\nValidation complete:")
    logger.info(f"  Valid scenes: {len(valid_results)}")
    logger.info(f"  Invalid scenes: {len(invalid_results)}")
    logger.info(f"  Output written to: {output_path}")
    
    # Show statistics
    if args.stats and valid_results:
        total_rooms = sum(r.stats.get("num_rooms", 0) for r in valid_results)
        total_furniture = sum(r.stats.get("valid_furniture", 0) for r in valid_results)
        total_doors = sum(r.stats.get("num_doors", 0) for r in valid_results)
        total_windows = sum(r.stats.get("num_windows", 0) for r in valid_results)
        total_outdoor = sum(r.stats.get("outdoor_rooms", 0) for r in valid_results)
        
        # Mesh vs furniture breakdown
        mesh_doors = sum(r.stats.get("mesh_doors", 0) for r in valid_results)
        mesh_windows = sum(r.stats.get("mesh_windows", 0) for r in valid_results)
        furn_doors = sum(r.stats.get("furniture_doors", 0) for r in valid_results)
        furn_windows = sum(r.stats.get("furniture_windows", 0) for r in valid_results)
        
        logger.info(f"\nStatistics for valid scenes:")
        logger.info(f"  Total rooms: {total_rooms}")
        logger.info(f"  Total outdoor rooms: {total_outdoor}")
        logger.info(f"  Total furniture: {total_furniture}")
        logger.info(f"  Total doors: {total_doors} (mesh: {mesh_doors}, furniture: {furn_doors})")
        logger.info(f"  Total windows: {total_windows} (mesh: {mesh_windows}, furniture: {furn_windows})")
        logger.info(f"  Avg rooms/scene: {total_rooms / len(valid_results):.1f}")
        logger.info(f"  Avg furniture/scene: {total_furniture / len(valid_results):.1f}")
        
        # Count scenes with furniture-only doors/windows
        scenes_furniture_only_doors = sum(1 for r in valid_results 
            if r.stats.get("mesh_doors", 0) == 0 and r.stats.get("furniture_doors", 0) > 0)
        scenes_furniture_only_windows = sum(1 for r in valid_results 
            if r.stats.get("mesh_windows", 0) == 0 and r.stats.get("furniture_windows", 0) > 0)
        
        if scenes_furniture_only_doors > 0:
            logger.info(f"  ⚠ Scenes with furniture-only doors: {scenes_furniture_only_doors}")
        if scenes_furniture_only_windows > 0:
            logger.info(f"  ⚠ Scenes with furniture-only windows: {scenes_furniture_only_windows}")
    
    # Show common errors
    if invalid_results:
        error_counts = {}
        for result in invalid_results:
            for err in result.errors:
                key = err.split(":")[0] if ":" in err else err[:50]
                error_counts[key] = error_counts.get(key, 0) + 1
        
        logger.info(f"\nCommon errors ({len(invalid_results)} invalid scenes):")
        for err, count in sorted(error_counts.items(), key=lambda x: -x[1])[:10]:
            logger.info(f"  {count}x {err}")


if __name__ == "__main__":
    main()