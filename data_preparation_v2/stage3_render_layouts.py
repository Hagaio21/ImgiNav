#!/usr/bin/env python3
"""
Stage 3: Layout Rendering

Features:
- Scene list support (--scene-list with scene IDs)
- Room filtering properly clips to room bbox
- Floor and wall have different colors in textured version
- Room layouts show only the room content, not entire scene
- Uses taxonomy colors for doors/windows
- HPC mode with Xvfb for headless rendering (only when --hpc flag is set)
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
from PIL import Image, ImageDraw

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global Xvfb display reference
_xvfb_display = None


def setup_hpc_rendering() -> bool:
    """
    Set up Xvfb virtual display for HPC headless rendering.
    
    Returns True if successful, False otherwise.
    """
    global _xvfb_display
    
    try:
        from xvfbwrapper import Xvfb
        _xvfb_display = Xvfb(width=1280, height=720)
        _xvfb_display.start()
        logger.info(f"Xvfb started on display :{_xvfb_display.new_display}")
        return True
    except ImportError:
        logger.error("xvfbwrapper not installed. Install with: pip install xvfbwrapper")
        return False
    except Exception as e:
        logger.error(f"Failed to start Xvfb: {e}")
        return False


def cleanup_hpc_rendering():
    """Clean up Xvfb display."""
    global _xvfb_display
    if _xvfb_display is not None:
        try:
            _xvfb_display.stop()
            logger.info("Xvfb stopped")
        except Exception:
            pass
        _xvfb_display = None


def safe_mkdir(path: Path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_scene_list(scene_list_path: Path) -> List[str]:
    """Load scene IDs from a text file (one per line)."""
    scenes = []
    with open(scene_list_path, "r", encoding="utf-8") as f:
        for line in f:
            scene_id = line.strip()
            if scene_id and not scene_id.startswith("#"):
                scenes.append(scene_id)
    return scenes


def load_glb_with_transforms(glb_path: Path) -> List[trimesh.Trimesh]:
    """
    Load a GLB file and return list of meshes with transforms applied.
    """
    scene = trimesh.load(str(glb_path), process=False, force="scene")
    
    if not isinstance(scene, trimesh.Scene):
        return [scene] if isinstance(scene, trimesh.Trimesh) else []
    
    try:
        meshes = scene.dump(concatenate=False)
        if meshes is None:
            meshes = []
        meshes = [m for m in meshes if isinstance(m, trimesh.Trimesh) and len(m.vertices) > 0]
        return meshes
    except Exception as e:
        logger.warning(f"scene.dump() failed: {e}")
        return []


def get_mesh_color(mesh: trimesh.Trimesh) -> np.ndarray:
    """Extract vertex colors from a mesh."""
    n_vertices = len(mesh.vertices)
    default_color = np.full((n_vertices, 3), 180, dtype=np.uint8)
    
    if not hasattr(mesh, 'visual'):
        return default_color
    
    visual = mesh.visual
    
    # Try vertex colors
    if hasattr(visual, 'vertex_colors') and visual.vertex_colors is not None:
        colors = np.array(visual.vertex_colors)
        if len(colors) == n_vertices and colors.shape[-1] >= 3:
            return colors[:, :3].astype(np.uint8)
    
    # Try to_color conversion
    try:
        if hasattr(visual, 'to_color'):
            color_visual = visual.to_color()
            if hasattr(color_visual, 'vertex_colors') and color_visual.vertex_colors is not None:
                colors = np.array(color_visual.vertex_colors)
                if len(colors) == n_vertices and colors.shape[-1] >= 3:
                    return colors[:, :3].astype(np.uint8)
    except Exception:
        pass
    
    # Try material
    if hasattr(visual, 'material') and visual.material is not None:
        mat = visual.material
        for attr in ['diffuse', 'baseColorFactor', 'main_color']:
            if hasattr(mat, attr):
                color = getattr(mat, attr)
                if color is not None:
                    color = np.array(color).flatten()
                    if len(color) >= 3:
                        if color.max() <= 1.0:
                            color = (color * 255).astype(np.uint8)
                        return np.tile(color[:3].astype(np.uint8), (n_vertices, 1))
    
    return default_color


def get_meshes_bounds(meshes: List[trimesh.Trimesh]) -> Tuple[np.ndarray, np.ndarray]:
    """Get combined bounds from list of meshes."""
    all_vertices = []
    for mesh in meshes:
        if len(mesh.vertices) > 0:
            all_vertices.append(mesh.vertices)
    
    if not all_vertices:
        return np.array([0, 0, 0]), np.array([1, 1, 1])
    
    all_vertices = np.vstack(all_vertices)
    return all_vertices.min(axis=0), all_vertices.max(axis=0)


def clip_mesh_to_bbox(
    mesh: trimesh.Trimesh,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray
) -> Optional[trimesh.Trimesh]:
    """
    Clip a mesh to only include faces within a bounding box.
    Returns None if no faces remain.
    
    Only checks X and Z coordinates (floor plane), not Y (height).
    This prevents furniture from being clipped vertically.
    """
    if len(mesh.vertices) == 0:
        return None
    
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Check which vertices are inside bbox on X-Z plane only (ignore Y/height)
    margin = 0.1
    inside_x = (vertices[:, 0] >= bbox_min[0] - margin) & (vertices[:, 0] <= bbox_max[0] + margin)
    inside_z = (vertices[:, 2] >= bbox_min[2] - margin) & (vertices[:, 2] <= bbox_max[2] + margin)
    inside = inside_x & inside_z  # Only check X and Z, not Y
    
    # Keep faces where at least one vertex is inside
    face_mask = inside[faces].any(axis=1)
    
    if not face_mask.any():
        return None
    
    # Get the faces that pass the filter
    kept_faces = faces[face_mask]
    
    # Get unique vertices used by kept faces
    unique_verts = np.unique(kept_faces.flatten())
    
    # Create mapping from old to new vertex indices
    vert_map = np.zeros(len(vertices), dtype=np.int64)
    vert_map[unique_verts] = np.arange(len(unique_verts))
    
    # Create new mesh
    new_vertices = vertices[unique_verts]
    new_faces = vert_map[kept_faces]
    
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    
    # Copy colors
    if hasattr(mesh, 'visual'):
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            old_colors = np.array(mesh.visual.vertex_colors)
            if len(old_colors) == len(vertices):
                new_mesh.visual.vertex_colors = old_colors[unique_verts]
        elif hasattr(mesh.visual, 'material'):
            new_mesh.visual.material = mesh.visual.material
    
    return new_mesh


def filter_meshes_to_bbox(
    meshes: List[trimesh.Trimesh],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray
) -> List[trimesh.Trimesh]:
    """Filter and clip meshes to a bounding box."""
    filtered = []
    
    for mesh in meshes:
        clipped = clip_mesh_to_bbox(mesh, bbox_min, bbox_max)
        if clipped is not None and len(clipped.vertices) > 0:
            filtered.append(clipped)
    
    return filtered


def render_topdown(
    meshes: List[trimesh.Trimesh],
    resolution: int = 512,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    bounds_override: Optional[Tuple[float, float, float, float]] = None
) -> Tuple[Image.Image, Tuple[float, float, float, float]]:
    """
    Render a top-down view.
    
    Args:
        meshes: List of meshes to render
        resolution: Output image resolution
        bg_color: Background color
        bounds_override: Optional (x_min, x_max, z_min, z_max) to use instead of auto-computed bounds
    """
    if not meshes:
        if bounds_override:
            return Image.new("RGB", (resolution, resolution), bg_color), bounds_override
        return Image.new("RGB", (resolution, resolution), bg_color), (0, 1, 0, 1)
    
    mesh_bounds_min, mesh_bounds_max = get_meshes_bounds(meshes)
    
    x_min, y_min, z_min = mesh_bounds_min
    x_max, y_max, z_max = mesh_bounds_max
    
    # Use override bounds if provided (for consistent room rendering)
    if bounds_override:
        x_min, x_max, z_min, z_max = bounds_override
    
    extent_x = x_max - x_min
    extent_z = z_max - z_min
    extent = max(extent_x, extent_z) * 1.1
    
    if extent < 1e-6:
        extent = 1.0
    
    margin = 10
    scale = (resolution - 2 * margin) / extent
    
    center_x = (x_min + x_max) / 2
    center_z = (z_min + z_max) / 2
    
    img = Image.new("RGB", (resolution, resolution), bg_color)
    
    # Collect triangles
    triangles = []
    
    for mesh in meshes:
        if len(mesh.vertices) == 0:
            continue
        
        colors = get_mesh_color(mesh)
        
        for face in mesh.faces:
            v0, v1, v2 = mesh.vertices[face]
            c0, c1, c2 = colors[face]
            
            avg_y = (v0[1] + v1[1] + v2[1]) / 3
            avg_color = tuple(int((int(c0[i]) + int(c1[i]) + int(c2[i])) / 3) for i in range(3))
            
            def world_to_img(wx, wz):
                ix = (wx - center_x) * scale + resolution / 2
                iy = resolution / 2 - (wz - center_z) * scale
                return ix, iy
            
            p0 = world_to_img(v0[0], v0[2])
            p1 = world_to_img(v1[0], v1[2])
            p2 = world_to_img(v2[0], v2[2])
            
            triangles.append((avg_y, [p0, p1, p2], avg_color))
    
    triangles.sort(key=lambda x: x[0])
    
    draw = ImageDraw.Draw(img)
    for _, points, color in triangles:
        flat_points = [coord for point in points for coord in point]
        try:
            draw.polygon(flat_points, fill=color)
        except Exception:
            pass
    
    bounds = (float(x_min), float(x_max), float(z_min), float(z_max))
    return img, bounds


def render_orthographic_layout(glb_path: Path, resolution: int = 512):
    """Render scene layout."""
    meshes = load_glb_with_transforms(glb_path)
    logger.info(f"  Loaded {len(meshes)} meshes")
    return render_topdown(meshes, resolution)


def render_room_layout(
    glb_path: Path,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    resolution: int = 512
):
    """Render room layout - clips meshes to room bbox."""
    meshes = load_glb_with_transforms(glb_path)
    
    # Filter and clip meshes to room bbox
    filtered = filter_meshes_to_bbox(meshes, bbox_min, bbox_max)
    
    logger.info(f"    Filtered {len(meshes)} -> {len(filtered)} meshes")
    
    # Use room bbox as bounds for consistent framing
    room_bounds = (bbox_min[0], bbox_max[0], bbox_min[2], bbox_max[2])
    
    return render_topdown(filtered, resolution, bounds_override=room_bounds)


def draw_door_window_overlay(
    img: Image.Image,
    doors: List[Dict],
    windows: List[Dict],
    bbox_bounds: Tuple[float, float, float, float],
    resolution: int,
    margin: int = 10,
    door_color: Tuple[int, int, int] = (255, 100, 100),  # Default salmon
    window_color: Tuple[int, int, int] = (100, 200, 255)  # Default light blue
) -> Image.Image:
    """Draw doors and windows as filled rectangles using taxonomy colors."""
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    
    x_min, x_max, z_min, z_max = bbox_bounds
    
    extent_x = x_max - x_min
    extent_z = z_max - z_min
    extent = max(extent_x, extent_z) * 1.1
    
    if extent < 1e-6:
        return img_copy
    
    scale = (resolution - 2 * margin) / extent
    center_x = (x_min + x_max) / 2
    center_z = (z_min + z_max) / 2
    
    def world_to_img(wx, wz):
        ix = (wx - center_x) * scale + resolution / 2
        iy = resolution / 2 - (wz - center_z) * scale
        return int(ix), int(iy)
    
    for door in doors:
        bbox = door.get("bbox")
        if bbox is None:
            pos = door.get("position")
            if pos and len(pos) >= 3:
                ix, iy = world_to_img(pos[0], pos[2])
                draw.rectangle([ix-5, iy-5, ix+5, iy+5], fill=door_color)
            continue
        
        d_x_min, d_z_min = bbox["min"][0], bbox["min"][2]
        d_x_max, d_z_max = bbox["max"][0], bbox["max"][2]
        
        x1, y1 = world_to_img(d_x_min, d_z_max)
        x2, y2 = world_to_img(d_x_max, d_z_min)
        
        if abs(x2 - x1) < 3:
            mid = (x1 + x2) // 2
            x1, x2 = mid - 3, mid + 3
        if abs(y2 - y1) < 3:
            mid = (y1 + y2) // 2
            y1, y2 = mid - 3, mid + 3
        
        draw.rectangle([min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)], fill=door_color)
    
    for window in windows:
        bbox = window.get("bbox")
        if bbox is None:
            pos = window.get("position")
            if pos and len(pos) >= 3:
                ix, iy = world_to_img(pos[0], pos[2])
                draw.rectangle([ix-5, iy-5, ix+5, iy+5], fill=window_color)
            continue
        
        w_x_min, w_z_min = bbox["min"][0], bbox["min"][2]
        w_x_max, w_z_max = bbox["max"][0], bbox["max"][2]
        
        x1, y1 = world_to_img(w_x_min, w_z_max)
        x2, y2 = world_to_img(w_x_max, w_z_min)
        
        if abs(x2 - x1) < 3:
            mid = (x1 + x2) // 2
            x1, x2 = mid - 3, mid + 3
        if abs(y2 - y1) < 3:
            mid = (y1 + y2) // 2
            y1, y2 = mid - 3, mid + 3
        
        draw.rectangle([min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)], fill=window_color)
    
    return img_copy


def load_taxonomy(taxonomy_path: Path) -> Dict:
    """Load taxonomy and return color mappings."""
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_door_window_colors(taxonomy: Dict) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Get door and window colors from taxonomy."""
    category_to_color = taxonomy.get("category_to_color", {})
    
    door_color = category_to_color.get("Door", [255, 100, 100])
    window_color = category_to_color.get("Window", [100, 200, 255])
    
    return tuple(door_color), tuple(window_color)


def process_one_scene(
    scene_id: str,
    tex_glb_path: Path,
    seg_glb_path: Path,
    scene_metadata: Dict,
    rooms_metadata: List[Dict],
    output_dir: Path,
    resolution: int = 512,
    margin: int = 10,
    door_color: Tuple[int, int, int] = (255, 100, 100),
    window_color: Tuple[int, int, int] = (100, 200, 255)
) -> Tuple[bool, Optional[str]]:
    """Process one scene."""
    try:
        logger.info(f"Generating layouts for scene {scene_id}...")
        
        # Scene layouts
        tex_img, tex_bounds = render_orthographic_layout(tex_glb_path, resolution)
        seg_img, seg_bounds = render_orthographic_layout(seg_glb_path, resolution)
        
        logger.info(f"  Scene bounds: {tex_bounds}")
        
        all_doors = scene_metadata.get("doors", [])
        all_windows = scene_metadata.get("windows", [])
        
        tex_img = draw_door_window_overlay(tex_img, all_doors, all_windows, tex_bounds, resolution, margin, door_color, window_color)
        seg_img = draw_door_window_overlay(seg_img, all_doors, all_windows, seg_bounds, resolution, margin, door_color, window_color)
        
        scene_tex_output = output_dir / "tex" / f"{scene_id}_tex_layout.png"
        scene_seg_output = output_dir / "seg" / f"{scene_id}_seg_layout.png"
        safe_mkdir(scene_tex_output.parent)
        safe_mkdir(scene_seg_output.parent)
        tex_img.save(scene_tex_output)
        seg_img.save(scene_seg_output)
        
        # Room layouts
        for room_meta in rooms_metadata:
            room_type = room_meta.get("room_type")
            room_index = room_meta.get("room_index", 0)
            
            if not room_type:
                continue
            
            room_id = room_meta.get("room_id")
            if not room_id:
                same_type_count = sum(1 for r in rooms_metadata if r.get("room_type") == room_type)
                if same_type_count > 1:
                    room_id = f"{room_type}_{room_index + 1}"
                else:
                    room_id = room_type
            
            room_bbox = room_meta.get("bbox", {})
            if not room_bbox or "min" not in room_bbox or "max" not in room_bbox:
                logger.warning(f"No bbox for room {scene_id}_{room_id}")
                continue
            
            bbox_min = np.array(room_bbox["min"])
            bbox_max = np.array(room_bbox["max"])
            
            logger.info(f"  Room {room_id}:")
            logger.info(f"    BBox: [{bbox_min[0]:.1f},{bbox_min[1]:.1f},{bbox_min[2]:.1f}] to [{bbox_max[0]:.1f},{bbox_max[1]:.1f},{bbox_max[2]:.1f}]")
            
            # Render room with clipping
            room_tex_img, room_tex_bounds = render_room_layout(tex_glb_path, bbox_min, bbox_max, resolution)
            room_seg_img, room_seg_bounds = render_room_layout(seg_glb_path, bbox_min, bbox_max, resolution)
            
            # Get doors/windows for room
            room_doors = room_meta.get("doors", [])
            room_windows = room_meta.get("windows", [])
            
            if not room_doors:
                room_doors = []
                for door in all_doors:
                    pos = door.get("position", [0,0,0])
                    bbox = door.get("bbox")
                    if bbox:
                        cx = (bbox["min"][0] + bbox["max"][0]) / 2
                        cz = (bbox["min"][2] + bbox["max"][2]) / 2
                    else:
                        cx, cz = pos[0], pos[2]
                    
                    if (bbox_min[0] - 0.5 <= cx <= bbox_max[0] + 0.5 and
                        bbox_min[2] - 0.5 <= cz <= bbox_max[2] + 0.5):
                        room_doors.append(door)
            
            if not room_windows:
                room_windows = []
                for window in all_windows:
                    pos = window.get("position", [0,0,0])
                    bbox = window.get("bbox")
                    if bbox:
                        cx = (bbox["min"][0] + bbox["max"][0]) / 2
                        cz = (bbox["min"][2] + bbox["max"][2]) / 2
                    else:
                        cx, cz = pos[0], pos[2]
                    
                    if (bbox_min[0] - 0.5 <= cx <= bbox_max[0] + 0.5 and
                        bbox_min[2] - 0.5 <= cz <= bbox_max[2] + 0.5):
                        room_windows.append(window)
            
            logger.info(f"    Doors: {len(room_doors)}, Windows: {len(room_windows)}")
            
            room_tex_img = draw_door_window_overlay(room_tex_img, room_doors, room_windows, room_tex_bounds, resolution, margin, door_color, window_color)
            room_seg_img = draw_door_window_overlay(room_seg_img, room_doors, room_windows, room_seg_bounds, resolution, margin, door_color, window_color)
            
            room_tex_output = output_dir / "tex" / f"{scene_id}_{room_id}_tex_layout.png"
            room_seg_output = output_dir / "seg" / f"{scene_id}_{room_id}_seg_layout.png"
            room_tex_img.save(room_tex_output)
            room_seg_img.save(room_seg_output)
        
        return True, None
    
    except Exception as e:
        logger.exception(f"Failed: {e}")
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Render layouts")
    parser.add_argument("--geometry-dir", required=True, help="Directory containing geometry GLB files")
    parser.add_argument("--metadata-dir", required=True, help="Directory containing metadata JSON files")
    parser.add_argument("--taxonomy", required=True, help="Path to taxonomy.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for layout images")
    parser.add_argument("--scene-list", type=str, default=None, help="Path to text file with scene IDs (one per line)")
    parser.add_argument("--resolution", type=int, default=512, help="Output image resolution")
    parser.add_argument("--hpc", action="store_true", help="Enable HPC mode with Xvfb virtual display")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of scenes to process")
    args = parser.parse_args()
    
    # Set up HPC rendering only if --hpc flag is set
    if args.hpc:
        if not setup_hpc_rendering():
            logger.error("Failed to set up HPC rendering, exiting")
            return
    
    geometry_dir = Path(args.geometry_dir)
    metadata_dir = Path(args.metadata_dir)
    taxonomy_path = Path(args.taxonomy)
    output_dir = Path(args.output_dir)
    
    try:
        # Load taxonomy for colors
        taxonomy = load_taxonomy(taxonomy_path)
        door_color, window_color = get_door_window_colors(taxonomy)
        logger.info(f"Door color: {door_color}, Window color: {window_color}")
        
        # Determine which scenes to process
        if args.scene_list:
            # Load scene IDs from file
            scene_list_path = Path(args.scene_list)
            if not scene_list_path.exists():
                logger.error(f"Scene list not found: {scene_list_path}")
                return
            
            scene_ids = load_scene_list(scene_list_path)
            logger.info(f"Loaded {len(scene_ids)} scene IDs from {scene_list_path}")
        else:
            # Discover scenes from metadata directory
            scene_meta_files = list((metadata_dir / "scenes").glob("*.json"))
            if not scene_meta_files:
                logger.error(f"No scene metadata found in {metadata_dir / 'scenes'}")
                return
            scene_ids = [f.stem for f in scene_meta_files]
            logger.info(f"Discovered {len(scene_ids)} scenes from metadata directory")
        
        if args.limit:
            scene_ids = scene_ids[:args.limit]
        
        logger.info(f"Processing {len(scene_ids)} scenes...")
        
        success_count = 0
        skip_count = 0
        
        for i, scene_id in enumerate(scene_ids, 1):
            # Load scene metadata
            scene_meta_path = metadata_dir / "scenes" / f"{scene_id}.json"
            if not scene_meta_path.exists():
                logger.warning(f"Scene metadata not found: {scene_meta_path}")
                skip_count += 1
                continue
            
            with open(scene_meta_path, "r") as f:
                scene_metadata = json.load(f)
            
            # Check for GLB files
            tex_glb = geometry_dir / "tex" / f"{scene_id}_tex.glb"
            seg_glb = geometry_dir / "seg" / f"{scene_id}_seg.glb"
            
            if not tex_glb.exists() or not seg_glb.exists():
                logger.warning(f"GLB not found for {scene_id}")
                skip_count += 1
                continue
            
            # Load room metadata
            rooms_metadata = []
            rooms_dir = metadata_dir / "rooms"
            if rooms_dir.exists():
                for room_meta_path in rooms_dir.glob(f"{scene_id}_*.json"):
                    with open(room_meta_path, "r") as f:
                        room_data = json.load(f)
                        if room_data.get("scene_id") == scene_id:
                            rooms_metadata.append(room_data)
            
            success, error = process_one_scene(
                scene_id, tex_glb, seg_glb, scene_metadata, rooms_metadata,
                output_dir, args.resolution, door_color=door_color, window_color=window_color
            )
            
            if success:
                success_count += 1
                logger.info(f"[{i}/{len(scene_ids)}] ✓ {scene_id}")
            else:
                logger.warning(f"[{i}/{len(scene_ids)}] ✗ {scene_id}: {error}")
        
        logger.info(f"\nDone: {success_count}/{len(scene_ids)} succeeded, {skip_count} skipped")
    
    finally:
        # Clean up HPC rendering only if it was set up
        if args.hpc:
            cleanup_hpc_rendering()


if __name__ == "__main__":
    main()