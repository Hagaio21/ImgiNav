#!/usr/bin/env python3
"""
Headless GLB renderer using Open3D.
Camera orientation and projection untouched.
Rooms always fully in frame with zero margin.
Glare minimized by disabling specular and using flat shading.
"""

import json
import numpy as np
from pathlib import Path
import argparse
from dataclasses import dataclass
import open3d as o3d
from PIL import Image, ImageDraw

@dataclass
class RoomBounds:
    room_id: str
    room_type: str
    min_bound: np.ndarray
    max_bound: np.ndarray
    center: np.ndarray
    size: np.ndarray

class HeadlessGLBRenderer:
    
    def __init__(self, metadata_path: str, output_dir: str = "renders"):
        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.up_direction = np.array(self.metadata.get('up_direction', [0, 1, 0]))
        print(f"Up direction from metadata: {self.up_direction}")
        
        self.rooms = self._parse_rooms()
        self.doors = self.metadata.get('doors', [])
    
    def _parse_rooms(self) -> list[RoomBounds]:
        rooms = []
        room_data = self.metadata.get('rooms', [])
        
        for idx, room in enumerate(room_data):
            room_id = str(room.get('room_id', idx))
            room_type = room.get('room_type', f'Room_{idx}')
            
            bounds = room.get('room_bbox', {})
            min_bound = np.array(bounds.get('min', [0, 0, 0]))
            max_bound = np.array(bounds.get('max', [1, 1, 1]))
            
            rooms.append(RoomBounds(
                room_id=room_id,
                room_type=room_type,
                min_bound=min_bound,
                max_bound=max_bound,
                center=(min_bound + max_bound) / 2,
                size=max_bound - min_bound
            ))
        
        print(f"Parsed {len(rooms)} rooms from metadata")
        return rooms
    
    def load_glb_with_trimesh(self, glb_path: str, debug: bool = True) -> o3d.geometry.TriangleMesh:
        import trimesh
        
        glb_path = Path(glb_path)
        if not glb_path.exists():
            raise FileNotFoundError(f"GLB file not found: {glb_path}")
        
        print(f"Loading GLB: {glb_path}")
        
        scene = trimesh.load(str(glb_path), force="scene", process=False)
        
        o3d_meshes = []
        
        if isinstance(scene, trimesh.Scene):
            if debug:
                print(f"  Scene has {len(scene.geometry)} geometries")
            
            for node_name in scene.graph.nodes_geometry:
                try:
                    transform, geometry_name = scene.graph.get(node_name)
                    
                    geometry = None
                    if geometry_name in scene.geometry:
                        geometry = scene.geometry[geometry_name]
                    elif node_name in scene.geometry:
                        geometry = scene.geometry[node_name]
                    
                    if geometry is None or not isinstance(geometry, trimesh.Trimesh):
                        continue
                    
                    mesh_world = geometry.copy()
                    if not np.allclose(transform, np.eye(4)):
                        mesh_world.apply_transform(transform)
                    
                    o3d_mesh = o3d.geometry.TriangleMesh()
                    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh_world.vertices.astype(np.float64))
                    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh_world.faces.astype(np.int32))
                    
                    colors_found = False
                    
                    if hasattr(mesh_world.visual, 'vertex_colors') and mesh_world.visual.vertex_colors is not None:
                        vcolors = np.array(mesh_world.visual.vertex_colors)
                        if len(vcolors) == len(mesh_world.vertices) and vcolors.shape[1] >= 3:
                            colors = vcolors[:, :3].astype(np.float64)
                            if colors.max() > 1.0:
                                colors = colors / 255.0
                            if not np.allclose(colors[0], colors).all():
                                o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
                                colors_found = True
                    
                    if not colors_found:
                        o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])
                    
                    o3d_mesh.compute_vertex_normals()
                    o3d_meshes.append(o3d_mesh)
                    
                except:
                    continue
        
        elif isinstance(scene, trimesh.Trimesh):
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(scene.vertices.astype(np.float64))
            o3d_mesh.triangles = o3d.utility.Vector3iVector(scene.faces.astype(np.int32))
            
            if hasattr(scene.visual, 'vertex_colors') and scene.visual.vertex_colors is not None:
                colors = np.array(scene.visual.vertex_colors)[:, :3].astype(np.float64)
                if colors.max() > 1.0:
                    colors = colors / 255.0
                o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            else:
                o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])
            
            o3d_mesh.compute_vertex_normals()
            o3d_meshes.append(o3d_mesh)
        
        if not o3d_meshes:
            raise ValueError(f"No valid meshes found in {glb_path}")
        
        combined = o3d_meshes[0]
        for m in o3d_meshes[1:]:
            combined += m
        
        combined.compute_vertex_normals()
        
        if debug and combined.has_vertex_colors():
            colors = np.asarray(combined.vertex_colors)
            print(f"  Colors: min={colors.min():.3f}, max={colors.max():.3f}")
        
        print(f"  Total: {len(combined.vertices)} vertices, {len(combined.triangles)} triangles")
        
        return combined

    def replace_background_pixels(self, image_path: Path, background_color: list) -> None:
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img, dtype=np.float32)
        bg = (np.array(background_color) * 255).astype(np.float32)
        mask = np.all(img_array == bg, axis=2)
        img_array[mask] = [255, 255, 255]
        Image.fromarray(img_array.astype(np.uint8)).save(image_path)

    def draw_doors_2d(self, image_path: Path, target_min: np.ndarray, target_max: np.ndarray, margin_percent: float):
        if not self.doors:
            return
        try:
            img = Image.open(image_path).convert("RGB")
            width, height = img.size
            draw = ImageDraw.Draw(img)
            fill_color = (194, 178, 128)
            outline_color = (100, 90, 60)
            target_center = (target_min + target_max) / 2
            target_size = target_max - target_min
            max_extent = max(target_size[0], target_size[2])
            margin_factor = 1.0 + (margin_percent / 100.0)
            view_size_3d = max_extent * margin_factor
            view_min_x = target_center[0] - (view_size_3d / 2)
            view_max_x = target_center[0] + (view_size_3d / 2)
            view_min_z = target_center[2] - (view_size_3d / 2)
            view_max_z = target_center[2] + (view_size_3d / 2)
            doors_drawn = 0
            for door in self.doors:
                bbox = door.get('bbox', {})
                if not bbox:
                    continue
                d_min = np.array(bbox['min'])
                d_max = np.array(bbox['max'])
                if (d_max[0] < view_min_x or d_min[0] > view_max_x or
                    d_max[2] < view_min_z or d_min[2] > view_max_z):
                    continue
                x0 = (d_min[0] - view_min_x) / (view_max_x - view_min_x) * width
                x1 = (d_max[0] - view_min_x) / (view_max_x - view_min_x) * width
                y0 = (d_min[2] - view_min_z) / (view_max_z - view_min_z) * height
                y1 = (d_max[2] - view_min_z) / (view_max_z - view_min_z) * height
                draw.rectangle([x0, y0, x1, y1], fill=fill_color, outline=outline_color)
                doors_drawn += 1
            img.save(image_path)
            if doors_drawn > 0:
                print(f"    Overlaid {doors_drawn} doors")
        except Exception as e:
            print(f"    Error drawing doors: {e}")

    def apply_orthographic_projection(self, ctr, width, height):
        params = ctr.convert_to_pinhole_camera_parameters()
        fx = fy = 100000.0
        params.intrinsic.set_intrinsics(width, height, fx, fy, width / 2, height / 2)
        ctr.convert_from_pinhole_camera_parameters(params)

    def render_birds_eye_view(
        self,
        mesh: o3d.geometry.TriangleMesh,
        target_min_bound: np.ndarray,
        target_max_bound: np.ndarray,
        output_path: Path,
        width: int = 500,
        height: int = 500,
        margin_percent: float = 0.0,
        background_color: list = None
    ) -> str:
        
        if background_color is None:
            background_color = [1.0, 0.0, 1.0]
        
        if len(mesh.triangles) == 0:
            print(f"    Warning: No triangles")
            return None
        
        target_center = (target_min_bound + target_max_bound) / 2
        target_size = target_max_bound - target_min_bound
        
        max_extent = max(target_size[0], target_size[2])
        padded_extent = max_extent  # no margin
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height, visible=False)
        
        vis.add_geometry(mesh)
        
        opt = vis.get_render_option()
        opt.background_color = np.array(background_color)
        opt.mesh_show_back_face = True
        opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
        
        opt.light_on = True
        opt.specular_power = 0.0
        opt.shade_flat = True
        opt.point_size = 1.0
        
        ctr = vis.get_view_control()
        vis.reset_view_point(True)
        
        ctr.set_front([0, 1, 0])
        ctr.set_lookat(target_center.tolist())
        ctr.set_up([0, 0, -1])
        
        self.apply_orthographic_projection(ctr, width, height)
        
        zoom = max_extent / (max_extent * 2)
        zoom = 0.5
        
        ctr.set_zoom(zoom)
        
        print(f"    Target size: {target_size}, Zero margin zoom: {zoom:.3f}")
        
        for _ in range(4):
            vis.poll_events()
            vis.update_renderer()
        
        vis.capture_screen_image(str(output_path), do_render=True)
        vis.destroy_window()
        
        self.replace_background_pixels(output_path, background_color)
        self.draw_doors_2d(output_path, target_min_bound, target_max_bound, margin_percent)
        
        print(f"  Saved: {output_path}")
        return str(output_path)
    
    def render_scene(
        self,
        glb_path: str,
        output_prefix: str = "scene",
        width: int = 500,
        height: int = 500,
        margin_percent: float = 0.0,
        background_color: list = None
    ) -> str:
        mesh = self.load_glb_with_trimesh(glb_path, debug=True)
        
        bbox = mesh.get_axis_aligned_bounding_box()
        min_bound = np.array(bbox.min_bound)
        max_bound = np.array(bbox.max_bound)
        
        print(f"\nRendering full scene...")
        print(f"  Mesh bounds: min={min_bound}, max={max_bound}")
        
        output_path = self.output_dir / f"{output_prefix}_birds_eye.png"
        return self.render_birds_eye_view(mesh, min_bound, max_bound, output_path, width, height, margin_percent, background_color)
    
    def render_rooms(
        self,
        glb_path: str,
        output_prefix: str = "room",
        width: int = 500,
        height: int = 500,
        margin_percent: float = 0.0,
        background_color: list = None
    ) -> list[str]:
        
        if not self.rooms:
            print("No rooms found")
            return []
        
        mesh = self.load_glb_with_trimesh(glb_path, debug=False)
        
        results = []
        
        print(f"\nRendering {len(self.rooms)} rooms...")
        
        for room in self.rooms:
            safe_type = "".join(c if c.isalnum() else '_' for c in room.room_type)
            output_path = self.output_dir / f"{output_prefix}_{room.room_id}_{safe_type}_birds_eye.png"
            
            print(f"\n  Room {room.room_id}: {room.room_type}")
            print(f"    Bounds: min={room.min_bound}, max={room.max_bound}")
            
            try:
                path = self.render_birds_eye_view(
                    mesh,
                    room.min_bound,
                    room.max_bound,
                    output_path,
                    width,
                    height,
                    margin_percent,
                    background_color
                )
                if path:
                    results.append(path)
                
            except Exception as e:
                print(f"    Error: {e}")
                import traceback
                traceback.print_exc()
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Headless GLB renderer (Open3D only)")
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--color-glb", type=str, required=True)
    parser.add_argument("--texture-glb", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="renders")
    parser.add_argument("--resolution", type=int, default=500)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--background", type=str, default="1,0,1")
    parser.add_argument("--scene-only", action="store_true")
    parser.add_argument("--rooms-only", action="store_true")
    
    args = parser.parse_args()
    
    width = height = args.resolution
    
    try:
        bg_color = [float(x) for x in args.background.split(',')]
        if len(bg_color) != 3:
            bg_color = [1.0, 0.0, 1.0]
    except:
        bg_color = [1.0, 0.0, 1.0]
    
    for p in [args.metadata, args.color_glb, args.texture_glb]:
        if not Path(p).exists():
            print(f"File not found: {p}")
            return 1
    
    print("=" * 60)
    print("HEADLESS GLB RENDERER (Open3D)")
    print("=" * 60)
    print(f"Resolution: {width}x{height}")
    print(f"Margin: {args.margin}%")
    print(f"Background: {bg_color}")
    
    renderer = HeadlessGLBRenderer(args.metadata, args.output_dir)
    
    color_dir = Path(args.output_dir) / "color_segmented"
    texture_dir = Path(args.output_dir) / "textured"
    color_dir.mkdir(parents=True, exist_ok=True)
    texture_dir.mkdir(parents=True, exist_ok=True)
    
    outputs = []
    
    print("\n" + "=" * 60)
    print("COLOR-SEGMENTED GLB")
    print("=" * 60)
    renderer.output_dir = color_dir
    
    if not args.rooms_only:
        try:
            outputs.append(renderer.render_scene(args.color_glb, "scene", width, height, args.margin, bg_color))
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    if not args.scene_only:
        try:
            outputs.extend(renderer.render_rooms(args.color_glb, "room", width, height, args.margin, bg_color))
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("TEXTURED GLB")
    print("=" * 60)
    renderer.output_dir = texture_dir
    
    if not args.rooms_only:
        try:
            outputs.append(renderer.render_scene(args.texture_glb, "scene", width, height, args.margin, bg_color))
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    if not args.scene_only:
        try:
            outputs.extend(renderer.render_rooms(args.texture_glb, "room", width, height, args.margin, bg_color))
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"DONE. Rendered {len([o for o in outputs if o])} images")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main())
