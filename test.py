#!/usr/bin/env python3
"""
Simple diagnostic script to test GLB file rendering with Open3D.
Tests different approaches to loading and rendering GLB files.
"""

import sys
import numpy as np
import open3d as o3d
import trimesh
from pathlib import Path

def test_direct_open3d_load(glb_path):
    """Test 1: Try loading GLB directly with Open3D"""
    print("\n=== Test 1: Direct Open3D Load ===")
    try:
        # Open3D can load some GLB files directly
        mesh = o3d.io.read_triangle_mesh(str(glb_path))
        if mesh.is_empty():
            print("ERROR: Mesh is empty after loading")
            return False
        
        print(f"Success! Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
        
        # Try to visualize
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh], window_name="Direct Open3D Load")
        return True
    except Exception as e:
        print(f"ERROR: Failed to load with Open3D directly: {e}")
        return False

def test_trimesh_to_open3d_simple(glb_path):
    """Test 2: Load with trimesh and convert to Open3D (simple approach)"""
    print("\n=== Test 2: Trimesh to Open3D (Simple) ===")
    try:
        # Load with trimesh
        scene = trimesh.load(str(glb_path), process=False, maintain_order=True)
        
        if isinstance(scene, trimesh.Scene):
            print(f"Loaded scene with {len(scene.geometry)} geometries")
            
            # Convert all geometries to Open3D
            o3d_meshes = []
            for name, geom in scene.geometry.items():
                if isinstance(geom, trimesh.Trimesh):
                    # Simple conversion
                    o3d_mesh = o3d.geometry.TriangleMesh()
                    o3d_mesh.vertices = o3d.utility.Vector3dVector(geom.vertices)
                    o3d_mesh.triangles = o3d.utility.Vector3iVector(geom.faces)
                    
                    # Try to get colors
                    if hasattr(geom.visual, 'vertex_colors'):
                        colors = geom.visual.vertex_colors[:, :3] / 255.0
                        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
                    
                    o3d_mesh.compute_vertex_normals()
                    o3d_meshes.append(o3d_mesh)
                    print(f"  Converted geometry '{name}': {len(geom.vertices)} vertices")
            
            if o3d_meshes:
                o3d.visualization.draw_geometries(o3d_meshes, window_name="Trimesh to Open3D Simple")
                return True
        else:
            # Single mesh
            print("Loaded single mesh")
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(scene.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(scene.faces)
            
            if hasattr(scene.visual, 'vertex_colors'):
                colors = scene.visual.vertex_colors[:, :3] / 255.0
                o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            
            o3d_mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries([o3d_mesh], window_name="Trimesh to Open3D Simple")
            return True
            
    except Exception as e:
        print(f"ERROR: Failed trimesh to Open3D conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trimesh_export_ply(glb_path):
    """Test 3: Convert GLB to PLY via trimesh, then load with Open3D"""
    print("\n=== Test 3: GLB -> PLY -> Open3D ===")
    try:
        # Load with trimesh
        scene = trimesh.load(str(glb_path), process=False, maintain_order=True)
        
        # Export to PLY
        ply_path = glb_path.with_suffix('.ply')
        if isinstance(scene, trimesh.Scene):
            # Merge scene into single mesh
            mesh = scene.dump(concatenate=True)
        else:
            mesh = scene
        
        mesh.export(ply_path)
        print(f"Exported to PLY: {ply_path}")
        
        # Load PLY with Open3D
        o3d_mesh = o3d.io.read_triangle_mesh(str(ply_path))
        if o3d_mesh.is_empty():
            print("ERROR: PLY mesh is empty")
            return False
        
        print(f"Loaded PLY: {len(o3d_mesh.vertices)} vertices, {len(o3d_mesh.triangles)} triangles")
        o3d_mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([o3d_mesh], window_name="GLB via PLY")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed GLB->PLY->Open3D: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scene_graph_conversion(glb_path):
    """Test 4: Convert with scene graph transforms (like your code)"""
    print("\n=== Test 4: Scene Graph Conversion ===")
    try:
        scene = trimesh.load(str(glb_path), process=False, maintain_order=True)
        
        if not isinstance(scene, trimesh.Scene):
            print("Not a scene, wrapping in scene")
            new_scene = trimesh.Scene()
            new_scene.add_geometry(scene)
            scene = new_scene
        
        o3d_meshes = []
        
        # Process scene graph
        for node_name in scene.graph.nodes_geometry:
            try:
                transform, geometry_name = scene.graph.get(node_name)
                
                # Get geometry
                if geometry_name in scene.geometry:
                    geometry = scene.geometry[geometry_name]
                elif node_name in scene.geometry:
                    geometry = scene.geometry[node_name]
                else:
                    continue
                
                if isinstance(geometry, trimesh.Trimesh):
                    # Create Open3D mesh
                    o3d_mesh = o3d.geometry.TriangleMesh()
                    
                    # Apply transform to vertices
                    vertices = geometry.vertices
                    if not np.allclose(transform, np.eye(4)):
                        vertices_hom = np.column_stack([vertices, np.ones(len(vertices))])
                        vertices = (transform @ vertices_hom.T).T[:, :3]
                    
                    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
                    o3d_mesh.triangles = o3d.utility.Vector3iVector(geometry.faces)
                    
                    # Colors
                    if hasattr(geometry.visual, 'vertex_colors'):
                        colors = geometry.visual.vertex_colors[:, :3] / 255.0
                        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
                    
                    o3d_mesh.compute_vertex_normals()
                    o3d_meshes.append(o3d_mesh)
                    print(f"  Converted node '{node_name}': {len(vertices)} vertices")
                    
            except Exception as e:
                print(f"  Failed to convert node {node_name}: {e}")
                continue
        
        if o3d_meshes:
            print(f"Total meshes converted: {len(o3d_meshes)}")
            o3d.visualization.draw_geometries(o3d_meshes, window_name="Scene Graph Conversion")
            return True
        else:
            print("ERROR: No meshes were converted")
            return False
            
    except Exception as e:
        print(f"ERROR: Failed scene graph conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_glb_file(glb_path):
    """Analyze the GLB file structure"""
    print("\n=== GLB File Analysis ===")
    try:
        scene = trimesh.load(str(glb_path), process=False, maintain_order=True)
        
        print(f"Type: {type(scene)}")
        
        if isinstance(scene, trimesh.Scene):
            print(f"Scene with {len(scene.geometry)} geometries")
            print(f"Graph nodes: {len(scene.graph.nodes_geometry)}")
            
            # List geometries
            for i, (name, geom) in enumerate(scene.geometry.items()):
                if i < 5:  # Show first 5
                    print(f"  - {name}: {type(geom).__name__} with {len(geom.vertices) if hasattr(geom, 'vertices') else 0} vertices")
                    if hasattr(geom, 'visual'):
                        print(f"    Visual type: {type(geom.visual).__name__}")
                        if hasattr(geom.visual, 'material'):
                            print(f"    Material: {type(geom.visual.material).__name__ if geom.visual.material else 'None'}")
        else:
            print(f"Single mesh with {len(scene.vertices)} vertices")
            if hasattr(scene, 'visual'):
                print(f"Visual type: {type(scene.visual).__name__}")
        
        # Check bounds
        bounds = scene.bounds
        print(f"Bounds: min={bounds[0]}, max={bounds[1]}")
        print(f"Size: {bounds[1] - bounds[0]}")
        
    except Exception as e:
        print(f"ERROR analyzing GLB: {e}")
        import traceback
        traceback.print_exc()

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_glb_open3d.py <path_to_glb_file>")
        sys.exit(1)
    
    # Convert to Path object
    glb_path = Path(sys.argv[1])
    
    if not glb_path.exists():
        print(f"ERROR: File not found: {glb_path}")
        sys.exit(1)
    
    print(f"Testing GLB file: {glb_path}")
    
    # First analyze the file
    analyze_glb_file(glb_path)
    
    # Run tests
    print("\n" + "="*50)
    print("Running rendering tests...")
    print("="*50)
    
    # Test 1: Direct Open3D load
    test_direct_open3d_load(glb_path)
    
    # Test 2: Simple trimesh to Open3D
    test_trimesh_to_open3d_simple(glb_path)
    
    # Test 3: Via PLY export
    test_trimesh_export_ply(glb_path)
    
    # Test 4: Scene graph conversion (like your code)
    test_scene_graph_conversion(glb_path)

if __name__ == "__main__":
    main()