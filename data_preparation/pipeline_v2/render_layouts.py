#!/usr/bin/env python3
"""
Render layouts only (scene and room layouts, RGB and segmentation).
No POV rendering, no geometry export.
"""

import os
import time

# Try to import xvfbwrapper (like old pipeline)
try:
    from xvfbwrapper import Xvfb
    XVFBWRAPPER_AVAILABLE = True
except ImportError:
    Xvfb = None
    XVFBWRAPPER_AVAILABLE = False

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

# Global variable to hold Xvfb instance
VFB = None

import sys

# Add project root to path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from common.taxonomy import Taxonomy
from data_preparation.pipeline_v2.scene_loader import load_front_scene, extract_rooms_from_scene
from data_preparation.pipeline_v2.renderer import render_layout_rgb, render_layout_seg

# Set Open3D verbosity to errors only
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def main():
    parser = argparse.ArgumentParser(description="Render 3D-FRONT scene layouts using Open3D")
    parser.add_argument("--scene_json", required=True, help="Path to 3D-FRONT JSON file")
    parser.add_argument("--future_root", required=True, help="Path to 3D-FUTURE model directory")
    parser.add_argument("--output_dir", required=True, help="Output dataset root directory")
    parser.add_argument("--taxonomy", required=True, help="Path to taxonomy.json")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hpc", action="store_true", default=False,
                        help="Run inside Xvfb for headless HPC rendering (like old pipeline)")
    
    args = parser.parse_args()
    
    # Start Xvfb if needed (like old pipeline)
    global VFB
    if args.hpc:
        if XVFBWRAPPER_AVAILABLE:
            try:
                VFB = Xvfb(width=256, height=256, colordepth=24)
                VFB.start()
                os.environ['DISPLAY'] = f':{VFB.new_display}'
                print(f"Started Xvfb virtual display: {os.environ['DISPLAY']}")
            except Exception as e:
                print(f"Warning: Failed to start Xvfb: {e}")
                print("Continuing without Xvfb (may fail if no display available)")
                VFB = None
        else:
            print("Warning: --hpc flag set but xvfbwrapper not installed")
            print("Install with: pip install xvfbwrapper")
            if 'DISPLAY' not in os.environ:
                raise RuntimeError("Cannot render without display. Install xvfbwrapper or use xvfb-run.")
    
    # Load taxonomy
    taxonomy = Taxonomy(Path(args.taxonomy))
    
    # Load scene
    scene_json = Path(args.scene_json)
    scene_id = scene_json.stem
    
    print(f"Loading scene: {scene_id}")
    trimesh_scene = load_front_scene(scene_json, Path(args.future_root), taxonomy)
    
    # Create output directories
    output_dir = Path(args.output_dir)
    layouts_rgb_dir = output_dir / "layouts" / "rgb"
    layouts_seg_dir = output_dir / "layouts" / "seg"
    graphs_dir = output_dir / "graphs"
    
    for d in [layouts_rgb_dir, layouts_seg_dir, graphs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Extract rooms from scene
    print("Extracting rooms from scene...")
    room_scenes = extract_rooms_from_scene(trimesh_scene)
    print(f"  Found {len(room_scenes)} rooms: {list(room_scenes.keys())}")
    
    # Scene-level Layout Pass: RGB
    print("Rendering scene-level layout RGB...")
    try:
        layout_rgb = render_layout_rgb(trimesh_scene, hide_ceilings=True, width=256, height=256)
        layout_rgb_path = layouts_rgb_dir / f"{scene_id}_scene.png"
        Image.fromarray(layout_rgb).save(layout_rgb_path)
        print(f"  Saved scene layout RGB: {layout_rgb_path}")
        if not layout_rgb_path.exists():
            raise FileNotFoundError(f"Scene layout RGB file was not created: {layout_rgb_path}")
    except Exception as e:
        print(f"ERROR: Failed to render scene layout RGB: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Scene-level Layout Pass: Segmentation
    print("Rendering scene-level layout segmentation...")
    try:
        layout_seg = render_layout_seg(trimesh_scene, taxonomy, hide_ceilings=True, width=256, height=256)
        layout_seg_path = layouts_seg_dir / f"{scene_id}_scene.png"
        Image.fromarray(layout_seg).save(layout_seg_path)
        print(f"  Saved scene layout segmentation: {layout_seg_path}")
        if not layout_seg_path.exists():
            raise FileNotFoundError(f"Scene layout segmentation file was not created: {layout_seg_path}")
    except Exception as e:
        print(f"ERROR: Failed to render scene layout segmentation: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Room-level Layout Pass
    print("Rendering room-level layouts...")
    for room_name, room_scene in room_scenes.items():
        # Sanitize room name for filename
        safe_room_name = room_name.replace(" ", "_").replace("/", "_").lower()
        
        print(f"  Processing room: {room_name}")
        
        # Room RGB layout
        try:
            room_layout_rgb = render_layout_rgb(room_scene, hide_ceilings=True, width=256, height=256)
            room_layout_rgb_path = layouts_rgb_dir / f"{scene_id}_{safe_room_name}_room.png"
            Image.fromarray(room_layout_rgb).save(room_layout_rgb_path)
            print(f"    Saved room RGB layout: {room_layout_rgb_path}")
            if not room_layout_rgb_path.exists():
                raise FileNotFoundError(f"Room RGB layout file was not created: {room_layout_rgb_path}")
        except Exception as e:
            print(f"    ERROR: Failed to render room RGB layout for {room_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Room segmentation layout
        try:
            room_layout_seg = render_layout_seg(room_scene, taxonomy, hide_ceilings=True, width=256, height=256)
            room_layout_seg_path = layouts_seg_dir / f"{scene_id}_{safe_room_name}_room.png"
            Image.fromarray(room_layout_seg).save(room_layout_seg_path)
            print(f"    Saved room segmentation layout: {room_layout_seg_path}")
            if not room_layout_seg_path.exists():
                raise FileNotFoundError(f"Room segmentation layout file was not created: {room_layout_seg_path}")
        except Exception as e:
            print(f"    ERROR: Failed to render room segmentation layout for {room_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Build graphs from segmentation layouts
    print("Building graphs...")
    from data_preparation.pipeline_v2.graph_builder import build_room_graph_from_layout as build_graph
    
    # Scene-level graph
    try:
        scene_layout_seg_path = layouts_seg_dir / f"{scene_id}_scene.png"
        if scene_layout_seg_path.exists():
            build_graph(
                scene_id, "scene", scene_layout_seg_path, taxonomy, graphs_dir
            )
            print("  Scene graph built successfully")
    except Exception as e:
        print(f"  Warning: Failed to build scene graph: {e}")
        import traceback
        traceback.print_exc()
    
    # Room-level graphs
    for room_name, room_scene in room_scenes.items():
        safe_room_name = room_name.replace(" ", "_").replace("/", "_").lower()
        try:
            room_layout_seg_path = layouts_seg_dir / f"{scene_id}_{safe_room_name}_room.png"
            if room_layout_seg_path.exists():
                build_graph(
                    scene_id, room_name, room_layout_seg_path, taxonomy, graphs_dir
                )
                print(f"  Room graph built for {room_name}")
        except Exception as e:
            print(f"  Warning: Failed to build graph for room {room_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Completed layout rendering for scene: {scene_id}")
    print(f"  Scene layouts: RGB and segmentation")
    print(f"  Room layouts: {len(room_scenes)} rooms (RGB and segmentation each)")
    print(f"  Graphs: 1 scene graph + {len(room_scenes)} room graphs")
    
    # Stop Xvfb if we started it
    if VFB is not None:
        try:
            VFB.stop()
            print("Stopped Xvfb virtual display")
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure Xvfb is stopped even on error
        if 'VFB' in globals() and globals()['VFB'] is not None:
            try:
                globals()['VFB'].stop()
            except Exception:
                pass

