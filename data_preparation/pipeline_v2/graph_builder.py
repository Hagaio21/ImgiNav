#!/usr/bin/env python3
"""
Graph builder for pipeline v2.
Wraps existing graph building functionality from build_graphs.py.
"""

import sys
import shutil
from pathlib import Path

# Add project root to path for imports
# __file__ is at: .../ImgiNav/data_preparation/pipeline_v2/graph_builder.py
# Project root is: .../ImgiNav/
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent  # Go up from pipeline_v2 -> data_preparation -> ImgiNav
sys.path.insert(0, str(project_root))

from data_preparation.build_graphs import build_room_graph_from_layout as _build_room_graph
from common.taxonomy import Taxonomy


def build_room_graph_from_layout(scene_id: str, room_id: str, layout_path: Path,
                                 taxonomy: Taxonomy, output_dir: Path):
    """
    Build room graph from segmentation layout image.
    
    Args:
        scene_id: Scene identifier
        room_id: Room identifier (use "0" for scene-level)
        layout_path: Path to segmentation layout image
        taxonomy: Taxonomy object
        output_dir: Directory to save graph files
    """
    # Get color to label mapping from taxonomy
    color_to_label = taxonomy.get_color_to_label_dict()
    
    # Build graph using existing function
    # Note: build_room_graph_from_layout saves files relative to layout_path
    # We need to temporarily change the output location or copy files
    
    # Call the original function - it will save next to layout_path
    result = _build_room_graph(
        scene_id, room_id, layout_path, color_to_label, taxonomy=taxonomy
    )
    
    # Move graph files to output directory if they were created elsewhere
    graph_json = layout_path.parent / f"{scene_id}_{room_id}_graph.json"
    graph_txt = layout_path.parent / f"{scene_id}_{room_id}_graph.txt"
    graph_vis = layout_path.parent / f"{scene_id}_{room_id}_graph_vis.png"
    
    if graph_json.exists() and graph_json.parent != output_dir:
        shutil.move(str(graph_json), str(output_dir / graph_json.name))
    if graph_txt.exists() and graph_txt.parent != output_dir:
        shutil.move(str(graph_txt), str(output_dir / graph_txt.name))
    if graph_vis.exists() and graph_vis.parent != output_dir:
        shutil.move(str(graph_vis), str(output_dir / graph_vis.name))
    
    return result

