#!/usr/bin/env python3
"""
collect_all.py
---------------
Stage 5 of the data pipeline.
Scans the 3D-FRONT-derived dataset structure and builds a unified manifest CSV
linking every layout, graph, and POV (seg/tex) for both scenes and rooms.

Each manifest row represents one unique training sample:
    layout  ← target to be generated
    POV + graph  ← conditioning context

The function records extensive statistics about missing or invalid samples.
"""

import os
import csv
from pathlib import Path
from tqdm import tqdm
import pandas as pd


def collect_all(data_root: str, output_manifest: str):
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")

    manifest_rows = []

    fieldnames = [
        "sample_id",
        "sample_type",
        "scene_id",
        "room_id",
        "pov_type",
        "viewpoint",
        "pov_image",
        "pov_embedding",
        "graph_text",
        "graph_embedding",
        "layout_image",
        "layout_embedding",
        "is_empty",
    ]

    n_scenes = n_rooms = n_povs = 0
    n_missing_layout = n_missing_graph = 0
    n_valid_scene = n_valid_room = 0
    n_skipped_room = n_skipped_scene = 0

    # -------------------------------------------------------------------------
    # Traverse all scenes
    # -------------------------------------------------------------------------
    scene_dirs = sorted([d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("scene")])
    if not scene_dirs:
        raise RuntimeError(f"No scene directories found under {data_root}")

    for scene_dir in tqdm(scene_dirs, desc="Scanning scenes"):
        scene_id = scene_dir.name
        n_scenes += 1
        rooms_root = scene_dir / "rooms"
        layouts_root = scene_dir / "layouts"
        scene_layout_path = layouts_root / f"{scene_id}_scene_layout.png"
        scene_graph_path = scene_dir / f"{scene_id}_scene_graph.txt"

        # -------------------------------------------------
        # Scene-level sample
        # -------------------------------------------------
        scene_ok = True
        if not scene_layout_path.exists():
            n_missing_layout += 1
            scene_ok = False
        if not scene_graph_path.exists():
            n_missing_graph += 1
            scene_ok = False

        if scene_ok:
            n_valid_scene += 1
            layout_emb = scene_layout_path.with_name(f"{scene_id}_scene_layout_emb.pt")
            manifest_rows.append({
                "sample_id": f"{scene_id}_scene",
                "sample_type": "scene",
                "scene_id": scene_id,
                "room_id": "",
                "pov_type": "",
                "viewpoint": "",
                "pov_image": "",
                "pov_embedding": "",
                "graph_text": str(scene_graph_path.resolve()),
                "graph_embedding": str(scene_graph_path.with_suffix(".pt").resolve())
                    if scene_graph_path.with_suffix(".pt").exists() else "",
                "layout_image": str(scene_layout_path.resolve()),
                "layout_embedding": str(layout_emb.resolve()) if layout_emb.exists() else "",
                "is_empty": 0,
            })
        else:
            n_skipped_scene += 1

        # -------------------------------------------------
        # Room-level samples
        # -------------------------------------------------
        if not rooms_root.exists():
            continue

        for room_dir in sorted(rooms_root.iterdir()):
            if not room_dir.is_dir():
                continue

            room_id = room_dir.name
            try:
                room_id_int = int(room_id)
            except ValueError:
                room_id_int = room_id

            n_rooms += 1
            layout_path = room_dir / "layouts" / f"{scene_id}_{room_id}_room_seg_layout.png"
            graph_path = room_dir / "layouts" / f"{scene_id}_{room_id}_graph.txt"

            if not layout_path.exists() or not graph_path.exists():
                n_skipped_room += 1
                if not layout_path.exists():
                    n_missing_layout += 1
                if not graph_path.exists():
                    n_missing_graph += 1
                continue

            n_valid_room += 1
            layout_emb = layout_path.with_name(f"{scene_id}_{room_id}_layout_emb.pt")

            # -------------------------------------------------
            # Find POVs
            # -------------------------------------------------
            for pov_type in ["seg", "tex"]:
                pov_dir = room_dir / "povs" / pov_type
                if not pov_dir.exists():
                    continue

                for pov_img in sorted(pov_dir.glob(f"{scene_id}_{room_id}_v*_pov_{pov_type}.png")):
                    n_povs += 1
                    fname = pov_img.stem
                    parts = fname.split("_")
                    view_token = [p for p in parts if p.startswith("v")]
                    viewpoint = view_token[0] if view_token else "v00"
                    sample_id = f"{scene_id}_{room_id}_{pov_type}_{viewpoint}"

                    pov_emb = pov_img.with_suffix(".pt")
                    graph_emb = graph_path.with_suffix(".pt")

                    manifest_rows.append({
                        "sample_id": sample_id,
                        "sample_type": "room",
                        "scene_id": scene_id,
                        "room_id": room_id_int,
                        "pov_type": pov_type,
                        "viewpoint": viewpoint,
                        "pov_image": str(pov_img.resolve()),
                        "pov_embedding": str(pov_emb.resolve()) if pov_emb.exists() else "",
                        "graph_text": str(graph_path.resolve()),
                        "graph_embedding": str(graph_emb.resolve()) if graph_emb.exists() else "",
                        "layout_image": str(layout_path.resolve()),
                        "layout_embedding": str(layout_emb.resolve()) if layout_emb.exists() else "",
                        "is_empty": 0,
                    })

    # -------------------------------------------------------------------------
    # Save manifest
    # -------------------------------------------------------------------------
    output_manifest = Path(output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows, columns=fieldnames).to_csv(output_manifest, index=False)

    # -------------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------------
    print("\n[SUMMARY]")
    print(f"Scenes total: {n_scenes}")
    print(f"Rooms total:  {n_rooms}")
    print(f"POVs total:   {n_povs}")
    print(f"Valid scenes: {n_valid_scene}, Skipped scenes: {n_skipped_scene}")
    print(f"Valid rooms:  {n_valid_room}, Skipped rooms:  {n_skipped_room}")
    print(f"Missing graphs:  {n_missing_graph}")
    print(f"Missing layouts: {n_missing_layout}")
    print(f"Final manifest entries: {len(manifest_rows)}")
    print(f"[INFO] Wrote manifest → {output_manifest}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Collect unified dataset manifest.")
    ap.add_argument("--data_root", required=True, help="Path to dataset root directory")
    ap.add_argument("--output_manifest", required=True, help="Output CSV manifest path")
    args = ap.parse_args()
    collect_all(args.data_root, args.output_manifest)
