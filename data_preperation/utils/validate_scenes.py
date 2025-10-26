#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scan_rooms.py

Scan all JSON scene files in --in_dir and identify which contain rooms.
Outputs two JSONs in --out_dir:
  - valid_files.json: { "scene_file.json": ["LivingRoom", "Bedroom", ...], ... }
  - invalid_files.json: ["scene_file.json", ...]
"""

import json
from pathlib import Path
import argparse
from tqdm import tqdm


def check_rooms(scene_path: Path):
    """Return list of room types if scene exposes rooms, else None."""
    try:
        data = json.loads(scene_path.read_text(encoding="utf-8"))
    except Exception as e:
        return None, f"load error: {e}"

    rooms = data.get("scene", {}).get("room") or data.get("scene", {}).get("rooms")
    if not rooms:  # empty or missing
        return None, None

    room_types = [r.get("type", "UnknownRoom") for r in rooms]
    return room_types, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True,
                    help="Directory with scene JSON files to scan")
    ap.add_argument("--out_dir", required=True,
                    help="Directory to write valid_files.json and invalid_files.json")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.json"))
    valid = {}
    invalid = []

    for sp in tqdm(files, desc="Scanning scenes"):
        room_types, err = check_rooms(sp)
        if err:
            invalid.append(sp.name)
        elif room_types:  # non-empty list of rooms
            valid[sp.name] = room_types
        else:
            invalid.append(sp.name)

    # write outputs
    (out_dir / "valid_files.json").write_text(json.dumps(valid, indent=2), encoding="utf-8")
    (out_dir / "invalid_files.json").write_text(json.dumps(invalid, indent=2), encoding="utf-8")

    print(f"✅ wrote {len(valid)} valid and {len(invalid)} invalid files to {out_dir}")


if __name__ == "__main__":
    main()
