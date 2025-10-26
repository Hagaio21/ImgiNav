import os
import csv
import argparse

def load_empty_map(layouts_csv):
    """Load room emptiness info from layouts.csv into dict[(scene_id, room_id)] = is_empty"""
    empty_map = {}
    if not os.path.isfile(layouts_csv):
        return empty_map
    with open(layouts_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("scene_id", "")
            rid = row.get("room_id", "")
            try:
                empty = int(row.get("is_empty", "0"))
            except ValueError:
                empty = 0
            empty_map[(sid, rid)] = empty
    return empty_map


def collect_povs(root, out_csv, empty_map):
    rows = []
    for scene_id in os.listdir(root):
        scene_dir = os.path.join(root, scene_id)
        if not os.path.isdir(scene_dir):
            continue
        rooms_dir = os.path.join(scene_dir, "rooms")
        if not os.path.isdir(rooms_dir):
            continue

        for room_id in os.listdir(rooms_dir):
            povs_dir = os.path.join(rooms_dir, room_id, "povs")
            if not os.path.isdir(povs_dir):
                continue
            for pov_type in ("seg", "tex"):
                tdir = os.path.join(povs_dir, pov_type)
                if not os.path.isdir(tdir):
                    continue
                for f in os.listdir(tdir):
                    if not f.endswith(".png"):
                        continue
                    path = os.path.join(tdir, f)
                    is_empty = empty_map.get((scene_id, room_id), 0)
                    rows.append([scene_id, room_id, pov_type, path, is_empty])

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_id", "room_id", "type", "pov_path", "is_empty"])
        writer.writerows(rows)


def collect_data(root, out_csv):
    rows = []

    for scene_id in os.listdir(root):
        scene_dir = os.path.join(root, scene_id)
        if not os.path.isdir(scene_dir):
            continue

        # --- scene-level files ---
        for f in os.listdir(scene_dir):
            path = os.path.join(scene_dir, f)
            if not os.path.isfile(path):
                continue

            if f.endswith("_scene_info.json"):
                cat = "scene_info"
            elif f.endswith("_sem_pointcloud.parquet"):
                cat = "scene_parquet"
            elif f.endswith("_scene_layout.png"):
                cat = "scene_layout"
            else:
                cat = "other"
            rows.append([scene_id, "", cat, path])

        # --- rooms ---
        rooms_dir = os.path.join(scene_dir, "rooms")
        if not os.path.isdir(rooms_dir):
            continue

        for room_id in os.listdir(rooms_dir):
            room_dir = os.path.join(rooms_dir, room_id)
            if not os.path.isdir(room_dir):
                continue

            for f in os.listdir(room_dir):
                path = os.path.join(room_dir, f)
                if not os.path.isfile(path):
                    continue
                if f.endswith(".parquet"):
                    cat = "room_parquet"
                elif f.endswith("_meta.json"):
                    cat = "room_meta"
                else:
                    cat = "other"
                rows.append([scene_id, room_id, cat, path])

            # --- room layouts ---
            layouts_dir = os.path.join(room_dir, "layouts")
            if os.path.isdir(layouts_dir):
                for f in os.listdir(layouts_dir):
                    path = os.path.join(layouts_dir, f)
                    if f.endswith("_room_seg_layout.png"):
                        cat = "room_layout_seg"
                    else:
                        cat = "other"
                    rows.append([scene_id, room_id, cat, path])

            # --- povs ---
            povs_dir = os.path.join(room_dir, "povs")
            if os.path.isdir(povs_dir):
                for f in os.listdir(povs_dir):
                    path = os.path.join(povs_dir, f)
                    if f.endswith("_pov_meta.json"):
                        cat = "pov_meta"
                    elif f.endswith("_minimap.png"):
                        cat = "pov_minimap"
                    else:
                        cat = "other"
                    if os.path.isfile(path):
                        rows.append([scene_id, room_id, cat, path])

                for pov_type in ("seg", "tex"):
                    tdir = os.path.join(povs_dir, pov_type)
                    if not os.path.isdir(tdir):
                        continue
                    for f in os.listdir(tdir):
                        path = os.path.join(tdir, f)
                        if not f.endswith(".png"):
                            cat = "other"
                        else:
                            cat = "pov_seg" if pov_type == "seg" else "pov_tex"
                        rows.append([scene_id, room_id, cat, path])

    # --- write CSV ---
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_id", "room_id", "category", "file_path"])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Collect POVs and full dataset manifest.")
    parser.add_argument("--root", required=True, help="Dataset root directory")
    parser.add_argument("--out", required=True, help="Output directory for CSVs")
    parser.add_argument("--layouts", required=True, help="Path to layouts.csv (for is_empty info)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    empty_map = load_empty_map(args.layouts)

    povs_csv = os.path.join(args.out, "povs.csv")
    data_csv = os.path.join(args.out, "data.csv")

    collect_povs(args.root, povs_csv, empty_map)
    collect_data(args.root, data_csv)


if __name__ == "__main__":
    main()
