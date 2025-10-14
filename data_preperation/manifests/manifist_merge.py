import argparse
import pandas as pd

def load_and_clean_csv(path):
    df = pd.read_csv(path)
    df = df.fillna("")
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    return df

def is_valid_path(x):
    invalid = {"", "false", "0", "none"}
    return str(x).strip().lower() not in invalid

def build_room_dataset(graphs_path, layouts_path, pov_path, out_path):
    graphs = load_and_clean_csv(graphs_path)
    layouts = load_and_clean_csv(layouts_path)
    pov = load_and_clean_csv(pov_path)

    graphs_room = graphs[graphs["type"] == "room"]
    layouts_room = layouts[layouts["type"] == "room"]

    merged = pov.merge(graphs_room, on=["scene_id", "room_id"], how="left", suffixes=("", "_graph"))
    merged = merged.merge(layouts_room, on=["scene_id", "room_id"], how="left", suffixes=("", "_layout"))

    df = pd.DataFrame({
        "number": range(len(merged)),
        "SCENE_ID": merged["scene_id"],
        "POV_TYPE": merged["type"],
        "POV_PATH": merged["pov_path"],
        "POV_EMBEDDING_PATH": merged["embedding_path"],
        "ROOM_ID": merged["room_id"],
        "ROOM_GRAPH_PATH": merged["graph_path"],
        "ROOM_GRAPH_EMBEDDING_PATH": merged["embedding_path_graph"],
        "ROOM_LAYOUT_PATH": merged["layout_path"],
        "ROOM_LAYOUT_EMBEDDING_PATH": merged["embedding_path_layout"]
    })

    mask = (
        df["POV_PATH"].apply(is_valid_path)
        & df["POV_EMBEDDING_PATH"].apply(is_valid_path)
        & df["ROOM_GRAPH_PATH"].apply(is_valid_path)
        & df["ROOM_GRAPH_EMBEDDING_PATH"].apply(is_valid_path)
        & df["ROOM_LAYOUT_PATH"].apply(is_valid_path)
        & df["ROOM_LAYOUT_EMBEDDING_PATH"].apply(is_valid_path)
    )
    df = df[mask].reset_index(drop=True)
    df["number"] = range(len(df))
    df.to_csv(out_path, index=False)
    print(f"Room dataset saved: {out_path} ({len(df)} rows)")

def build_scene_dataset(graphs_path, layouts_path, out_path):
    graphs = load_and_clean_csv(graphs_path)
    layouts = load_and_clean_csv(layouts_path)

    graphs_scene = graphs[graphs["type"] == "scene"]
    layouts_scene = layouts[layouts["type"] == "scene"]

    merged = graphs_scene.merge(layouts_scene, on="scene_id", how="left", suffixes=("", "_layout"))

    df = pd.DataFrame({
    "number": range(len(merged)),
    "SCENE_ID": merged["scene_id"],
    "SCENE_GRAPH_PATH": merged["graph_path"],
    "SCENE_GRAPH_EMBEDDING_PATH": merged["embedding_path"],
    "SCENE_LAYOUT_PATH": merged["layout_path_layout"],
    "SCENE_LAYOUT_EMBEDDING_PATH": merged["embedding_path_layout"]
    })


    mask = (
        df["SCENE_GRAPH_PATH"].apply(is_valid_path)
        & df["SCENE_GRAPH_EMBEDDING_PATH"].apply(is_valid_path)
        & df["SCENE_LAYOUT_PATH"].apply(is_valid_path)
        & df["SCENE_LAYOUT_EMBEDDING_PATH"].apply(is_valid_path)
    )
    df = df[mask].reset_index(drop=True)
    df["number"] = range(len(df))
    df.to_csv(out_path, index=False)
    print(f"Scene dataset saved: {out_path} ({len(df)} rows)")

def main():
    parser = argparse.ArgumentParser(description="Unify graphs, layouts, and POV manifests into room and scene datasets.")
    parser.add_argument("--graphs", required=True, help="Path to graphs.csv")
    parser.add_argument("--layouts", required=True, help="Path to layouts.csv")
    parser.add_argument("--pov", required=True, help="Path to pov.csv")
    parser.add_argument("--room_out", required=True, help="Output CSV for room_dataset")
    parser.add_argument("--scene_out", required=True, help="Output CSV for scene_dataset")
    args = parser.parse_args()

    build_room_dataset(args.graphs, args.layouts, args.pov, args.room_out)
    build_scene_dataset(args.graphs, args.layouts, args.scene_out)

if __name__ == "__main__":
    main()
