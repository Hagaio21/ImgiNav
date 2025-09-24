#!/usr/bin/env python3
"""
visualize_palette.py

Generates one PNG per super-category showing its categories and assigned colors.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def visualize_per_super(taxonomy_path: str, out_dir: str):
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    super2id = taxonomy["super2id"]
    category2id = taxonomy["category2id"]
    category2super = taxonomy["category2super"]
    id2color = taxonomy["id2color"]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for supercat, sid in super2id.items():
        # categories belonging to this super
        categories = sorted([c for c, s in category2super.items() if s == supercat])

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis("off")

        # draw super block
        scol = tuple(c / 255 for c in id2color.get(str(sid), [127, 127, 127]))
        ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, facecolor=scol, edgecolor="black"))
        ax.text(0.5, -0.2, f"SUPER: {supercat} (ID {sid})", ha="center", va="top", fontsize=12, fontweight="bold")

        # draw categories in a grid
        cols = 6
        for i, cat in enumerate(categories):
            cid = category2id[cat]
            ccol = tuple(c / 255 for c in id2color.get(str(cid), [127, 127, 127]))
            row, col = divmod(i, cols)
            x, y = col * 1.5, -(row + 2) * 1.2
            ax.add_patch(mpatches.Rectangle((x, y), 1, 1, facecolor=ccol, edgecolor="black"))
            ax.text(x + 0.5, y - 0.2, f"{cat}\nID {cid}", ha="center", va="top", fontsize=8)

        ax.set_xlim(-0.5, cols * 1.5)
        ax.set_ylim(-(len(categories) // cols + 4) * 1.2, 1.5)

        out_path = out_dir / f"palette_{supercat.replace('/', '_')}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize taxonomy color palette (one PNG per super).")
    parser.add_argument("taxonomy", help="Path to taxonomy.json")
    parser.add_argument("--out-dir", required=True, help="Directory to save super-category palettes")
    args = parser.parse_args()

    visualize_per_super(args.taxonomy, args.out_dir)
