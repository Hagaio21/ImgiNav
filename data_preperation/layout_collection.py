#!/usr/bin/env python3
"""
layout_collection.py (DEPRECATED)
----------------------------------

DEPRECATED: This script has been consolidated into collect.py.
Use: python collect.py --type layouts --root <path> --output <layouts.csv>

This wrapper is kept for backward compatibility with HPC scripts.
"""

import argparse
import sys
import multiprocessing
from pathlib import Path

# Redirect to unified collect.py
sys.path.insert(0, str(Path(__file__).parent))
from collect import collect_layouts


def main():
    ap = argparse.ArgumentParser(
        description="Collect layout manifest (DEPRECATED: use collect.py --type layouts)"
    )
    ap.add_argument("--root", required=True, help="Root folder containing layout PNGs")
    ap.add_argument("--out", default="layouts.csv", help="Output CSV path")
    ap.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                    help="Number of parallel workers (default: all cores)")
    args = ap.parse_args()

    print("[WARNING] layout_collection.py is deprecated. Use: collect.py --type layouts")
    collect_layouts(Path(args.root), Path(args.out), args.workers)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
