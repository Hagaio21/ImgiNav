#!/usr/bin/env python3
"""
Merge POV info from shard files after parallel processing.

Usage:
    python merge_pov_info_shards.py --dataset-root /path/to/dataset
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def merge_pov_info_shards(dataset_root: Path) -> int:
    """Merge all shard_*.json files from pov_info/ directory into pov_info.json."""
    # Check new location first (pov_info/)
    pov_info_dir = dataset_root / "pov_info"
    
    if not pov_info_dir.exists():
        # Fallback to old location (povs/)
        povs_dir = dataset_root / "povs"
        if povs_dir.exists():
            logger.info(f"Using legacy location: {povs_dir}")
            pov_info_dir = povs_dir
        else:
            logger.error(f"POV info directory not found: {pov_info_dir} or {povs_dir}")
            return 1
    
    logger.info(f"Looking for shard files in: {pov_info_dir}")
    
    # Find all shard files (new format: shard_XXXX.json)
    shard_files = sorted(pov_info_dir.glob("shard_*.json"))
    
    if not shard_files:
        # Try old pattern (pov_info_shard_*.json)
        logger.info("No shard_*.json files found, trying old pattern...")
        shard_files = sorted(pov_info_dir.glob("pov_info_shard_*.json"))
    
    if not shard_files:
        logger.warning("No shard files found. Looking for existing pov_info.json...")
        # Check both locations
        for check_dir in [pov_info_dir, dataset_root / "povs"]:
            if (check_dir / "pov_info.json").exists():
                logger.info(f"Found existing pov_info.json at: {check_dir / 'pov_info.json'}")
                return 0
        logger.error("No POV info files found")
        return 1
    
    logger.info(f"Found {len(shard_files)} shard files in {pov_info_dir}")
    
    # Merge all shards
    all_pov_info: List[dict] = []
    
    for shard_file in shard_files:
        try:
            with open(shard_file, "r") as f:
                shard_data = json.load(f)
                all_pov_info.extend(shard_data)
                logger.debug(f"  {shard_file.name}: {len(shard_data)} POVs")
        except Exception as e:
            logger.warning(f"Failed to read {shard_file}: {e}")
    
    logger.info(f"Total POVs: {len(all_pov_info)}")
    
    # Write merged file to pov_info directory
    output_path = pov_info_dir / "pov_info.json"
    with open(output_path, "w") as f:
        json.dump(all_pov_info, f, indent=2)
    
    logger.info(f"Merged POV info saved to: {output_path}")
    
    # Optionally clean up shard files
    # for shard_file in shard_files:
    #     shard_file.unlink()
    # logger.info("Cleaned up shard files")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description="Merge POV info shard files")
    parser.add_argument("--dataset-root", required=True, help="Dataset root directory")
    parser.add_argument("--clean", action="store_true", help="Delete shard files after merge")
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    exit_code = merge_pov_info_shards(dataset_root)
    
    if exit_code == 0 and args.clean:
        pov_info_dir = dataset_root / "pov_info"
        if not pov_info_dir.exists():
            pov_info_dir = dataset_root / "povs"
        
        # Clean both old and new patterns
        for pattern in ["shard_*.json", "pov_info_shard_*.json"]:
            for shard_file in pov_info_dir.glob(pattern):
                shard_file.unlink()
                logger.info(f"Deleted {shard_file.name}")
    
    return exit_code


if __name__ == "__main__":
    exit(main())
