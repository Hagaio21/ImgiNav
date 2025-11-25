#!/usr/bin/env python3
"""
Script to remove early stopping settings from all diffusion experiment configs
and revert eval_interval to 10.
"""

import yaml
from pathlib import Path
import sys

def update_config(config_path: Path):
    """Remove early stopping from a single config file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        updated = False
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check if this is an early stopping line
            if 'early_stopping_patience' in line or 'early_stopping_min_delta' in line:
                # Skip this line
                updated = True
                i += 1
                continue
            
            # Check if eval_interval is 5 and revert to 10
            if 'eval_interval: 5' in line:
                new_lines.append(line.replace('eval_interval: 5', 'eval_interval: 10'))
                updated = True
                i += 1
                continue
            
            new_lines.append(line)
            i += 1
        
        if updated:
            with open(config_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            print(f"  Updated {config_path}")
            return True
        else:
            print(f"  No changes needed for {config_path}")
            return False
    except Exception as e:
        print(f"  Error updating {config_path}: {e}")
        return False

def main():
    base_dir = Path('experiments/diffusion/clip')
    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist")
        sys.exit(1)
    
    config_files = list(base_dir.rglob('*.yaml'))
    print(f"Found {len(config_files)} YAML files")
    
    updated = 0
    for config_file in sorted(config_files):
        if update_config(config_file):
            updated += 1
    
    print(f"\nUpdated {updated} config files")

if __name__ == '__main__':
    main()

