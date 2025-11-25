#!/usr/bin/env python3
"""
Script to update all diffusion experiment configs to enable early stopping.
Sets eval_interval to 5 and adds early_stopping_patience: 10.
"""

import yaml
from pathlib import Path
import sys

def update_config(config_path: Path):
    """Update a single config file to add early stopping."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Check if training section exists
        if 'training' not in config:
            print(f"  Skipping {config_path}: No training section")
            return False
        
        training = config['training']
        
        # Update eval_interval if it exists
        if 'eval_interval' in training:
            if training['eval_interval'] == 10:
                training['eval_interval'] = 5
            elif training['eval_interval'] != 5:
                print(f"  Warning: {config_path} has eval_interval={training['eval_interval']}, not updating")
        
        # Add early stopping if not already present
        if 'early_stopping_patience' not in training:
            # Insert after eval_interval
            training['early_stopping_patience'] = 10
            training['early_stopping_min_delta'] = 0.0
        else:
            print(f"  {config_path} already has early_stopping_patience, skipping")
            return False
        
        # Write back
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        print(f"  Updated {config_path}")
        return True
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

