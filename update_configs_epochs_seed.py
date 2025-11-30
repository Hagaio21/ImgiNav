#!/usr/bin/env python3
"""Update all experiment config files: change epochs to 300 and add sampling_seed_base"""

import re
from pathlib import Path

def update_config_file(file_path):
    """Update a single config file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = content = f.read()
        
        # Change epochs: 1000 to epochs: 300
        content = re.sub(r'(\s+epochs:\s+)1000', r'\1300', content)
        # Also handle epochs_target if present
        content = re.sub(r'(\s+epochs_target:\s+)1000', r'\1300', content)
        
        # Add sampling_seed_base if not already present
        if 'sampling_seed_base' not in content:
            # Look for num_conditioned_samples_per_type and add sampling_seed_base before it
            pattern1 = r'(  num_conditioned_samples_per_type: \d+\n)'
            replacement1 = r'  sampling_seed_base: 42\n\1'
            content = re.sub(pattern1, replacement1, content)
            
            # If num_conditioned_samples_per_type not found, try adding before loss:
            if 'sampling_seed_base' not in content:
                pattern2 = r'(  loss:)'
                replacement2 = r'  sampling_seed_base: 42\n\1'
                content = re.sub(pattern2, replacement2, content)
        
        # Only write if something changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

# Find all YAML files in experiments/diffusion
base_dir = Path("experiments/diffusion")
yaml_files = list(base_dir.rglob("*.yaml"))
# Exclude temp files and README
yaml_files = [f for f in yaml_files if ".tmp" not in f.name and "README" not in f.name]

updated_count = 0
epochs_changed = 0
seed_added = 0

for yaml_file in yaml_files:
    original_content = ""
    with open(yaml_file, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    if update_config_file(yaml_file):
        updated_count += 1
        
        # Check what changed
        with open(yaml_file, 'r', encoding='utf-8') as f:
            new_content = f.read()
        
        if 'epochs: 1000' in original_content and 'epochs: 300' in new_content:
            epochs_changed += 1
        if 'sampling_seed_base' not in original_content and 'sampling_seed_base' in new_content:
            seed_added += 1
        
        print(f"Updated: {yaml_file}")

print(f"\nSummary:")
print(f"  Total files updated: {updated_count}")
print(f"  Epochs changed (1000 -> 300): {epochs_changed}")
print(f"  sampling_seed_base added: {seed_added}")

