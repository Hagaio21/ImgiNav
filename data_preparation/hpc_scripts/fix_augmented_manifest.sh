#!/bin/bash
#BSUB -J fix_manifest
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/fix_manifest.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/fix_manifest.%J.err
#BSUB -n 1
#BSUB -R "rusage[mem=2000]"
#BSUB -W 00:15
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

MANIFEST="/work3/s233249/ImgiNav/datasets/augmented/manifest_images.csv"
BASE_DIR="/work3/s233249/ImgiNav/datasets/augmented"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || conda activate scenefactor || exit 1
fi

python3 << EOF
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("${MANIFEST}", low_memory=False)
base = Path("${BASE_DIR}")

def fix_path(p):
    if pd.isna(p):
        return p
    p = str(p).strip()
    # If already absolute, resolve and fix double augmented
    if Path(p).is_absolute():
        path_obj = Path(p)
        # Replace /augmented/augmented/ with /augmented/ in the path
        parts = list(path_obj.parts)
        filtered = []
        prev = None
        for part in parts:
            if part != "augmented" or prev != "augmented":
                filtered.append(part)
            prev = part
        return str(Path(*filtered))
    # If relative, remove all leading "augmented/" and join with base
    p = p.lstrip('/')
    while p.startswith('augmented/'):
        p = p[10:]
    return str((base / p).resolve())

df['layout_path'] = df['layout_path'].apply(fix_path)
df.to_csv("${MANIFEST}", index=False)
print(f"Fixed {len(df)} rows")
EOF

echo "Done"