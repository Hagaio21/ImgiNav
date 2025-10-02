#!/bin/bash
#BSUB -J layout_embeddings
#BSUB -o /zhome/62/5/203350/ws/ImgiNav/data_preperation/hpc_scripts/logs/layout_embeddings.%J.out
#BSUB -e /zhome/62/5/203350/ws/ImgiNav/data_preperation/hpc_scripts/logs/layout_embeddings.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=8000, ngpus_excl_p=1]"   # request 1 GPU + 8GB RAM
#BSUB -W 08:00
#BSUB -q gpuv100   # or gpu queue available on your cluster

set -euo pipefail

# Paths
CONFIG_PATH="/work3/s233249/ImgiNav/models/autoencoders/ae_config.yaml"
CKPT_PATH="/work3/s233249/ImgiNav/models/autoencoders/ae_checkpoint.pth"
IN_MANIFEST="/work3/s233249/ImgiNav/datasets/layouts.csv"
OUT_MANIFEST="/work3/s233249/ImgiNav/datasets/layouts_with_embeddings.csv"
SCRIPT_PATH="/zhome/62/5/203350/ws/ImgiNav/data_preperation/layout_embeddings.py"

# Conda activation (if you use it)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate scenefactor || true
fi

echo "Running layout embedding creation"
python "$SCRIPT_PATH" \
  --config "$CONFIG_PATH" \
  --ckpt "$CKPT_PATH" \
  --manifest "$IN_MANIFEST" \
  --out_manifest "$OUT_MANIFEST" \
  --batch_size 128 \
  --num_workers 4 \
  --device cuda \
  --format pt

echo "Finished. Updated manifest: $OUT_MANIFEST"
