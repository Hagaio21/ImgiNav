#!/bin/bash
#BSUB -J ae_embed
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preperation/hpc_scripts/logs/ae_embed.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preperation/hpc_scripts/logs/ae_embed.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=32000]"
#BSUB -gpu "num=1"
#BSUB -W 10:00
#BSUB -q gpul40s

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# ----------------------------------------------------------------------
# Configuration
SCRIPT_PATH="/work3/s233249/ImgiNav/ImgiNav/data_preperation/stage6_create_layout_embeddings.py"
CONFIG="/work3/s233249/ImgiNav/experiments/ae_configs/config_diff_4ch_64x64_vanilla.yml"
CKPT="/work3/s233249/ImgiNav/experiments/autoencoder_final_64x64x4_vanila/20251004-082051_ae_final/best.pt"
MANIFEST="/work3/s233249/ImgiNav/datasets/layouts.csv"
OUT_MANIFEST="/work3/s233249/ImgiNav/datasets/layout_manifest_with_emb.csv"
BATCH_SIZE=8
NUM_WORKERS=4
DEVICE="cuda"
FORMAT="pt"

echo "Running AutoEncoder embedding job"

# ----------------------------------------------------------------------
# Conda environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || true
fi

# ----------------------------------------------------------------------
# Run embedding
python "${SCRIPT_PATH}" \
  --config "${CONFIG}" \
  --ckpt "${CKPT}" \
  --manifest "${MANIFEST}" \
  --out_manifest "${OUT_MANIFEST}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --format "${FORMAT}"

echo "AutoEncoder embedding completed successfully"
