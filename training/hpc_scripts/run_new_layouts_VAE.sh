#!/bin/bash
#BSUB -J new_layouts_VAE
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/new_layouts_VAE.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/new_layouts_VAE.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 24:00
#BSUB -q gpuv100

module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
export MKL_INTERFACE_LAYER=LP64
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate imginav || conda activate scenefactor

cd /work3/s233249/ImgiNav/ImgiNav
python training/train.py experiments/autoencoders/new_layouts/new_layouts_VAE_64x64_structural_256.yaml

