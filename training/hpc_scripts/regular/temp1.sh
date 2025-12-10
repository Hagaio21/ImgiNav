#!/bin/bash
# Submit only the gpua100 experiments, but into gpuv100 queue

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/train_diffusion.py"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

mkdir -p "${LOG_DIR}"

echo "Submitting gpua100 experiments to gpuv100 queue"

GPUA100_CONFIGS=(
    "experiments/diffusion/v2/seg/rooms_both_large_wide_shallow_v2.yaml"
    "experiments/diffusion/v2/seg/rooms_both_medium_wide_shallow_v2.yaml"
)

for config in "${GPUA100_CONFIGS[@]}"; do
    config_path="${BASE_DIR}/${config}"

    if [ ! -f "${config_path}" ]; then
        echo "Missing config: ${config_path}"
        continue
    fi

    exp_name=$(python3 - <<EOF
import yaml, re
try:
    c=yaml.safe_load(open("${config_path}"))
    n=c.get("experiment",{}).get("name","unnamed")
    n=re.sub(r"[^a-zA-Z0-9_]","_",n)
    n=re.sub(r"_+","_",n).strip("_")
    print(n[:50])
except:
    print("unnamed")
EOF
)

    log_suffix=$(echo "${config}" | sed 's|experiments/diffusion/v2/||; s|/|_|g; s|.yaml||')

    echo "Submitting ${exp_name} to gpuv100"

    bsub -J "${exp_name}" \
         -o "${LOG_DIR}/train_diff_${log_suffix}.%J.out" \
         -e "${LOG_DIR}/train_diff_${log_suffix}.%J.err" \
         -n 4 \
         -R "rusage[mem=20000]" \
         -gpu "num=1" \
         -W 24:00 \
         -q gpuv100 \
         bash -c "cd ${BASE_DIR} &&
                  module load cuda/11.8 &&
                  module load cudnn/v8.6.0.163-prod-cuda-11.X &&
                  source \$HOME/miniconda3/etc/profile.d/conda.sh &&
                  conda activate imginav &&
                  python ${PYTHON_SCRIPT} ${config_path} --resume"

    sleep 2
done

echo "Done submitting gpua100 experiments into gpuv100."
