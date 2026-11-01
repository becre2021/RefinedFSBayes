#!/usr/bin/env bash
set -euo pipefail

# ==================== Editable configuration ====================
# Previous experiment source: /home/yohan/project_research/emp_fbnn_new/src_vit_new_new/ablation_runs/trajectory_20261010_151855_1276216/results_table_dtd.csv
PYTHON_BIN="${PYTHON_BIN:-/home/yohan/miniconda3/envs/myenv_fbnn/bin/python}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"
PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
RUN_VERSION="${RUN_VERSION:-$(date +%s)}"
TRAIN_SCRIPT="train_vit_bmafbnn.py"
CONFIGS="./configs/exp_vit_dtd_proposed_previous"  # Without .yaml
MODELNAME="vitbp16r224"
RUN_NAME="vit_dtd_proposed"

NEXP=1
SEED=1111
INFER=vbp
NSTEP=2000
NBATCH=32
NFSAMPLE=10
SCALELOGSTD=-4
NCTSET=4
REGLAMBDA=1e-05
LR=0.01
PRIORSTDOUT=10.0
NEPOCHRATIO=0.66
MIXALPHA=1.0
LOSSTYPE=elbo
PARAMTRACE=0
# Additional training settings come from the YAML selected above.
# ==================== Run experiment ====================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONDONTWRITEBYTECODE
RUN_DIR="results/${RUN_NAME}_seed${SEED}_${RUN_VERSION}"

command=(
    "$PYTHON_BIN" -u "$TRAIN_SCRIPT"
    --configs "$CONFIGS"
    --modelname "$MODELNAME" --infer "$INFER"
    --nexp "$NEXP" --seeds "$SEED" --v "$RUN_VERSION"
    --nstep "$NSTEP" --nbatch "$NBATCH" --nfsample "$NFSAMPLE"
    --scalelogstd "$SCALELOGSTD" --nctset "$NCTSET" --reglambda "$REGLAMBDA"
    --lr "$LR" --priorstdout "$PRIORSTDOUT" --nepochratio "$NEPOCHRATIO" --mixalpha "$MIXALPHA" --losstype "$LOSSTYPE" --paramtrace "$PARAMTRACE"
)
if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
if (( $# )); then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 2
fi
mkdir -p "$RUN_DIR"
cp "${CONFIGS}.yaml" "$RUN_DIR/config.yaml"
printf '%q ' "${command[@]}" > "$RUN_DIR/command.txt"
printf '\n' >> "$RUN_DIR/command.txt"
echo "Training and validation log: $RUN_DIR/train.log"
"${command[@]}" 2>&1 | tee "$RUN_DIR/train.log"
echo "Finished. Checkpoint location is recorded in $RUN_DIR/train.log"
