#!/usr/bin/env bash
set -euo pipefail

# ==================== Editable configuration ====================
# Previous experiment source: /home/yohan/project_research/emp_fbnn_new/src_resnet_new/results/statistics/runs_cifar10/bmafbnn_st_parallel_20261027_011322/v2000_bmafbresnet18st_seed1111_logstd-9_reg1e-4_mix0.1_prior1.ood.json
PYTHON_BIN="${PYTHON_BIN:-/home/yohan/miniconda3/envs/myenv_fbnn/bin/python}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"
PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
RUN_VERSION="${RUN_VERSION:-$(date +%s)}"
TRAIN_SCRIPT="train_bmafbnn_st.py"
CONFIGS="./configs/exp_cifar10_resnet18st_previous"  # Without .yaml
MODELNAME="bmafbresnet18st"
RUN_NAME="resnet18st"

NEXP=1
SEED=1111
NTRAINSET=45000
NEPOCH=200
SCALELOGSTD=-9
REGLAMBDA=0.0001
PRIORSTDOUT=1.0
NEPOCHRATIO=0.75
MIXALPHA=0.1
NCTSET=4
NFSAMPLE=10

OODSETS=(svhn2 lsun32 place365)
EVAL_NFSAMPLES=(1 10 20)
TABLE_SEEDS=("$SEED")
TABLE_NFSAMPLES=(10)
PLACES365_ROOT="${PLACES365_ROOT:-/home/yohan/Datasets/places365/val_256}"
# Additional training settings come from the YAML selected above.
# ==================== Run experiment ====================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONDONTWRITEBYTECODE
export PLACES365_ROOT
RUN_DIR="results/${RUN_NAME}_seed${SEED}_${RUN_VERSION}"

command=(
    "$PYTHON_BIN" -u "$TRAIN_SCRIPT"
    --configs "$CONFIGS"
    --modelname "$MODELNAME"
    --nexp "$NEXP" --seeds "$SEED" --v "$RUN_VERSION"
    --ntrainset "$NTRAINSET" --nepoch "$NEPOCH"
    --scalelogstd "$SCALELOGSTD" --reglambda "$REGLAMBDA" --priorstdout "$PRIORSTDOUT"
    --nepochratio "$NEPOCHRATIO" --mixalpha "$MIXALPHA"
    --nctset "$NCTSET" --nfsample "$NFSAMPLE"
    --oodsets "${OODSETS[@]}"
    --eval-nfsamples "${EVAL_NFSAMPLES[@]}"
    --table-seeds "${TABLE_SEEDS[@]}" --table-nfsamples "${TABLE_NFSAMPLES[@]}"
    --places365-root "$PLACES365_ROOT"
    --result-json "$RUN_DIR/results.json"
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
printf '%q ' "${command[@]}" > "$RUN_DIR/command.txt"
printf '\n' >> "$RUN_DIR/command.txt"
cp "${CONFIGS}.yaml" "$RUN_DIR/config.yaml"
echo "Training and evaluation log: $RUN_DIR/train.log"
"${command[@]}" 2>&1 | tee "$RUN_DIR/train.log"
echo "Finished. Evaluation results: $RUN_DIR/results.json"
