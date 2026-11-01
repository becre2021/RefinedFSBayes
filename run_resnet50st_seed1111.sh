#!/usr/bin/env bash
set -euo pipefail

# ==================== Editable configuration ====================
# Previous experiment: src_resnet_new/statistics/runs_cifar100/results_merged_configs.json, C01.
PYTHON_BIN="${PYTHON_BIN:-/home/yohan/miniconda3/envs/myenv_fbnn/bin/python}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"
PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
RUN_VERSION="${RUN_VERSION:-$(date +%s)}"
TRAIN_SCRIPT="train_bmafbnn_st.py"
CONFIGS="./configs/exp_cifar100_resnet50st_best_accuracy"  # Without .yaml
MODELNAME="bmafbresnet50st"
RUN_NAME="resnet50st"

NEXP=1
SEED=1111
NTRAINSET=45000
NEPOCH=200
SCALELOGSTD=-9
REGLAMBDA=1e-3
PRIORSTDOUT=10
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
