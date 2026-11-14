#!/usr/bin/env bash
set -euo pipefail

# ==================== Editable configuration ====================
# Selected run: version=1793707937; results/resnet50st_reg1e-4_seed1111_1793707937/results.json
PYTHON_BIN="${PYTHON_BIN:-/home/yohan/miniconda3/envs/myenv_fbnn/bin/python}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"
PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
RUN_VERSION="${RUN_VERSION:-$(date +%s)}"
TRAIN_SCRIPT="train_bmafbnn_st.py"
CONFIGS="./configs/exp_cifar100_resnet50st"  # Without .yaml
MODELNAME="bmafbresnet50st"
RUN_NAME="resnet50st"

NEXP=1
SEED=(1111)
NTRAINSET=(45000)
NEPOCH=(200)
SCALELOGSTD=(-10)
REGLAMBDA=(1e-4)
PRIORSTDOUT=(10)
NEPOCHRATIO=(0.75)
MIXALPHA=(0.1)
NCTSET=(4)
NFSAMPLE=(10)

OODSETS=(svhn2 lsun32 place365)
EVAL_NFSAMPLES=(1 10 20)
TABLE_NFSAMPLES=(10)
PLACES365_ROOT="${PLACES365_ROOT:-/home/yohan/Datasets/places365/val_256}"
# Arrays run every combination sequentially; use one value to keep a setting fixed.
# Additional training settings come from the YAML selected above.
# ==================== Run experiments ====================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONDONTWRITEBYTECODE
export PLACES365_ROOT
dry_run=false
if [[ "${1:-}" == --dry-run ]]; then
    dry_run=true
    shift
fi
if (( $# )); then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 2
fi

SWEEP_PARAMETERS=(SEED NTRAINSET NEPOCH SCALELOGSTD REGLAMBDA PRIORSTDOUT NEPOCHRATIO MIXALPHA NCTSET NFSAMPLE)
declare -A selected
run_index=0

run_experiment() {
    local experiment_version=$((RUN_VERSION + run_index))
    local RUN_DIR="results/${RUN_NAME}_reg${selected[REGLAMBDA]}_seed${selected[SEED]}_${experiment_version}"
    run_index=$((run_index + 1))
    local -a command
    command=(
        "$PYTHON_BIN" -u "$TRAIN_SCRIPT"
        --configs "$CONFIGS"
        --modelname "$MODELNAME"
        --nexp "$NEXP" --seeds "${selected[SEED]}" --v "$experiment_version"
        --ntrainset "${selected[NTRAINSET]}" --nepoch "${selected[NEPOCH]}"
        --scalelogstd "${selected[SCALELOGSTD]}" --reglambda "${selected[REGLAMBDA]}" --priorstdout "${selected[PRIORSTDOUT]}"
        --nepochratio "${selected[NEPOCHRATIO]}" --mixalpha "${selected[MIXALPHA]}"
        --nctset "${selected[NCTSET]}" --nfsample "${selected[NFSAMPLE]}"
        --oodsets "${OODSETS[@]}"
        --eval-nfsamples "${EVAL_NFSAMPLES[@]}"
        --table-seeds "${selected[SEED]}" --table-nfsamples "${TABLE_NFSAMPLES[@]}"
        --places365-root "$PLACES365_ROOT"
        --result-json "$RUN_DIR/results.json"
    )
    if [[ "$dry_run" == true ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
        return
    fi
    mkdir -p "$RUN_DIR"
    cp "${CONFIGS}.yaml" "$RUN_DIR/config.yaml"
    printf '%q ' "${command[@]}" > "$RUN_DIR/command.txt"
    printf '\n' >> "$RUN_DIR/command.txt"
    echo "Experiment log: $RUN_DIR/train.log"
    "${command[@]}" 2>&1 | tee "$RUN_DIR/train.log"
    echo "Finished. See $RUN_DIR/train.log"
}

# Recursively enumerate the Cartesian product without launching parallel jobs.
run_combinations() {
    local index=$1
    if (( index == ${#SWEEP_PARAMETERS[@]} )); then
        run_experiment
        return
    fi
    local key="${SWEEP_PARAMETERS[index]}" value
    local -n values="$key"
    if (( ${#values[@]} == 0 )); then
        echo "$key must contain at least one value" >&2
        return 2
    fi
    for value in "${values[@]}"; do
        selected[$key]="$value"
        run_combinations "$((index + 1))"
    done
}
run_combinations 0
