#!/usr/bin/env bash
set -euo pipefail

# ==================== Editable configuration ====================
# Previous experiment source: /home/yohan/project_research/emp_fbnn_new/src_vit_new_new/baseline_evaluation_results/full_8e9618465cd9/per_checkpoint/6cb8d1a618a204eb.json
PYTHON_BIN="${PYTHON_BIN:-/home/yohan/miniconda3/envs/myenv_fbnn/bin/python}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"
PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
RUN_VERSION="${RUN_VERSION:-$(date +%s)}"
TRAIN_SCRIPT="train_vit_base.py"
CONFIGS="./configs/exp_vit_dtd_map_previous"  # Without .yaml
MODELNAME="vitbp16r224"
RUN_NAME="vit_dtd_map"

NEXP=1
SEED=(1111)
INFER=(map)
NSTEP=(2000)
NBATCH=(32)
NFSAMPLE=(1)
SCALELOGSTD=(-8)
NCTSET=(1)
REGLAMBDA=(1e-05)
NFBATCH=(128)
# Evaluate each completed run before starting the next configuration.
EVAL_BATCH_SIZE=2048
EVAL_NFSAMPLES=10  # MAP automatically uses 1.
EVAL_WORKERS=2
EVAL_DATA_ROOT=/home/yohan/Datasets
EVAL_OUTPUT=results/checkpoint_evaluations
# Arrays run every combination sequentially; use one value to keep a setting fixed.
# Additional training settings come from the YAML selected above.
# ==================== Run experiments ====================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONDONTWRITEBYTECODE
dry_run=false
if [[ "${1:-}" == --dry-run ]]; then
    dry_run=true
    shift
fi
if (( $# )); then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 2
fi

SWEEP_PARAMETERS=(SEED INFER NSTEP NBATCH NFSAMPLE SCALELOGSTD NCTSET REGLAMBDA NFBATCH)
declare -A selected
run_index=0

run_experiment() {
    local experiment_version=$((RUN_VERSION + run_index))
    local RUN_DIR="results/${RUN_NAME}_reg${selected[REGLAMBDA]}_seed${selected[SEED]}_${experiment_version}"
    run_index=$((run_index + 1))
    local -a command eval_command
    command=(
        "$PYTHON_BIN" -u "$TRAIN_SCRIPT"
        --checkpoint-manifest "$RUN_DIR/checkpoints.json"
        --configs "$CONFIGS"
        --modelname "$MODELNAME" --infer "${selected[INFER]}"
        --nexp "$NEXP" --seeds "${selected[SEED]}" --v "$experiment_version"
        --nstep "${selected[NSTEP]}" --nbatch "${selected[NBATCH]}" --nfsample "${selected[NFSAMPLE]}"
        --scalelogstd "${selected[SCALELOGSTD]}" --nctset "${selected[NCTSET]}" --reglambda "${selected[REGLAMBDA]}"
        --nfbatch "${selected[NFBATCH]}"
    )
    eval_command=(
        "$PYTHON_BIN" -u evaluate_checkpoints.py
        --checkpoint-manifest "$RUN_DIR/checkpoints.json"
        --batch-size "$EVAL_BATCH_SIZE" --nfsamples "$EVAL_NFSAMPLES"
        --workers "$EVAL_WORKERS" --data-root "$EVAL_DATA_ROOT" --output "$EVAL_OUTPUT"
    )
    if [[ "$dry_run" == true ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
        printf '%q ' "${eval_command[@]}"
        printf '\n'
        printf '%q ' "$PYTHON_BIN" merge_results.py
        printf '\n'
        return
    fi
    mkdir -p "$RUN_DIR"
    cp "${CONFIGS}.yaml" "$RUN_DIR/config.yaml"
    printf '%q ' "${command[@]}" > "$RUN_DIR/command.txt"
    printf '\n' >> "$RUN_DIR/command.txt"
    echo "Experiment log: $RUN_DIR/train.log"
    "${command[@]}" 2>&1 | tee "$RUN_DIR/train.log"
    printf '%q ' "${eval_command[@]}" > "$RUN_DIR/evaluation_command.txt"
    printf '\n' >> "$RUN_DIR/evaluation_command.txt"
    echo "Evaluating best checkpoint(s): $RUN_DIR/evaluation.log"
    "${eval_command[@]}" 2>&1 | tee "$RUN_DIR/evaluation.log"
    "$PYTHON_BIN" merge_results.py
    echo "Training and evaluation finished. See $RUN_DIR and results/merged/"
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
