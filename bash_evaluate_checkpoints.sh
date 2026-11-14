#!/usr/bin/env bash
set -euo pipefail
# Editable evaluation settings. These commands do not train models.
PYTHON_BIN="${PYTHON_BIN:-/home/yohan/miniconda3/envs/myenv_fbnn/bin/python}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"
NFSAMPLES=10  # Bayesian models; MAP always uses 1.
BATCH_SIZE=2048
WORKERS=2
DATA_ROOT=/home/yohan/Datasets
OUTPUT=results/checkpoint_evaluations
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
shopt -s nullglob
# Add saved last-layer ResNet checkpoint paths here when available.
CHECKPOINTS=(results/dtd_tinyimagenet/*/seeds*.pth.tar)
if (( ${#CHECKPOINTS[@]} == 0 )); then
    echo 'No completed DTD checkpoints found' >&2
    exit 1
fi
command=("$PYTHON_BIN" -u evaluate_checkpoints.py "${CHECKPOINTS[@]}"
         --nfsamples "$NFSAMPLES" --batch-size "$BATCH_SIZE" --workers "$WORKERS"
         --data-root "$DATA_ROOT" --output "$OUTPUT")
if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
if (( $# )); then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 2
fi
mkdir -p "$OUTPUT"
"${command[@]}" 2>&1 | tee "$OUTPUT/evaluation.log"
"$PYTHON_BIN" merge_results.py
