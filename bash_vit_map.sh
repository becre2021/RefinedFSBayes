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
SEED=1111
INFER=map
NSTEP=2000
NBATCH=32
NFSAMPLE=1
SCALELOGSTD=-8
NCTSET=1
REGLAMBDA=1e-05
NFBATCH=128
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
    --nfbatch "$NFBATCH"
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
