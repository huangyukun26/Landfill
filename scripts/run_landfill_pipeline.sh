#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <project_root> [run_name]" >&2
  exit 1
fi

PROJECT_ROOT="$(cd "$1" && pwd)"
RUN_NAME="${2:-vit_b_20ep}"

CONDA_ENV="${CONDA_ENV:-landfill}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CODE_DIR="$PROJECT_ROOT/code"
OUTPUT_ROOT="$PROJECT_ROOT/outputs/$RUN_NAME"
EVAL_ROOT="${OUTPUT_ROOT}_eval"
CKPT_PATH="${CKPT_PATH:-$PROJECT_ROOT/checkpoints/sam_vit_b_01ec64.pth}"
LIST_DIR="${LIST_DIR:-$PROJECT_ROOT/ImageSets}"
LORA_CKPT="${LORA_CKPT:-}"
VIT_NAME="${VIT_NAME:-vit_b}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-3}"
N_GPU="${N_GPU:-1}"

mkdir -p "$OUTPUT_ROOT" "$EVAL_ROOT"

run_python() {
  conda run -n "$CONDA_ENV" "$PYTHON_BIN" "$@"
}

echo "[pipeline] project_root=$PROJECT_ROOT"
echo "[pipeline] run_name=$RUN_NAME"
echo "[pipeline] max_epochs=$MAX_EPOCHS batch_size=$BATCH_SIZE"

TRAIN_CMD=(
  run_python "$CODE_DIR/train.py"
  --root_path "$PROJECT_ROOT"
  --list_dir "$LIST_DIR"
  --output "$OUTPUT_ROOT"
  --ckpt "$CKPT_PATH"
  --vit_name "$VIT_NAME"
  --max_epochs "$MAX_EPOCHS"
  --batch_size "$BATCH_SIZE"
  --n_gpu "$N_GPU"
)

if [[ -n "$LORA_CKPT" ]]; then
  TRAIN_CMD+=(--lora_ckpt "$LORA_CKPT")
fi

"${TRAIN_CMD[@]}"

LATEST_RUN="$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
BEST_CKPT="$LATEST_RUN/checkpoint_best.pth"

if [[ ! -f "$BEST_CKPT" ]]; then
  echo "best checkpoint not found: $BEST_CKPT" >&2
  exit 2
fi

echo "[pipeline] best_ckpt=$BEST_CKPT"

run_python "$CODE_DIR/test_landfill.py" \
  --volume_path "$PROJECT_ROOT" \
  --list_dir "$LIST_DIR" \
  --output_dir "$EVAL_ROOT" \
  --ckpt "$CKPT_PATH" \
  --lora_ckpt "$BEST_CKPT" \
  --vit_name "$VIT_NAME"

run_python "$PROJECT_ROOT/scripts/calc_seg_metrics.py" \
  --pred-dir "$EVAL_ROOT/predictions" \
  --output "$EVAL_ROOT/metrics.txt"

echo "[pipeline] finished"
echo "[pipeline] eval_dir=$EVAL_ROOT"
