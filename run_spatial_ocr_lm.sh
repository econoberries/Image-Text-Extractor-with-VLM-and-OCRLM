#!/usr/bin/env bash
# Seventh OCRBench v2 English KIE experiment:
# spatial LaOCR text -> Qwen3.5-9B -> official scoring.
# Run from the repository root or any other directory.
#
# Usage:
#   bash run_spatial_ocr_lm.sh [--submit|--prepare-only|--score-only|--dry-run] BATCH_JSON
#
# The original six model runs and their results are never changed.

set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

usage() {
  cat <<'HELP'
Usage: bash run_spatial_ocr_lm.sh [--submit|--prepare-only|--score-only|--dry-run] BATCH_JSON

  --submit         Prepare spatial text, queue GPU inference, then CPU scoring (default).
  --prepare-only   Reconstruct spatial text from existing LaOCR cache, without a GPU.
  --score-only     Queue scoring for existing spatial predictions (no GPU).
  --dry-run        Check paths and print planned configuration without running jobs.

Example:
  bash run_spatial_ocr_lm.sh --dry-run \
    runs/batches/cbdd58de-531a-4c9c-9778-5bc8b0b0be48/batch.json

Configuration (optional environment variables):
  GPU_PARTITION=gpu-single
  GPU_GRES=gpu:A100:1,gpumem_per_gpu:70GB
  GPU_TIME=06:00:00
  CPU_PARTITION=cpu-single
HELP
}

ACTION=submit
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == --* ]]; then
  case "$1" in
    --submit) ACTION=submit ;;
    --prepare-only) ACTION=prepare-only ;;
    --score-only) ACTION=score-only ;;
    --dry-run) ACTION=dry-run ;;
    *) usage >&2; echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
fi
if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

BATCH="$1"
PYTHON="$ROOT/.venv/bin/python"
SCRIPT="$ROOT/experiments/spatial_ocr_lm.py"
SBATCH_SCRIPT="$ROOT/slurm/spatial_ocr_lm.sbatch"

for file in "$BATCH" "$SCRIPT" "$SBATCH_SCRIPT"; do
  [[ -f "$file" ]] || { echo "Missing file: $file" >&2; exit 1; }
done

BATCH="$(realpath -- "$BATCH")"
BATCH_ID="$(basename -- "$(dirname -- "$BATCH")")"
RUN_DIR="$ROOT/runs/analysis/spatial_ocr_lm_${BATCH_ID}"

GPU_PARTITION="${GPU_PARTITION:-gpu-single}"
GPU_GRES="${GPU_GRES:-gpu:A100:1,gpumem_per_gpu:70GB}"
GPU_TIME="${GPU_TIME:-06:00:00}"
CPU_PARTITION="${CPU_PARTITION:-cpu-single}"

if [[ "$ACTION" == dry-run ]]; then
  echo "Project: $ROOT"
  echo "Batch: $BATCH"
  echo "Output: $RUN_DIR"
  echo "GPU request: partition=$GPU_PARTITION gres=$GPU_GRES time=$GPU_TIME"
  echo "Scoring: partition=$CPU_PARTITION"
  [[ ! -f "$RUN_DIR/predictions_scored.json" ]] || echo "Already scored; --submit would skip."
  echo "Dry run: nothing prepared or submitted."
  exit 0
fi

[[ -x "$PYTHON" ]] || {
  echo "Missing $PYTHON. Set up the existing Qwen/Torch .venv first." >&2
  exit 1
}

if [[ "$ACTION" == prepare-only ]]; then
  exec "$PYTHON" "$SCRIPT" prepare --batch "$BATCH"
fi

if [[ "$ACTION" == submit && -f "$RUN_DIR/predictions_scored.json" ]]; then
  echo "Already scored: $RUN_DIR/predictions_scored.json"
  echo "No new GPU job submitted."
  exit 0
fi

command -v sbatch >/dev/null 2>&1 || {
  echo "Slurm sbatch is unavailable; submit this from the Helix cluster." >&2
  exit 1
}

# Submit CPU scoring with an optional afterok dependency on GPU inference.
queue_score() {
  local dependency="${1:-}"
  local score_cmd job
  local -a opts=(
    --parsable
    --job-name=ulm-spatial-score
    "--partition=$CPU_PARTITION"
    --cpus-per-task=2
    --mem=16G
    --time=01:00:00
    "--chdir=$ROOT"
    "--output=$RUN_DIR/score-%j.log"
  )
  [[ -z "$dependency" ]] || opts+=("--dependency=afterok:$dependency")
  printf -v score_cmd '%q ' "$PYTHON" "$SCRIPT" score --batch "$BATCH"
  job="$(sbatch "${opts[@]}" --wrap="$score_cmd")"
  job="${job%%;*}"
  [[ "$job" =~ ^[0-9]+$ ]] || { echo "Invalid scoring job ID: $job" >&2; return 1; }
  printf '%s\n' "$job"
}

if [[ "$ACTION" == score-only ]]; then
  [[ -f "$RUN_DIR/predictions.json" ]] || {
    echo "Missing predictions.json; run inference first." >&2
    exit 1
  }
  SCORE_JOB="$(queue_score)"
  echo "CPU scoring submitted: $SCORE_JOB"
  exit 0
fi

# prepare() intentionally refuses to overwrite an existing experiment folder.
echo "[1/3] Preparing spatial OCR text from saved LaOCR data..."
"$PYTHON" "$SCRIPT" prepare --batch "$BATCH"
mkdir -p "$RUN_DIR"

echo "[2/3] Submitting Qwen3.5-9B inference..."
GPU_JOB="$(sbatch --parsable \
  --job-name=ulm-spatial-qwen35 \
  "--partition=$GPU_PARTITION" \
  "--gres=$GPU_GRES" \
  --cpus-per-task=4 --mem=64G "--time=$GPU_TIME" \
  "--output=$RUN_DIR/spatial-%j.log" \
  "$SBATCH_SCRIPT" "$BATCH")"
GPU_JOB="${GPU_JOB%%;*}"
[[ "$GPU_JOB" =~ ^[0-9]+$ ]] || { echo "Invalid GPU job ID: $GPU_JOB" >&2; exit 1; }

echo "[3/3] Queuing official scoring after successful inference..."
SCORE_JOB="$(queue_score "$GPU_JOB")"
echo "GPU job: $GPU_JOB; dependent scoring job: $SCORE_JOB"
echo "Monitor: squeue -j $GPU_JOB,$SCORE_JOB"
echo "Results: $RUN_DIR"
echo "The original six model runs remain unchanged."
