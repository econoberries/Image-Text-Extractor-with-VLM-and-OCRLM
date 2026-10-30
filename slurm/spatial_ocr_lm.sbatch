#!/usr/bin/env bash
# Standalone seventh experiment (not part of the original six-model Slurm pipeline).
# Request an A100 80GB class GPU using sbatch CLI flags, as documented in experiments/README.md.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch [GPU options] slurm/spatial_ocr_lm.sbatch runs/batches/<id>/batch.json" >&2
    exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
    echo "Missing .venv/bin/python; use the existing Torch/Qwen environment." >&2
    exit 2
fi
export TOKENIZERS_PARALLELISM=false
exec .venv/bin/python experiments/spatial_ocr_lm.py infer --batch "$1"
