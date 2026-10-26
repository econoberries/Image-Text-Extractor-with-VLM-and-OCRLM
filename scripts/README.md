# Scripts

- `check_*.py`: short dependency/GPU/OCR/model checks before a full run.
- `bootstrap_kie.py`: resamples matched official document scores, with replacement and fixed dataset subgroup counts.
- `finalize_benchmark.py`: **after scoring**, checks six model runs and OCR worker GPUs, calculates final paired CIs and saves result tables and figures.

The scripts are called using the Python environments and commands in the **root README**. Do not execute CUDA checks on a login node.
