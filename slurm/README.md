# Slurm workers

- `check_environment.sbatch`: checks the actual GPU, available VRAM and one-image diagnostics.
- `worker.sbatch`: GPU preprocessing and model inference, using separate Torch/Paddle environments and a fail-closed VRAM check.
- `collector.sbatch`: CPU-only official evaluation and result collection.

**Do not call worker jobs manually for a full batch.** Use `extractor benchmark` so dependencies, run UUIDs, and provenance are written correctly. See the **root README, Sections 5–8**.
