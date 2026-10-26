# Requirements

- `helix-pytorch-cu130.txt`: pinned CUDA PyTorch/Torchvision wheels for `.venv`.
- `helix-paddle-cu130.txt`: Paddle GPU wheel, installed **only** in isolated OCR environments.
- `helix-torch-repair-cu130.txt`: optional recovery after incorrectly mixing Torch and Paddle.
- `ocrbench-v2-kie-eval.txt`: dependencies for the unmodified official English KIE evaluator.
- `analysis-lock.txt`: CPU-only dependencies for final statistics and figures.

The **root README, Step 5.2** shows the required installation order.
