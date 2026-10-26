# Seven-run publication dataset

This directory contains an **exact-score** derivative of OCRBench v2 English KIE evaluation and provenance of the seven completed experiments.

- `experiments.csv` maps each publication label to an immutable batch UUID, run UUID, checkpoint revision, original scored-file SHA256, official mean, and OCR representation.
- `scored_documents.json` stores the official document-level score panel in a compact, lossless dictionary-index format. The seven columns are aligned on 400 English KIE questions numbered 900–1299. Four-way subset membership is packed in two bits per row. The script expands this automatically.
- `efficiency_observations.csv` preserves the run-level timing and Torch-memory summary, explicitly including observed A100 40 GB versus 80 GB devices.

The original OCRBench questions, answers and images are **not** included. Model raw predictions and OCR cache outputs are also not redistributed. The compact score panel contains measured derivative scores only.

The original dataset manifest SHA256 is `fc28721b7e3fef4cd1226a198a5fbdf92b62f28e83de7592defc31d555c48214`.

To independently verify the derivative scores and raw-file SHA256 checks, use `python scripts/validate_final_results.py --archive "runs(3).zip"` after installing the analysis environment. Without `--archive`, the validator checks the published panel and its agreement with the seven official run means but **cannot** independently establish original file provenance.

**Important:** `qwen35_ppstructure_v1` and `qwen35_ppstructure_v2` are publication labels; the original inference configuration used `qwen35_ppstructure` for both, with different cache formats and run UUIDs. Never mix their results.
