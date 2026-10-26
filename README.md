# Image–Text Extraction with VLM and OCR→LM

**Publication-ready research implementation and reproducible evaluation for OCRBench v2 English key information extraction.**

Compare direct vision-language models against OCR-based language-model pipelines with **400 matched English KIE items**, the unmodified benchmark questions and official item-level scoring. The project ran on bwForCluster **Helix**, using pinned model revisions and greedy decoding. No model training or custom OCRBench scoring was performed.

> **Reproduce the paper without a GPU.** This repository includes a small lossless derivative of the **seven final officially scored runs**. The exact paper accuracy tables, stratified paired bootstrap intervals, and figures are rebuilt from those scores on an ordinary CPU. Neither OCRBench images, questions, ground-truth answers, nor pretrained model weights are redistributed.

## Results at a glance

| Pipeline / publication label | Official mean KIE score |
| --- | ---: |
| **Qwen3.5-9B direct VLM** · `qwen35_vlm` | **0.870673** |
| PaddleOCR flat text → Qwen3.5-9B · `qwen35_ocr` | 0.767842 |
| Coordinate-aware PaddleOCR → Qwen3.5-9B · `qwen35_laocr` | 0.742936 |
| PP-StructureV3 **recovered** Markdown → Qwen3.5-9B · `qwen35_ppstructure_v2` | 0.722947 |
| PaddleOCR flat text → Qwen2.5-14B · `qwen25_ocr` | 0.655764 |
| PP-StructureV3 **original** Markdown → Qwen3.5-9B · `qwen35_ppstructure_v1` | 0.590125 |
| MiniCPM-V-4.6 direct VLM · `minicpm_vlm` | 0.559314 |

The most controlled architectural comparison—**direct Qwen3.5 versus flat PaddleOCR→the same Qwen3.5 model**—shows a 0.102831 mean score advantage (10.283 percentage points). Its previously computed 95% paired bootstrap CI is approximately **8.36–12.21 points**. Because the bootstrap resamples the *same observed documents*, it does not demonstrate transfer to an unrelated document population.

The original PP Markdown v1 and recovered Markdown v2 results are **separate scored runs**, despite the inference code historically using one experiment ID, `qwen35_ppstructure`. Their publication labels and cached formats distinguish them unambiguously.

See [detailed findings](docs/RESULTS.md), [evaluation methodology](docs/METHODOLOGY.md) and the [immutable seven-run registry](analysis/experiments.csv).

## Reproduce all publication outputs (CPU only)

**Python 3.12** is required. These are working repository commands, not illustrative placeholders.

```bash
git clone https://github.com/econoberries/Image-Text-Extractor-with-VLM-and-OCRLM.git
cd Image-Text-Extractor-with-VLM-and-OCRLM

python3.12 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r requirements/analysis-lock.txt
.venv-analysis/bin/python -m pip install -e .

# Validates all 400 matched items and all seven official model means:
.venv-analysis/bin/python scripts/validate_final_results.py

# Generates three bootstrap panels, six CSV result tables, figures and provenance:
.venv-analysis/bin/python scripts/build_final_results.py --draws 10000 --seed 2026
```

Outputs are written to `results/final/` (intentionally Git-ignored to avoid accidentally committing stale generated results). Their generator, inputs and expected data checks are version-controlled. The [GitHub Actions test workflow](.github/workflows/tests.yml) regenerates and uploads the complete reproduction artifact on every push.

If you also have the separate historical `runs(3).zip` archive, validate the **raw** predictions, hashes, and cache formats—not just the republished scores:

```bash
.venv-analysis/bin/python scripts/validate_final_results.py --archive "runs(3).zip"
```

The scored-run archive is **not** published with this repository pending an independent redistribution review. The public exact-score derivative *is* sufficient to reproduce the statistical calculations, but without that archive outsiders cannot independently inspect or verify raw model generations. Full reproducibility instructions and interpretation boundaries are in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Architectures and executable code

```text
Image ----------------------------------------> Qwen3.5-9B / MiniCPM-V-4.6
    |
    +--> PaddleOCR (flat text) ----------------> Qwen3.5-9B / Qwen2.5-14B
    |
    +--> PaddleOCR (text + bounding boxes) ----> Qwen3.5-9B
    |
    +--> PP-StructureV3 (Markdown)
           + English full-image OCR recovery ---> Qwen3.5-9B
```

- `src/ulm_extractor/`: image validation, model adapters, OCR/layout extraction, Markdown recovery, worker, batching, resume, official scoring and publication analysis.
- `model.yml`, `model-baselines.yml`, `model-laocr.yml`, `model-ppstructure.yml`: explicit model configurations.
- `slurm/`: isolated, dependency-aware Slurm GPU workers and a CPU collector.
- `scripts/`: CPU publication reproduction plus GPU smoke/dependency checks.
- `analysis/`: seven-run registry, lossless score panel, and historically observed efficiency.
- `docs/`: research methods, results, reproducibility, and the full Helix inference runbook.
- `tests/`: offline inference orchestration, OCR text recovery, publication statistics and data-integrity tests.

The original source code used for publication is anchored to [research commit `4aff4fb`](https://github.com/econoberries/ULM-Project-Extractor-VLM-OCRLM/tree/4aff4fb97a5f818884584720bd2e607c7a1f4245). No historical predictions or batch UUIDs were overwritten when publishing this repository.

## Full inference reproduction (GPU/Helix)

If you want to regenerate model predictions from OCRBench images rather than just rebuild the analysis, follow the [separate Helix inference runbook](docs/HELIX_INFERENCE.md). That explains the distinct Torch and Paddle CUDA environments, model pinning, diagnostics, Slurm scheduling, and the official upstream scorer.

**Important limitations:** the paper's processing-time estimates combine separately measured OCR and LM stages; PyTorch peak-memory numbers do **not** include PaddlePaddle OCR memory. Archived experiments used a mixture of A100 **40 GB** and **80 GB** devices, so their timing/memory metrics are not a controlled same-hardware efficiency experiment. See [analysis/efficiency_observations.csv](analysis/efficiency_observations.csv).

## Tests

```bash
.venv-analysis/bin/python -m unittest discover -s tests -v
```

CI additionally re-runs the full 10,000-draw CPU bootstrap and renders the figures. Offline tests use mocks for GPU-dependent functionality; passing them does **not** establish that new GPUs reproduce the historical model tokens bit-for-bit.

## References, citation and permissions

- [Official OCRBench v2 code, data access and evaluator](https://github.com/Yuliang-Liu/MultimodalOCR)
- [OCRBench v2 research paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/8c2e6bb15be1894b8fb4e0f9bcad1739-Abstract-Datasets_and_Benchmarks_Track.html)
- [Project citation metadata](CITATION.cff)
- [Source/archive provenance](docs/REPRODUCIBILITY.md)

**License status:** repository code is available for inspection and research reproduction. The project team should approve an explicit open-source license before third-party code reuse is authorized. Dependencies, upstream evaluator, benchmark assets, and model weights retain their own licenses; no right to redistribute them is implied.
