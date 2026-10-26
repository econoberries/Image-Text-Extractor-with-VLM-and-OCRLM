# Reproduction guide

There are two distinct targets. **Level A (the submission minimum)** regenerates the paper tables and bootstrap figures from the published, exact 400-item official-score derivative without GPU/model downloads. **Level B** rebuilds predictions from the original OCRBench v2 data using Slurm and the pinned model revisions.

## Level A — CPU-only reproduction

Prerequisites: Python 3.12, Git, internet access for Python packages. On any standard Linux/macOS computer:

```bash
git clone https://github.com/econoberries/Image-Text-Extractor-with-VLM-and-OCRLM.git
cd Image-Text-Extractor-with-VLM-and-OCRLM
python3.12 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r requirements/analysis-lock.txt
.venv-analysis/bin/python -m pip install -e .
.venv-analysis/bin/python scripts/validate_final_results.py
.venv-analysis/bin/python scripts/build_final_results.py --draws 10000 --seed 2026
```

These are the **implemented** scripts; no placeholders or Helix account are required. The final outputs are generated under `results/final/`. The build is idempotent and reproduces tables when called again with the same input and seed. The repository contains all needed official *per-item scores*, not OCRBench images, questions, or model-generated text.

Generated files:

| File | Contents |
| --- | --- |
| `all_seven_document_scores.csv` | Aligned official scores for 400 items × 7 runs |
| `primary_model_confidence_intervals.csv` | Model means and stratified bootstrap intervals (original four) |
| `primary_paired_differences.csv` | Six paired comparisons and Bonferroni intervals |
| `expanded_* .csv` | Six-model supplementary comparisons |
| `pp_recovery_* .csv` | Historical PP Markdown v1 versus recovered Markdown v2 |
| `efficiency.csv` | Historical timing and hardware observations, not new benchmarks |
| `figure_model_accuracy.png` | Score means with percentile bootstrap intervals |
| `figure_primary_pairwise.png` | Primary paired differences |
| `figure_pp_recovery.png` | PP recovery improvement |
| `validation_report.json` | Reproduction parameters, source fingerprints and caveats |

### Optional strong verification against the original full scored-run archive

If you have the original **`runs(3).zip`** in your working directory:

```bash
.venv-analysis/bin/python scripts/validate_final_results.py --archive "runs(3).zip"
.venv-analysis/bin/python scripts/build_final_results.py --archive "runs(3).zip"
```

The archive audit checks the original scored-file SHA256 digests, batch manifests, model/run IDs, all 2,800 official item-level scores, and the PP v1/v2 cache format signatures. The full archive is not included in the public repository: it contains benchmark references, generated predictions and intermediate OCR content, and its redistribution would need an independent permissions and dataset-licensing review. **Without the archive the public data remains sufficient to reproduce the statistical tables, but not to independently audit raw model outputs.**

## Level B — End-to-end inference reproduction (Helix / Slurm)

1. Obtain OCRBench v2 media and annotations from the [official upstream repo](https://github.com/Yuliang-Liu/MultimodalOCR). Pin the upstream evaluator commit `4e363b3f499811b43d792ef0bb33932ef05eca27` and record the local dataset manifest hash.
2. Follow [HELIX_INFERENCE.md](HELIX_INFERENCE.md) to create the **four separate** CUDA/PyTorch/Paddle environments; do not mix PaddlePaddle and Torch's conflicting CUDA-library builds.
3. Run the GPU smoke tests in Slurm and check actual GPU model and VRAM.
4. Execute `extractor benchmark --input "$EN_PART" --models model.yml` with correct Slurm flags. This runs six current architectures, including **PP recovered Markdown v2**. It will not generate the historical original PP v1 result, because the publication source intentionally preserves the corrected implementation.
5. Confirm 400 successful scored items per implementation with the official OCRBench v2 evaluator; record the new run UUIDs. Do **not** expect byte-identical model generations across different hardware/framework versions even under greedy decoding.

The historical v1 original Markdown implementation and its scores remain separately identified in the seven-run registry and archival evidence. Reconstruction of the historical experiment code requires using its original historical source version, not today's corrected PP module.

## Smoke and unit tests

```bash
.venv-analysis/bin/python -m unittest discover -s tests -v
```

Some offline tests use fake CUDA workers and cannot prove production GPU compatibility. GitHub Actions reproduces the published CPU analysis outputs on each push and uploads a temporary `publication-reproduction` artifact. No CI workflow trains or runs a large language model.

## Provenance and interpretation

- Frozen source repository commit: `4aff4fb97a5f818884584720bd2e607c7a1f4245`.
- Shared OCRBench v2 KIE manifest SHA256: `fc28721b7e3fef4cd1226a198a5fbdf92b62f28e83de7592defc31d555c48214`.
- Original seven scored-file SHA256 hashes and model checkpoint revisions: [experiments.csv](../analysis/experiments.csv).
- PP original and recovery implementations are **separate analysis labels** attached to separate scored-run UUIDs.
- The confidence intervals describe variation among documents represented by this benchmark, not extrapolation to all real-world documents.

See [METHODOLOGY.md](METHODOLOGY.md) and [RESULTS.md](RESULTS.md).
