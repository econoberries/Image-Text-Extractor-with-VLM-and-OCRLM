# Post-benchmark research analysis (read-only)

This folder is separate from the model execution and official benchmark pipeline. It uses the existing six scored runs and three cached OCR outputs from the **400-item English KIE** evaluation of OCRBench v2.

**No inference, OCR, scoring or bootstrap is rerun.** Existing data under `runs/` is not tracked by Git.

## Verify the Helix files first

From the repository root on Helix:

```bash
.venv-analysis/bin/python analysis_workflows/check_helix_inventory.py
```

It verifies the six completed prediction files, the three OCR cache indexes, the original bootstrap output, the five H1–H3 diagnostic CSVs, and the 16 inspected cases. It reports missing files without modifying anything.

## Reproduce the five H1–H3 diagnostic CSVs

The analysis originally ran successfully on Helix and its outputs are in `runs/analysis/hypotheses_400/`. **You do not need to run it again to share these results.** The saved script on Helix is `runs/analysis/kie_hypotheses_400.py` and is the authoritative copy for exact reproduction of those five files.

To version-control that exact script, copy it into this folder before making a commit (and compare it with the GitHub copy before relying on exact reproducibility):

```bash
cp runs/analysis/kie_hypotheses_400.py analysis_workflows/kie_hypotheses_400.py
```

The GitHub script was reconstructed from the working Helix version; compare it against the original before using it for exact reproduction. The script includes default paths for batch `cbdd58de-531a-4c9c-9778-5bc8b0b0be48` and writes to `runs/analysis/hypotheses_400/`. **It refuses to overwrite existing outputs.** For a reproduction run, use a fresh output directory after reviewing the script.

The existing five outputs are:
- `kie_400_questions.csv` (400 rows): official per-question scores, output diagnostics, OCR representation characteristics.
- `kie_400_fields.csv` (2,831 rows): diagnostic parsed field matches across all six systems; not official scoring.
- `hypothesis_pairwise_descriptive.csv` (20 rows): H1 and H2 paired-score descriptive summaries.
- `model_error_summary.csv` (30 rows): model error counts by dataset.
- `efficiency_descriptive.csv` (six rows): saved processing-time and Torch-memory metrics.

The authoritative bootstrap confidence intervals remain in `runs/analysis/kie_scores_0to1/` and are **not** computed by this workflow.

## Qualitative case evidence

The 16 deliberately selected examples are saved in `runs/analysis/layout_case_review_extended/` on Helix. They contain original benchmark document images, reference answers, saved OCR representations, and model predictions. These cases illustrate *possible* failure mechanisms; they are **not** a random sample of 400 questions.

Do not commit benchmark images, raw OCR text, references, raw predictions, dataset content, or full run archives to public GitHub. Keep those under `runs/` or share privately under the benchmark's redistribution terms.

## Scope

The analysis tools and documentation are independent of `src/`, `scripts/finalize_benchmark.py`, and `scripts/bootstrap_kie.py`. No existing source file, scoring code, or model configuration is changed by this folder.