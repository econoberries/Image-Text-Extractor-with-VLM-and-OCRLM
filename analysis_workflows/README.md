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

The analysis ran successfully on Helix and its outputs are in `runs/analysis/hypotheses_400/`. **You do not need to run it again to share these results.** The script used in the original terminal session is preserved in `runs/analysis/kie_hypotheses_400.py`; a matching version based on that script has now been committed to this analysis branch.

You can check both versions are identical on Helix:

```bash
cmp runs/analysis/kie_hypotheses_400.py analysis_workflows/kie_hypotheses_400.py && echo "Scripts identical"
```

The script includes default paths for batch `cbdd58de-531a-4c9c-9778-5bc8b0b0be48` and writes to `runs/analysis/hypotheses_400/`. **It refuses to overwrite existing outputs.** There is no need to rerun it for this shared package.

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

## Qualitative case reproduction

The frozen 16-case selection, manually reviewed notes and read-only reproduction scripts are documented in [qualitative/README.md](qualitative/README.md).

- Generate a Markdown report from saved official scores and manual notes:

```bash
.venv-analysis/bin/python analysis_workflows/qualitative/build_case_report.py
```

- Optionally reconstruct private OCR/prediction evidence in a fresh ignored folder:

```bash
.venv-analysis/bin/python analysis_workflows/qualitative/export_cases.py --include-images
```

These scripts do not alter the original 16-case archive, official scores, or benchmark code. Never commit reconstructed evidence or original benchmark images.

## Scope

The analysis tools and documentation are independent of `src/`, `scripts/finalize_benchmark.py`, and `scripts/bootstrap_kie.py`. No existing source file, scoring code, or model configuration is changed by this folder.
## Teammate visualizations

Generate six draft PNG figures **only from existing CSVs** (overall accuracy and CIs, dataset breakdown, paired wins/ties/losses, accuracy versus processing time, accuracy versus Torch memory, and token-limit hits):

```bash
.venv-analysis/bin/python analysis_workflows/create_team_figures.py
```

This saves `runs/analysis/team_figures/` and refuses to overwrite an existing nonempty folder. For a rerun, use `--output-dir runs/analysis/team_figures_v2`. All figures use the existing `kie_scores_0to1/` and `hypotheses_400/` outputs. No score or bootstrap is recalculated.

A privately shared teammate package should contain derived figures, summary tables, and the qualitative case summary; do **not** upload the source benchmark images or raw predictions to a public repository.
