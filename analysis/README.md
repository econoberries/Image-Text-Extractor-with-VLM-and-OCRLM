# Statistical analysis

This folder explains how to generate the final statistical results. The six model jobs produce their predictions and official OCRBench v2 scores before analysis begins.

Once every run has `status=scored`, run the CPU-only finalizer from the repository root:

```bash
.venv-analysis/bin/python scripts/finalize_benchmark.py \
  --batch runs/batches/<BATCH_ID>/batch.json
```

The generated files are written to `runs/analysis/<BATCH_ID>/`, which is excluded from Git. Outputs include four- and six-model confidence intervals, paired bootstrap comparisons, aligned document-level scores, descriptive efficiency metrics, figures and provenance.

For the bootstrap method, assumptions, and complete setup instructions, use **Sections 9–11 of the root README**.
