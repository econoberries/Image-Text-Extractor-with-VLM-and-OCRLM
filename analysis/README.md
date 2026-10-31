# Statistical analysis

This folder explains how to generate the final statistical results. The six model jobs produce their predictions and official OCRBench v2 scores before analysis begins.

Once every run has `status=scored`, run the CPU-only finalizer from the repository root:

```bash
.venv-analysis/bin/python scripts/finalize_benchmark.py \
  --batch runs/batches/<BATCH_ID>/batch.json
```

The generated files are written to `runs/analysis/<BATCH_ID>/`, which is excluded from Git. Outputs include four- and six-model confidence intervals, paired bootstrap comparisons, aligned document-level scores, descriptive efficiency metrics, figures and provenance.

For the bootstrap method, assumptions, and complete setup instructions, use **Sections 9–11 of the root README**.

The experiments folder contain the scripts for an extension analysis that aggregates the data for different model runs and then reruns a new function that returns true positives, false positives, false negatives, a count of omitted values, and a count of correct keys/wrong values for a benchmark item. These were for calculating additional accuracy metrics: exact-match rate (EMR) is the sum of all benchmark items where all requested key value pairs were correctly returned, requested field accuracy (RFA) is the proportion of questions for which all requested pairs were extracted correctly.
