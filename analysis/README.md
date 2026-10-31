# Statistical analysis

This folder explains how to generate the final statistical results. The six model jobs produce their predictions and official OCRBench v2 scores before analysis begins.

Once every run has `status=scored`, run the CPU-only finalizer from the repository root:

```bash
.venv-analysis/bin/python scripts/finalize_benchmark.py \
  --batch runs/batches/<BATCH_ID>/batch.json
```

The generated files are written to `runs/analysis/<BATCH_ID>/`, which is excluded from Git. Outputs include four- and six-model confidence intervals, paired bootstrap comparisons, aligned document-level scores, descriptive efficiency metrics, figures and provenance.

For the bootstrap method, assumptions, and complete setup instructions, use **Sections 9–11 of the root README**.

### Extension Analysis

The `experiments/` folder contains the scripts used for the extension analysis.

- **`aggregate_model_json.py`**: Aggregates the data generated across different model runs.

- **`vis_accuracyvsEff.py`**: Re-runs the evaluation using an updated function that returns the following metrics for each benchmark item:
  - True positives (TP)
  - False positives (FP)
  - False negatives (FN)
  - Number of omitted values
  - Number of correctly extracted keys with incorrect values

  These additional outputs are used to calculate two accuracy metrics:
  - **Exact-Match Rate (EMR):** The proportion of benchmark items for which all requested key-value pairs were returned correctly.
  - **Requested Field Accuracy (RFA):** The proportion of questions for which all requested key-value pairs were extracted correctly.

- **`visualize_processtimes.py`**: Generates the figure illustrating the accuracy–efficiency trade-off.
