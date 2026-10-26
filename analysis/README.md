# Final analysis

The new experiment uses one fresh six-model batch rather than the previous seven-run exploratory score archive.

After all six models have `status=scored`, run:

```bash
.venv-analysis/bin/python scripts/finalize_benchmark.py --batch runs/batches/<NEW_BATCH_UUID>/batch.json
```

The results are written into the **Git-ignored** `runs/analysis/<NEW_BATCH_UUID>/` directory: primary and six-model score CIs, pairwise differences, descriptive efficiency, aligned item scores, figures and provenance.

The statistical implementation is `scripts/bootstrap_kie.py`. See the **root README, Sections 9–11**, for assumptions and commands.

Historical publication scores and figures can still be recovered from the Git history at commit `c14567cab246ca9c26d575fece9da4b5983b44aa` or from the separate research repository. They are not valid results for the new controlled rerun.
