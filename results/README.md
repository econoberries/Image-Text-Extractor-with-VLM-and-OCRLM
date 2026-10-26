# Generated publication artifacts

`results/final/` is intentionally **not** version-controlled. It is regenerated deterministically from the frozen seven-run exact-score panel by:

```bash
python scripts/build_final_results.py --draws 10000 --seed 2026
```

The command produces all final CSV tables and plotted confidence intervals; its inputs reside in [analysis](../analysis/). Check the repository [GitHub Actions workflow](../.github/workflows/tests.yml) for a passing clean-environment reproduction and a downloadable `publication-reproduction` artifact.

Outputs will match the historical 10,000-resample analysis within numerical precision using the documented NumPy version and random seeds. The three panels use seeds 2026, 2027, and 2028; running them separately with other seeds leads to small Monte Carlo differences, not new model evidence.

For full independent original-data provenance checks, use `scripts/validate_final_results.py --archive "runs(3).zip"`. The public repository does not contain the original raw-score archive.
