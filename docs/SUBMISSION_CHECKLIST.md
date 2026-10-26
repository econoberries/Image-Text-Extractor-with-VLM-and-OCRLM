# Submission and publication checklist

- [x] Preserve full inference code, fixed checkpoint revisions and Slurm launchers from frozen research commit.
- [x] Retain the primary four, expanded six, and PP old-vs-recovered comparisons with unique publication labels.
- [x] Publish 400 × seven exact official item scores without redistributing OCRBench annotations, questions or images.
- [x] Record seven run UUIDs, batch UUIDs, original raw-score SHA256 digests and evaluator/data provenance.
- [x] Reproduce paired, subset-stratified 10,000-draw bootstrap results and figures with one CPU-only command.
- [x] Document inference and resampling limitations and the A100-40GB/A100-80GB mixed hardware.
- [x] Test with GitHub Actions and retain a downloadable reproduction artifact per passing workflow.
- [ ] **Team approval:** confirm author order, attribution and which software license (if any) the rights holders agree to grant.
- [ ] **Submission freeze:** review a final paper against the generated numeric CSVs and figures; then create an immutable version tag/Zenodo archive manually.
- [ ] **Optional external audit:** run `validate_final_results.py --archive runs(3).zip` in a clean environment with the original archive. It is not part of CI because redistribution of raw benchmark materials has not been authorized.
- [ ] **Optional full GPU replication:** rerun models on Helix; not required for re-creating already published accuracy tables.

**Scope note:** the CPU score panel reproduces published statistical analyses exactly. The full upstream OCRBench data and pretrained checkpoints are needed to reproduce end-to-end model inference, while historical PP Markdown v1 additionally requires its original unmodified implementation.
