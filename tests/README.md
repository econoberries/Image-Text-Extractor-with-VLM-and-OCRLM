# Offline tests

`python -m unittest discover -s tests -v` checks configuration, synthetic OCR/document processing, Slurm scheduling and resumes, paired bootstrap logic, and hardware-verifying finalization.

These tests do **not** download pretrained models or establish GPU inference compatibility; use the Helix smoke tests in the **root README, Step 6** for that.
