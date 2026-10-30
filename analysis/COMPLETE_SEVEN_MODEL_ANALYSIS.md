# Complete seven-model analysis + 16-case qualitative review

This add-on creates **one new, complete timestamped run folder** from the official scored results already saved on Helix:

`runs/analysis/ULM_KIE_Full_Analysis_YYYYMMDD_HHMMSSZ/`

It combines the **original six models** with the already scored **seventh Spatial OCR → Qwen3.5-9B** experiment on the same 400 English KIE questions. It **does not rerun models, OCR, or scoring**, overwrite earlier analyses, edit existing `runs/`, or submit Slurm jobs.

## How to run on your EXISTING Helix project

Your completed data and Python environments live in:
`~/Image-Text-Extractor-with-VLM-and-OCRLM`

You do not need to switch Git branches in that worktree. To copy only this script into it:

```bash
cd ~/Image-Text-Extractor-with-VLM-and-OCRLM
git fetch origin analysis/seven-model-complete
git show origin/analysis/seven-model-complete:scripts/build_complete_seven_model_analysis.py > scripts/build_complete_seven_model_analysis.py
```

Check that your CPU analysis environment has NumPy and Matplotlib:

```bash
.venv-analysis/bin/python -c 'import numpy, matplotlib; print("Analysis dependencies OK")'
```

Preview the input validation and 16 selected cases **without writing anything**:

```bash
BATCH="runs/batches/cbdd58de-531a-4c9c-9778-5bc8b0b0be48/batch.json"

.venv-analysis/bin/python scripts/build_complete_seven_model_analysis.py \
  --batch "$BATCH" --dry-run
```

After checking the preview, generate the full folder and its shareable ZIP:

```bash
.venv-analysis/bin/python scripts/build_complete_seven_model_analysis.py \
  --batch "$BATCH" \
  --n-bootstrap 10000 --seed 2026
```

The output folder and the ZIP with the **same timestamp** appear in `runs/analysis/`. Look for `ULM_KIE_Full_Analysis_*.zip` beside the folder.

## Folder structure

```text
ULM_KIE_Full_Analysis_YYYYMMDD_HHMMSSZ/
├── README.md                         Human-readable seven-model summary
├── model_runs/                       Seven complete scored run snapshots
│   ├── qwen35_vlm/
│   ├── minicpm_vlm/
│   ├── qwen35_ocr/
│   ├── qwen25_ocr/
│   ├── qwen35_laocr/
│   ├── qwen35_ppstructure/
│   └── qwen35_spatial_ocr/
├── tables/
│   ├── overall_model_summary.csv
│   ├── scores_by_dataset.csv
│   ├── seven_model_bootstrap_ci.csv
│   ├── seven_model_paired_bootstrap_ci.csv
│   ├── seven_model_document_scores.csv
│   ├── efficiency.csv
│   ├── spatial_head_to_head.csv
│   ├── qualitative_16_cases_selection.csv
│   └── qualitative_field_diagnostics.csv
├── figures/                          Seven figure families, PNG + PDF
├── qualitative_16_cases/
│   └── case_XX_<dataset>_<id>/
│       ├── case.json                 Seven predictions, official scores, gold
│       ├── review.md                 Detailed manual-review checklist
│       ├── spatial_text.txt          If saved spatial text is available
│       └── source_image.*            Only with --image-root
├── previous_statistical_analysis/    Existing results and figures if present
└── provenance/
    ├── manifest.json                 Dataset and run identity, SHA256, caveats
    └── output_checksums.csv
```

## Same 16 cases versus new 16 cases

By default, the script selects **four cases per dataset**, emphasizing spatial gains, losses and divergences with flat OCR, coordinate JSON and PP-Structure. These are *purposefully chosen cases*, not a random or representative sample.

To inspect the **same exact 16 cases** as your previous qualitative review, supply a CSV/JSON/TXT file containing 16 document keys (or unique OCRBench IDs; CSV can have `dataset_name,id`). For example:

```bash
.venv-analysis/bin/python scripts/build_complete_seven_model_analysis.py \
  --batch "$BATCH" --case-ids-file previous_16_ids.csv
```

If the old review folder exists, you can preserve it alongside the new analysis:

```bash
.venv-analysis/bin/python scripts/build_complete_seven_model_analysis.py \
  --batch "$BATCH" \
  --existing-review runs/analysis/layout_case_review_extended
```

If the 16 previous case identities cannot be recovered, **do not claim the new 16 are the same 16**. Use the default selection and record that as a new diagnostic sample.

To include copies of selected images in this private sharing package, optionally supply `--image-root "$HOME/OCRBench_v2/EN_part"`; omit it if images must not be redistributed.

## Reproduction / methodological caveats

- All model accuracy data come from the existing **official OCRBench v2 per-question KIE score**. The CSV of heuristic normalized field matches is only for exploratory qualitative diagnostics, **not** an official F1 score.
- Seven models, 400 matched English questions, 4 subsets (FUNSD 48, POIE 177, Receipt 25, SROIE 150).
- 10,000 paired, subset-stratified bootstrap draws; 21 seven-model pairwise comparisons; Bonferroni-adjusted intervals are included.
- Spatial inference reused LaOCR recognition and its saved OCR timing. This is *not* measured end-to-end wall time. PyTorch VRAM omits PaddleOCR GPU allocations.
- Human annotation of error mechanisms is deliberately left blank for review: official scores alone cannot establish what caused a mistake.
- The ZIP contains raw predictions and references; share privately with approved collaborators, not by committing it to the public code repository.

## Download from Helix to Mac

From your **Mac terminal** after the run:

```bash
scp 'tu_zxope66@helix.bwservices.uni-heidelberg.de:~/Image-Text-Extractor-with-VLM-and-OCRLM/runs/analysis/ULM_KIE_Full_Analysis_*.zip' \
  "/Users/khaira_abdillah/Documents/SoSe26/ulm_project_extractor/"
```

This is a read-only analysis of saved predictions. Running `--dry-run` is recommended first.
