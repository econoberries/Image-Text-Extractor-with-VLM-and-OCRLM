# How the 16 qualitative cases were selected

## Scope and purpose

This is the **actual 16-case selection** from the completed seven-model OCRBench v2 English KIE analysis, batch `cbdd58de-531a-4c9c-9778-5bc8b0b0be48`, generated **2026-10-30 23:38:17 UTC**. The 16 different questions are drawn from the same 400 English KIE questions scored across all seven models: four from each of **FUNSD**, **POIE**, **Receipt**, and **SROIE**.

**Crucially, this is a new, deliberately selected diagnostic sample, not a random sample and not necessarily the same 16 examples from the older `layout_case_review_extended` analysis.** Its purpose is to illustrate when reconstructing spatial text helps or harms field extraction.

## Exact selection algorithm

For each dataset separately, consider every document and use the **official OCRBench item score** for each model. Pick one *different* case in each of the following ordered categories:

1. **Spatial gain:** maximize `spatial_score - flat_ocr_score`.
2. **Spatial loss:** maximize `flat_ocr_score - spatial_score` among remaining documents.
3. **Spatial versus coordinate JSON:** maximize `abs(spatial_score - laocr_score)` among remaining documents.
4. **Spatial versus PP-Structure and general disagreement:** maximize `abs(spatial_score - ppstructure_score) + std(scores_of_six_original_models)` among remaining documents.

Ties are resolved by sorting OCRBench IDs as strings, then by original input position. The criteria are applied *sequentially*: the third and fourth cases are selected from candidates not already chosen. The algorithm does not manually choose easy or hard examples after viewing the images.

## Actual 16 selected cases

`Flat` = Qwen3.5-9B on plain PaddleOCR text. `LaOCR` = Qwen3.5-9B on coordinate/bounding-box JSON. `PP` = Qwen3.5-9B on PP-StructureV3 recovered Markdown. `Spatial` = Qwen3.5-9B on reconstructed line/space text. Scores are official OCRBench item scores rounded to 3 decimals.

| # | Subset | OCRBench ID | Why selected | Flat | LaOCR | PP | Spatial | Spatial − Flat |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 1 | FUNSD | 1044 | Spatial gain | 0.433 | 0.370 | 0.630 | 0.815 | +0.381 |
| 2 | FUNSD | 1023 | Spatial loss | 1.000 | 1.000 | 0.667 | 0.667 | -0.333 |
| 3 | FUNSD | 1212 | Spatial–LaOCR disagreement | 0.600 | 0.000 | 0.500 | 0.600 | +0.000 |
| 4 | FUNSD | 932 | Spatial–PP and model disagreement | 1.000 | 1.000 | 0.600 | 1.000 | +0.000 |
| 5 | POIE | 917 | Spatial gain | 0.091 | 0.818 | 0.545 | 0.727 | +0.636 |
| 6 | POIE | 1169 | Spatial loss | 0.813 | 0.563 | 1.000 | 0.313 | -0.500 |
| 7 | POIE | 1256 | Spatial–LaOCR disagreement | 1.000 | 0.000 | 1.000 | 1.000 | +0.000 |
| 8 | POIE | 934 | Spatial–PP and model disagreement | 1.000 | 0.000 | 0.000 | 1.000 | +0.000 |
| 9 | Receipt | 1009 | Spatial gain | 0.500 | 0.750 | 0.750 | 1.000 | +0.500 |
| 10 | Receipt | 1057 | Spatial loss | 0.750 | 0.750 | 0.750 | 0.500 | -0.250 |
| 11 | Receipt | 1114 | Spatial–LaOCR disagreement | 0.750 | 1.000 | 1.000 | 0.750 | +0.000 |
| 12 | Receipt | 1200 | Spatial–PP and model disagreement | 1.000 | 1.000 | 0.750 | 1.000 | +0.000 |
| 13 | SROIE | 999 | Spatial gain | 0.000 | 0.000 | 0.500 | 0.500 | +0.500 |
| 14 | SROIE | 979 | Spatial loss | 1.000 | 0.750 | 0.500 | 0.500 | -0.500 |
| 15 | SROIE | 1077 | Spatial–LaOCR disagreement | 0.500 | 0.750 | 0.500 | 0.250 | -0.250 |
| 16 | SROIE | 1035 | Spatial–PP and model disagreement | 0.750 | 0.500 | 0.250 | 0.750 | +0.000 |

The full seven-model scores, unique `document_key`, and source `image_path` for every selected case are in the **generated archive's** `tables/qualitative_16_cases_selection.csv`. The 16 case directories contain the full seven-model predictions and original answer keys, not just these rounded values.

## Text-based illustrations (not yet checked against source images)

- **Case 05, POIE ID 917:** Flat OCR swaps several per-100g and per-serving nutrition values (official score **0.091**). Spatial reconstructs several correct associations (score **0.727**), though it still misassigns some fields. Coordinate JSON scores **0.818**.
- **Case 06, POIE ID 1169:** Spatial assigns several protein, fat, carbohydrate, and sodium values to incorrect fields (**0.313**), below flat OCR (**0.813**); reconstructing the visual order is not uniformly beneficial.
- **Case 09, Receipt ID 1009:** Spatial predicts the accepted total `71.10`, while flat OCR predicts `100.00`; the official scores are **1.000** versus **0.500**.
- **Case 13, SROIE ID 999:** Spatial removes extra date-time and currency formatting from some answer fields, raising its score to **0.500** from flat OCR's **0.000**; other fields remain imperfect.
- **Case 15, SROIE ID 1077:** Spatial scores **0.250**, below coordinate JSON's **0.750**; the raw predictions contain extra suffixes and currency markers.

These are **hypotheses for manual inspection**, based on the saved reference answers and model predictions. Without the original document images, it is premature to claim that the layout reconstruction specifically caused an error.

## Inside the generated archive

```text
ULM_KIE_Full_Analysis_20261030_233817Z/
├── README.md
├── INDEX.html
├── model_runs/                    7 saved model predictions and scored runs
├── tables/
│   ├── qualitative_16_cases_selection.csv
│   ├── qualitative_field_diagnostics.csv
│   ├── seven_model_document_scores.csv
│   └── ...                       overall, subset, efficiency, bootstrap tables
├── figures/                      PNG and PDF figures
├── qualitative_16_cases/
│   ├── case_01_FUNSD_1044/
│   ├── ...
│   └── case_16_SROIE_1035/
└── provenance/
```

Each `case_XX_...` directory contains `case.json`, `review.md`, spatial OCR text (when available) and cached OCR inputs (when available). **No original benchmark images were included in the uploaded archive**; use its `image_path` references and the locally downloaded OCRBench dataset for a visual audit.

A manual case review should check the original image, recognized OCR text, alignment between keys and values, missing or extra fields, schema/format mismatches, and the model output. Classify errors only after this examination. The `review.md` checklists are not completed human annotations.

## Interpretation and reproducibility

- These 16 cases were **chosen using differences in the observed official scores**, so they overrepresent unusual disagreements. They **cannot estimate population error rates, an unbiased accuracy difference, or causal effects**.
- Quantitative claims should use all **400 paired questions**, the official per-item scores, and the 10,000-draw paired stratified bootstrap results.
- The field-level `heuristic_exact_normalized_match` column is a qualitative diagnostic, **not** the official OCRBench metric.
- Spatial OCR reuses LaOCR's saved OCR time; only the new LM step was measured. Component times are not end-to-end wall time.
- The generated complete ZIP also includes raw predictions and reference strings. Verify rights and privacy before publishing it to this **public** repository.

**Code:** [seven-model complete analysis builder](../../scripts/build_complete_seven_model_analysis.py) · [reproduction instructions](../COMPLETE_SEVEN_MODEL_ANALYSIS.md)
