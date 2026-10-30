# Reproducible qualitative KIE case analysis

This folder preserves the **16 selected document cases** from the six-system OCRBench v2 English KIE evaluation (400 questions). It is an **exploratory qualitative review** of contrasting errors, not a random sample or an estimate of mechanism prevalence.

## Files

- **case_notes.csv** — the frozen 16-case selection, comparison group, and manually reviewed observations. This is the public, derived research record; interpretations are not automatically verified causal labels.
- **export_cases.py** — reconstructs each selected case using the **existing** scored predictions and OCR caches; optionally copies the original document image for private inspection.
- **build_case_report.py** — creates a formatted Markdown summary from case_notes.csv and the saved 400-question score table. Uses original official scores, no rescoring.

## On Helix

Run from the repository root after updating the analysis branch. You already have the original case folders at runs/analysis/layout_case_review_extended/; **there is no need to regenerate them**.

To create a readable report:

~~~bash
.venv-analysis/bin/python analysis_workflows/qualitative/build_case_report.py
~~~

Output: runs/analysis/qualitative_report/qualitative_case_review.md

To reconstruct the detailed evidence separately (optional):

~~~bash
.venv-analysis/bin/python analysis_workflows/qualitative/export_cases.py --include-images
~~~

Output: runs/analysis/layout_case_review_reproduced/

Omit --include-images to export only saved OCR payloads, reference files and predictions. The script refuses to overwrite any nonempty output folder. If an output already exists and you want another reproduction, pass a **new** --output-dir within runs/analysis/.

Neither script calls model inference, OCR preprocessing, the official evaluator, or the bootstrap.

## Selection rationale

Selected cases come from four score-disagreement categories relative to standard Qwen3.5 OCR–LM:

1. LaOCR performed better.
2. Standard OCR performed better than LaOCR.
3. Standard OCR performed better than PP-StructureV3.
4. PP-StructureV3 performed better than standard OCR.

We deliberately selected four examples in each category across POIE, FUNSD and SROIE. The original selection was informed by the ranked question-level score differences; case_notes.csv records the final case identities and manually documented observations.

Some examples can illustrate more than one mechanism. LaOCR and standard OCR used the same recognized words for all 400 questions, but differed in output representation. PP-StructureV3 has a different recognition/recovery pipeline, so its differences cannot be attributed to layout encoding alone.

## Privacy, licensing, and reproducibility

- **Do not commit or publicly upload** benchmark images, OCR text payloads, benchmark questions/answers, raw model predictions or archived results.
- Evidence and generated reports are saved under runs/analysis/, which the project's .gitignore excludes.
- Only the derived selection, manually reviewed observations, and analysis code live in GitHub.
- Review benchmark redistribution terms before privately sharing any source image or annotation.
- The generated report is Markdown and can be viewed through VS Code Markdown Preview.
