#!/usr/bin/env python3
"""Read-only inventory of completed benchmark, statistical, and case-review files."""
import csv
import json
from pathlib import Path

BATCH_ID = "cbdd58de-531a-4c9c-9778-5bc8b0b0be48"
MODELS = {
    "qwen35_vlm", "minicpm_vlm", "qwen35_ocr",
    "qwen25_ocr", "qwen35_laocr", "qwen35_ppstructure",
}
ENGINES = ("paddleocr", "paddleocr_layout", "ppstructurev3")
EXPECTED_CSVS = {
    "kie_400_questions.csv": 400,
    "kie_400_fields.csv": 2831,
    "hypothesis_pairwise_descriptive.csv": 20,
    "model_error_summary.csv": 30,
    "efficiency_descriptive.csv": 6,
}
BOOTSTRAP_FILES = (
    "document_scores.csv", "primary_model_ci.csv", "primary_paired_ci.csv",
    "six_model_ci.csv", "six_model_paired_ci.csv", "efficiency.csv",
    "provenance.json", "figure_primary.png", "figure_six_models.png",
)

root = Path(__file__).resolve().parents[1]
runs = root / "runs"
batch_dir = runs / "batches" / BATCH_ID
errors = []

def required(path):
    if path.is_file():
        print("OK     ", path.relative_to(root))
    else:
        errors.append(str(path.relative_to(root)))
        print("MISSING", path.relative_to(root))
    return path.is_file()

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

print("Repository:", root)
batch_file = batch_dir / "batch.json"
documents_file = batch_dir / "documents.json"
if required(batch_file):
    batch = read(batch_file)
    specs = {m["id"]: m for m in batch["runs"]}
    if set(specs) != MODELS:
        errors.append("batch has incorrect set of six model IDs")
    for model in sorted(MODELS & set(specs)):
        run_dir = runs / specs[model]["run_id"]
        pred = run_dir / "predictions_scored.json"
        if required(pred):
            rows = read(pred)
            if len(rows) != 400 or any(not isinstance(r.get("score"), (int, float)) for r in rows):
                errors.append(f"{model}: predictions do not have 400 numeric scores")
        required(run_dir / "metrics.json")
        required(run_dir / "metadata.json")

if required(documents_file) and len(read(documents_file)) != 400:
    errors.append("documents.json does not contain 400 questions")

for engine in ENGINES:
    index_path = batch_dir / "ocr" / engine / "index.json"
    if required(index_path):
        idx = read(index_path)
        records = idx.get("records", {})
        print(f"  {engine}: {len(records)} cached documents")
        if len(records) != 400:
            errors.append(f"{engine}: expected 400 cached documents")
        missing = sum(
            not (root / entry["text_path"]).is_file()
            for entry in records.values()
            if isinstance(entry.get("text_path"), str)
        )
        if missing:
            errors.append(f"{engine}: {missing} cached text files not found")

stat = runs / "analysis" / "kie_scores_0to1"
for name in BOOTSTRAP_FILES:
    required(stat / name)

diagnostics = runs / "analysis" / "hypotheses_400"
for name, expected in EXPECTED_CSVS.items():
    path = diagnostics / name
    if required(path):
        with path.open(encoding="utf-8", newline="") as f:
            count = sum(1 for _ in csv.DictReader(f))
        print(f"  rows: {count} (expected {expected})")
        if count != expected:
            errors.append(f"{name}: {count} rows, expected {expected}")

case_dir = runs / "analysis" / "layout_case_review_extended"
case_folders = sorted(p for p in case_dir.glob("*") if p.is_dir()) if case_dir.is_dir() else []
print("Qualitative case folders:", len(case_folders), "(expected 16)")
if len(case_folders) != 16:
    errors.append("Expected 16 qualitative case folders")
for folder in case_folders:
    for name in (
        "reference.json", "plain_ocr.txt", "layout_ocr.json",
        "ppstructure.md", "qwen35_vlm.json", "qwen35_ocr.json",
        "qwen35_laocr.json", "qwen35_ppstructure.json", "qwen25_ocr.json",
    ):
        required(folder / name)

source_script = runs / "analysis" / "kie_hypotheses_400.py"
required(source_script)

print("\n", "COMPLETE" if not errors else f"INCOMPLETE: {len(errors)} issue(s)")
if errors:
    for e in errors:
        print(" -", e)
    raise SystemExit(1)
