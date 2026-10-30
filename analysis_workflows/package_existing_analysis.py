#!/usr/bin/env python3
"""Bundle existing OCRBench v2 KIE analysis outputs from Helix; never recompute them.

Runs entirely from the current repository checkout. Only files on a fixed allowlist
are included; raw images, OCR caches and model prediction JSON are excluded.
The ZIP contains detailed diagnostic CSVs and should be treated as private/team-only.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "runs" / "analysis"
STAT = ANALYSIS / "kie_scores_0to1"
HYP = ANALYSIS / "hypotheses_400"
FIG = ANALYSIS / "team_figures"


def selected_files():
    """Return explicit archive-name -> local-path pairs, with no directory walks."""
    files = {}
    for filename in (
        "document_scores.csv", "primary_model_ci.csv", "primary_paired_ci.csv",
        "six_model_ci.csv", "six_model_paired_ci.csv", "efficiency.csv",
        "provenance.json",
    ):
        files[f"01_bootstrap_tables/{filename}"] = STAT / filename
    for filename in (
        "kie_400_questions.csv", "kie_400_fields.csv",
        "hypothesis_pairwise_descriptive.csv", "model_error_summary.csv",
        "efficiency_descriptive.csv",
    ):
        files[f"02_diagnostic_tables/{filename}"] = HYP / filename
    for filename in (
        "01_accuracy_bootstrap.png", "02_accuracy_by_dataset.png",
        "03_question_wins_ties_losses.png", "04_accuracy_vs_processing_time.png",
        "05_accuracy_vs_torch_memory.png", "06_token_limit_hits.png",
        "README.txt",
    ):
        files[f"03_team_figures/{filename}"] = FIG / filename
    for filename in ("figure_primary.png", "figure_six_models.png"):
        files[f"03_team_figures/benchmark_corrected/{filename}"] = STAT / filename
    files["04_qualitative/qualitative_case_review.md"] = (
        ANALYSIS / "qualitative_report/qualitative_case_review.md"
    )
    files["04_qualitative/case_notes.csv"] = (
        ROOT / "analysis_workflows/qualitative/case_notes.csv"
    )
    for filename in (
        "README.md", "kie_hypotheses_400.py", "check_helix_inventory.py",
        "create_team_figures.py", "package_existing_analysis.py",
    ):
        files[f"05_reproduction_code/{filename}"] = (
            ROOT / "analysis_workflows" / filename
        )
    for filename in ("README.md", "export_cases.py", "build_case_report.py"):
        files[f"05_reproduction_code/qualitative/{filename}"] = (
            ROOT / "analysis_workflows/qualitative" / filename
        )
    for filename in ("bootstrap_kie.py", "finalize_benchmark.py", "kie_error_analysis.py"):
        files[f"05_reproduction_code/original_analysis_scripts/{filename}"] = (
            ROOT / "scripts" / filename
        )
    return files


def readme():
    return """OCRBench v2 English KIE — complete saved-analysis package

This ZIP was assembled from existing files in the Helix repository checkout.
It contains no new model outputs, OCR processing, rescoring or bootstrap runs.

FOLDERS
  01_bootstrap_tables/  Official score summaries, paired and model bootstrap
                        confidence intervals, efficiency and provenance.
  02_diagnostic_tables/  400-question results, field-level diagnostic records,
                        comparison tables and efficiency diagnostics.
  03_team_figures/      Six publication-draft figures generated from the
                        existing CSVs; benchmark_corrected/ has two figures
                        with 0–1 score axes.
  04_qualitative/       16 manually selected/reviewed cases and Markdown report.
  05_reproduction_code/ Analysis and reporting scripts used with saved results.
  MANIFEST.json        Sources and SHA256 checksums of all included files.

IMPORTANT INTERPRETATION NOTES
  - KIE scores and bootstrap tables are pre-existing official evaluations.
  - Question/field diagnostics are not independently official field scoring.
  - The 16 qualitative examples were purposively selected, not representative.
  - Processing times summarize component timings, not full wall-clock time.
  - Peak Torch GPU memory excludes Paddle OCR allocations.
  - Model bootstrap intervals may be unadjusted; consult paired CI tables
    for inferential model comparisons.

PRIVACY AND REDISTRIBUTION
  PRIVATE / TEAM-ONLY: Detailed diagnostic CSVs and provenance may contain
  benchmark-specific text, predictions, paths, or reference-derived content.
  Review each file and benchmark licensing before sharing externally.
  Do not upload this ZIP or its detailed CSVs to a public GitHub repository.
  Original document images, full OCR caches, scored prediction JSON, and
  full benchmark run archives are intentionally excluded from this package.

The Markdown report can be previewed in VS Code (Cmd+Shift+V on Mac) or
converted into a PDF separately.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ANALYSIS / "packages",
        help="Output directory inside ignored runs/analysis/ (default: runs/analysis/packages)"
    )
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if not output_dir.is_relative_to(ANALYSIS.resolve()):
        parser.error("Output directory must be inside runs/analysis/ (Git-ignored)")

    files = selected_files()
    missing = [(name, str(path.relative_to(ROOT))) for name, path in files.items()
               if not path.is_file()]
    if missing:
        print("Some required analysis outputs are missing. Nothing was written.")
        for archive_name, local_name in missing:
            print(f"  MISSING: {local_name} (for {archive_name})")
        if any(name.startswith("03_team_figures/") for name, _ in missing):
            print("Generate figures once using:")
            print("  .venv-analysis/bin/python analysis_workflows/create_team_figures.py")
            print("If runs/analysis/team_figures/ is already nonempty but incomplete,")
            print("inspect it before generating into a new folder.")
        return 2

    generated = datetime.now(timezone.utc)
    stamp = generated.strftime("%Y%m%d_%H%M%SZ")
    output = output_dir / f"ULM_KIE_Full_Analysis_{stamp}.zip"
    if output.exists():
        parser.error(f"Refusing to overwrite: {output}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at_utc": generated.isoformat(),
        "benchmark": "OCRBench v2 English KIE",
        "evaluated_questions": 400,
        "batch_id": "cbdd58de-531a-4c9c-9778-5bc8b0b0be48",
        "new_inference_or_scores": False,
        "files": [],
    }
    try:
        with ZipFile(output, "x", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr("README_FIRST.txt", readme())
            for name, path in files.items():
                content = path.read_bytes()
                archive.writestr(name, content)
                manifest["files"].append({
                    "archive_path": name,
                    "source_in_repo": str(path.relative_to(ROOT)),
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                })
            archive.writestr("MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
        with ZipFile(output, "r") as archive:
            corrupt = archive.testzip()
            if corrupt:
                raise ValueError(f"Invalid ZIP member: {corrupt}")
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    print(f"Package created: {output}")
    print(f"Included: {len(files)} existing files + README_FIRST.txt + MANIFEST.json")
    print(f"Size: {output.stat().st_size / 1024 / 1024:.2f} MiB")
    print("The ZIP is private/team-only; do NOT add it to public GitHub.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
