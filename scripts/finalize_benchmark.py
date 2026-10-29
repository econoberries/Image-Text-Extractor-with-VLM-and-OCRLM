"""Finalize one newly scored, single-hardware six-model OCRBench v2 KIE batch.

CPU-only. Reads existing six run folders; never resubmits inference or alters scores.
Fails if runs are incomplete, GPU variants differ, or score identities don't align.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from ulm_extractor.storage import read_json

_BOOTSTRAP = importlib.util.spec_from_file_location("bootstrap_kie", ROOT / "scripts" / "bootstrap_kie.py")
bootstrap = importlib.util.module_from_spec(_BOOTSTRAP)
_BOOTSTRAP.loader.exec_module(bootstrap)

PRIMARY = ("qwen35_vlm", "minicpm_vlm", "qwen35_ocr", "qwen25_ocr")
EXTENDED = (*PRIMARY, "qwen35_laocr", "qwen35_ppstructure")
ENGINES = ("paddleocr", "paddleocr_layout", "ppstructurev3")


def sha256(path):
    value = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def inspect_batch(batch_path):
    batch_path = bootstrap.as_batch_path(batch_path)
    batch = read_json(batch_path)
    specs = batch["runs"]
    ids = [item["id"] for item in specs]
    if len(ids) != 6 or set(ids) != set(EXTENDED):
        raise ValueError("Require one fresh batch containing exactly the six configured model IDs")
    if batch.get("gpu_family") != "A100":
        raise ValueError("This finalizer requires a controlled A100 batch")
    if int(batch.get("min_gpu_memory_gib", 0)) < 70 or "gpumem_per_gpu:" not in batch.get("gpu_gres", ""):
        raise ValueError("Batch did not require A100 80GB-class GPU allocation")
    root = batch_path.parents[2]
    runs = {}
    for model in specs:
        name = model["id"]
        folder = root / model["run_id"]
        state = read_json(folder / "run.json")
        if state.get("run_id") not in (None, model["run_id"]) or state.get("batch_id") != batch["batch_id"]:
            raise ValueError(f"{name}: run identity does not match batch")
        if state.get("manifest_sha256") not in (None, batch["manifest_sha256"]):
            raise ValueError(f"{name}: run document manifest differs from batch")
        if state.get("status") != "scored":
            raise ValueError(f"{name}: run not scored (status={state.get('status')})")
        metadata = read_json(folder / "metadata.json")
        if metadata.get("config") not in (None, model):
            raise ValueError(f"{name}: saved model configuration differs from batch")
        if metadata.get("manifest_sha256") not in (None, batch["manifest_sha256"]):
            raise ValueError(f"{name}: model metadata document manifest differs from batch")
        names = metadata.get("gpu_names") or []
        memory = metadata.get("gpu_memory_gib") or []
        if len(names) != 1 or "A100" not in names[0] or "80GB" not in names[0]:
            raise ValueError(f"{name}: wrong GPU variant {names}; must use A100 80GB")
        if len(memory) != 1 or float(memory[0]) < 70:
            raise ValueError(f"{name}: missing or inadequate VRAM metadata {memory}")
        metric_file = folder / "metrics.json"
        metrics = read_json(metric_file)
        if metrics.get("status") != "scored" or int(metrics.get("completed_documents", -1)) != 400:
            raise ValueError(f"{name}: metrics do not attest to 400 scored items")
        scored = folder / "predictions_scored.json"
        runs[name] = {
            "experiment_id": name, "run_id": model["run_id"],
            "batch_id": batch["batch_id"],
            "pipeline": model["pipeline"], "ocr_engine": model.get("ocr"),
            "model_checkpoint": model["checkpoint"], "model_revision": model["revision"],
            "gpu_name": names[0], "gpu_memory_gib": float(memory[0]),
            "torch_version": metadata.get("torch_version"),
            "transformers_version": metadata.get("transformers_version"),
            "inference_dtype": metadata.get("dtype"),
            "scored_sha256": sha256(scored), "metric_sha256": sha256(metric_file),
            "official_kie_f1": metrics["official_kie_f1"],
            "mean_processing_s": metrics.get("mean_latency_s"),
            "median_processing_s": metrics.get("median_latency_s"),
            "p95_processing_s": metrics.get("p95_latency_s"),
            "mean_ocr_s": metrics.get("mean_ocr_time_s"),
            "mean_lm_s": metrics.get("mean_lm_time_s"),
            "model_load_s": metrics.get("model_load_time_s"),
            "peak_torch_allocated_gib": metrics.get("peak_torch_memory_gib"),
            "mean_output_tokens": metrics.get("mean_output_tokens"),
            "token_limit_hits": metrics.get("token_limit_hits"),
        }
    ocr_hardware = {}
    for engine in ENGINES:
        cache_path = batch_path.parent / "ocr" / engine / "index.json"
        index = read_json(cache_path)
        if index.get("manifest_sha256") != batch["manifest_sha256"] or len(index["records"]) != 400:
            raise ValueError(f"{engine}: OCR cache incomplete or manifest mismatch")
        hw = index.get("gpu_hardware") or {}
        if "A100" not in hw.get("name", "") or "80GB" not in hw.get("name", "") or hw.get("memory_mib", 0) < 70*1024:
            raise ValueError(f"{engine}: preprocessing was not verified on A100 80GB: {hw}")
        ocr_hardware[engine] = hw
    return batch_path, batch, runs, ocr_hardware


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No records to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(path, models, pairs, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scores = [r for r in models if r["dataset"] == "ALL"]
    vals = np.array([r["official_mean_score"] for r in scores])
    lo = np.array([r["ci_low"] for r in scores])
    hi = np.array([r["ci_high"] for r in scores])
    fig, ax = plt.subplots(figsize=(10, max(4, len(scores)*0.65+1)))
    ax.errorbar(vals, np.arange(len(scores)), xerr=[vals-lo, hi-vals], fmt="o", capsize=3)
    ax.set_yticks(np.arange(len(scores)), [r["experiment_id"] for r in scores])
    ax.invert_yaxis()
    ax.set_xlabel("Mean KIE score (0–1)")
    ax.set_title(f"{label}: 95% paired-stratified bootstrap intervals")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def finalize(batch_path, output=None, draws=10000, seed=2026, figures=True):
    batch_path, batch, runs, ocr_hardware = inspect_batch(batch_path)
    ids, docs, scores, details, manifest = bootstrap.collect_models([batch_path], EXTENDED)
    if manifest != batch["manifest_sha256"]:
        raise ValueError("Manifest mismatch")
    root = batch_path.parents[2]
    out = Path(output).expanduser().resolve() if output else root / "analysis" / batch["batch_id"]
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing final analysis: {out}")
    # Calculate both panels before any output to avoid partial tables on errors.
    primary_models, primary_pairs, _ = bootstrap.make_results(
        ids, docs, scores, PRIMARY, draws=draws, seed=seed, confidence=0.95,
        stratified=True)
    six_models, six_pairs, aligned = bootstrap.make_results(
        ids, docs, scores, EXTENDED, draws=draws, seed=seed + 1, confidence=0.95,
        stratified=True)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "primary_model_ci.csv", primary_models)
    write_csv(out / "primary_paired_ci.csv", primary_pairs)
    write_csv(out / "six_model_ci.csv", six_models)
    write_csv(out / "six_model_paired_ci.csv", six_pairs)
    write_csv(out / "document_scores.csv", aligned)
    write_csv(out / "efficiency.csv", [runs[k] for k in EXTENDED])
    (out / "provenance.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": "paired dataset-stratified percentile bootstrap; fixed FUNSD/POIE/Receipt/SROIE sizes",
        "resamples": draws, "primary_seed": seed, "six_model_seed": seed + 1,
        "confidence": 0.95,
        "familywise_correction": "Bonferroni within each reported subset and model-pair panel",
        "batch_id": batch["batch_id"], "batch_file": str(batch_path),
        "source_git_commit": batch.get("source_git_commit"),
        "batch_file_sha256": sha256(batch_path),
        "dataset_manifest_sha256": manifest,
        "gpu_gres": batch["gpu_gres"],
        "minimum_gpu_memory_gib": batch["min_gpu_memory_gib"],
        "gpu_ocr_hardware": ocr_hardware,
        "models": [runs[k] for k in EXTENDED],
        "important_notes": [
            "400 observed benchmark items; resampling does not create new test examples.",
            "Model and OCR jobs were separated. Mean processing time sums components; not end-to-end job walltime.",
            "Peak Torch allocated memory omits separate PaddlePaddle OCR GPU memory.",
            "PP recovered Markdown is the only PP-Structure implementation in this six-model batch.",
        ],
    }, indent=2) + "\n", encoding="utf-8")
    if figures:
        make_figure(out / "figure_primary.png", primary_models, primary_pairs, "Original four")
        make_figure(out / "figure_six_models.png", six_models, six_pairs, "All six")
    print(f"Validated six A100 80GB-model runs and three A100 80GB OCR caches.")
    print(f"Finished {draws} paired bootstrap draws per panel. Results: {out}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, required=True, help="Fresh six-model runs/batches/<id>/batch.json")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)
    try:
        finalize(args.batch, args.output_dir, args.n_bootstrap, args.seed, not args.no_figures)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        p.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
