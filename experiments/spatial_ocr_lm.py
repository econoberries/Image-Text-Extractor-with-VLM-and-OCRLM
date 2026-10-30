#!/usr/bin/env python3
"""Seventh, isolated experiment: bbox-arranged PaddleOCR text -> existing Qwen3.5-9B.

prepare (CPU): reuse the original 400-item layout OCR cache, no OCR inference.
infer   (GPU): run ONLY Qwen3.5-9B on spatially reconstructed plain text.
score   (CPU): official OCRBench evaluator + paired comparison with saved OCR-LM runs.

Original batches, model configs, predictions and scores are never modified.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from ulm_extractor.storage import key_for, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "qwen35_spatial_ocr"
COMPARATORS = ("qwen35_ocr", "qwen35_laocr", "qwen35_ppstructure")
NEW_MODELS = (EXPERIMENT_ID, *COMPARATORS)
FORMAT = "bbox-spatial-text-v1"


def get_paths(batch_path):
    batch_path = batch_path.expanduser().resolve()
    if batch_path.is_dir():
        batch_path = batch_path / "batch.json"
    if not batch_path.is_file():
        raise FileNotFoundError(batch_path)
    batch = read_json(batch_path)
    if batch_path.parent.name != batch.get("batch_id"):
        raise ValueError("Batch file and batch_id disagree")
    folder = ROOT / "runs/analysis" / f"spatial_ocr_lm_{batch['batch_id']}"
    if not folder.resolve().is_relative_to((ROOT / "runs/analysis").resolve()):
        raise ValueError("Unsafe experiment directory")
    docs = read_json(batch_path.parent / "documents.json")
    if len(docs) != 400 or len({key_for(x) for x in docs}) != 400:
        raise ValueError("Expected 400 unique documents")
    if not any(m["id"] == "qwen35_ocr" for m in batch["runs"]):
        raise ValueError("The original batch must include qwen35_ocr")
    return batch, batch_path, docs, folder


def as_file(raw):
    path = Path(raw)
    if not path.is_absolute() and not path.is_file():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def arrange_text(lines):
    """Match arrange_text() in the provided ocr_output.py.

    Vertical-center row grouping: 0.5 * mean box height.
    Horizontal character width: 0.5 * mean box height.
    Pixel gaps are mapped to spaces using the current estimated text width.
    All detections must have xyxy boxes: never silently drop recognized text.
    """
    items = []
    for line in lines:
        box = line.get("bbox")
        if not box or len(box) != 4:
            raise ValueError("Missing bbox in saved LaOCR: refusing to drop OCR text")
        x1, y1, x2, y2 = map(float, box)
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            raise ValueError("Nonfinite OCR box")
        if x2 < x1 or y2 < y1:
            raise ValueError("Invalid OCR box")
        items.append({
            "text": str(line["text"]).strip(),
            "x": x1,
            "y": (y1 + y2) / 2,
            "height": y2 - y1,
        })
    if not items:
        return ""
    min_x = min(item["x"] for item in items)
    avg_height = sum(item["height"] for item in items) / len(items)
    y_tolerance = max(1, avg_height * 0.5)
    char_width = max(1, avg_height * 0.5)

    items.sort(key=lambda item: item["y"])
    rows = []
    for item in items:
        group = next(
            (row for row in reversed(rows)
             if abs(item["y"] - row["y"]) <= y_tolerance),
            None,
        )
        if group is None:
            rows.append({"y": item["y"], "items": [item]})
        else:
            group["items"].append(item)
            group["y"] = sum(x["y"] for x in group["items"]) / len(group["items"])

    result = []
    for row in rows:
        row["items"].sort(key=lambda item: item["x"])
        content = ""
        previous_x = min_x
        for item in row["items"]:
            spaces = max(0, round((item["x"] - previous_x) / char_width))
            content += " " * spaces + item["text"]
            previous_x = item["x"] + len(item["text"]) * char_width
        result.append(content.rstrip())
    return "\n".join(result)


def prepare(batch, batch_path, docs, folder):
    if folder.exists():
        raise FileExistsError(
            f"Experiment folder already exists; refusing to overwrite: {folder}"
        )
    layout = read_json(batch_path.parent / "ocr/paddleocr_layout/index.json")
    flat = read_json(batch_path.parent / "ocr/paddleocr/index.json")
    for label, index in (("LaOCR", layout), ("plain OCR", flat)):
        if index.get("manifest_sha256") != batch["manifest_sha256"]:
            raise ValueError(f"{label} OCR manifest differs from the six-model batch")
        if len(index["records"]) != 400:
            raise ValueError(f"{label} cache is not complete (400 expected)")
    if layout.get("format") != "paddleocr-layout-v1":
        raise ValueError("Layout cache format differs from recorded source")
    # Validate the entire input before writing anything.
    prepared = {}
    for doc in docs:
        key = key_for(doc)
        lj = read_json(as_file(layout["records"][key]["text_path"]))
        if lj.get("format") != "paddleocr-layout-v1":
            raise ValueError(f"Unexpected LaOCR JSON format for {key}")
        lines = lj["lines"]
        flat_text = as_file(flat["records"][key]["text_path"]).read_text(encoding="utf-8")
        recognized = "\n".join(str(line["text"]) for line in lines)
        if recognized != flat_text:
            raise ValueError(
                f"Flat and layout OCR recognized text differs for {key}; "
                "cannot isolate input representation on this document"
            )
        started = time.perf_counter()
        text = arrange_text(lines)
        seconds = time.perf_counter() - started
        if not text.strip():
            raise ValueError(f"No spatial text for {key}")
        prepared[key] = {
            "content": text,
            "format_seconds": seconds,
            "layout_ocr_seconds": float(layout["records"][key]["ocr_seconds"]),
            "n_lines": len(lines),
        }
    folder.mkdir(parents=True)
    cache = {
        "experiment_id": EXPERIMENT_ID,
        "representation": FORMAT,
        "original_batch_id": batch["batch_id"],
        "manifest_sha256": batch["manifest_sha256"],
        "layout_cache": str(batch_path.parent / "ocr/paddleocr_layout/index.json"),
        "flat_cache": str(batch_path.parent / "ocr/paddleocr/index.json"),
        "comparability_check": "exact recognition text and order matched plain OCR for every question",
        "records": {},
    }
    for key, row in prepared.items():
        text_file = folder / "spatial_text" / f"{key}.txt"
        text_file.parent.mkdir(parents=True, exist_ok=True)
        text_file.write_text(row["content"], encoding="utf-8")
        cache["records"][key] = {
            "text_path": str(text_file),
            "text_sha256": hashlib.sha256(row["content"].encode("utf-8")).hexdigest(),
            "layout_ocr_seconds": row["layout_ocr_seconds"],
            "format_seconds": row["format_seconds"],
            "n_lines": row["n_lines"],
        }
    write_json(folder / "spatial_cache.json", cache)
    print(f"Prepared {len(docs)} spatial OCR texts in {folder}")
    print("No original OCR or model results were modified.")


def infer(batch, docs, folder):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU inference must run inside a Slurm allocation, not a login node")
    from ulm_extractor.adapters import InferenceAdapter
    from ulm_extractor.gpu_hardware import gpu_hardware

    gpu = gpu_hardware()
    if "A100" not in gpu["name"] or gpu["memory_mib"] < 70 * 1024:
        raise RuntimeError(f"Require same A100 80GB class as the original batch: {gpu}")
    cache = read_json(folder / "spatial_cache.json")
    if cache["manifest_sha256"] != batch["manifest_sha256"] or len(cache["records"]) != 400:
        raise ValueError("Spatial OCR cache incomplete or from another batch")
    old_spec = next(m for m in batch["runs"] if m["id"] == "qwen35_ocr")
    if (old_spec["adapter"], old_spec["pipeline"], old_spec["ocr"]) != (
        "qwen35", "ocr_lm", "paddleocr"
    ):
        raise ValueError("Unexpected Qwen3.5 baseline configuration")
    spec = {k: v for k, v in old_spec.items() if k != "run_id"}
    spec["id"] = EXPERIMENT_ID
    # Keep pipeline='ocr_lm': the unchanged adapter accepts spatial plain text.
    model = InferenceAdapter(spec)
    output_file = folder / "predictions.json"
    prior = read_json(output_file) if output_file.is_file() else []
    saved = {key_for(r): r for r in prior}
    valid_keys = {key_for(d) for d in docs}
    if len(saved) != len(prior) or set(saved) - valid_keys:
        raise ValueError("Existing partial predictions contain duplicates or wrong documents")
    for row in prior:
        if row.get("error") or not isinstance(row.get("predict"), str):
            raise ValueError("Existing predictions include failures; inspect before resuming")
    pending = [doc for doc in docs if key_for(doc) not in saved]
    if not pending:
        print("All 400 spatial model predictions already exist; no GPU rerun.")
        return
    model.load()
    meta = {
        "experiment_id": EXPERIMENT_ID,
        "representation": FORMAT,
        "original_batch_id": batch["batch_id"],
        "manifest_sha256": batch["manifest_sha256"],
        "baseline_model_config": old_spec,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "gpu_hardware": gpu,
        **model.environment(),
    }
    if (folder / "metadata.json").is_file():
        original = read_json(folder / "metadata.json")
        if original["manifest_sha256"] != batch["manifest_sha256"]:
            raise ValueError("Prior metadata has different data manifest")
    else:
        write_json(folder / "metadata.json", meta)
    for pos, doc in enumerate(pending, 1):
        key = key_for(doc)
        entry = dict(doc)
        entry["document_key"] = key
        try:
            record = cache["records"][key]
            content = as_file(record["text_path"]).read_text(encoding="utf-8")
            if hashlib.sha256(content.encode("utf-8")).hexdigest() != record["text_sha256"]:
                raise ValueError(f"Spatial OCR content changed: {key}")
            outcome = model.infer(doc["question"], ocr_text=content)
            entry.update(outcome)
            entry["ocr_seconds"] = round(
                record["layout_ocr_seconds"] + record["format_seconds"], 4
            )
            entry["spatial_format_seconds"] = round(record["format_seconds"], 6)
            entry["lm_seconds"] = outcome["lm_seconds"]
            entry["inference_seconds"] = round(
                entry["ocr_seconds"] + outcome["lm_seconds"], 4
            )
        except Exception as exc:
            write_json(folder / "inference_failure.json", {
                "document_key": key, "error": f"{type(exc).__name__}: {exc}"
            })
            raise
        saved[key] = entry
        write_json(output_file, [saved[key_for(d)] for d in docs if key_for(d) in saved])
        print(f"[{pos}/{len(pending)}] Saved {key}", flush=True)
    print(f"Finished spatial OCR–LM for {len(saved)} questions: {output_file}")
    print("OCR stage timing is reused from saved LaOCR; only new LM time is measured.")


def score(batch, batch_path, docs, folder, draws):
    output_file = folder / "predictions.json"
    scored_file = folder / "predictions_scored.json"
    if not output_file.is_file():
        raise FileNotFoundError("Run infer on an A100 GPU before official scoring")
    predicted = read_json(output_file)
    if len(predicted) != 400 or {key_for(x) for x in predicted} != {key_for(x) for x in docs}:
        raise ValueError("Expected exactly 400 matching spatial predictions")
    if any(x.get("error") or not isinstance(x.get("predict"), str) for x in predicted):
        raise ValueError("One or more model predictions failed")
    if not scored_file.is_file():
        evaluator = as_file(batch["evaluator"])
        temporary = folder / "predictions_scored.pending.json"
        if temporary.exists():
            raise FileExistsError(f"Previous incomplete scoring file: {temporary}")
        subprocess.run(
            [sys.executable, str(evaluator), "--input_path", str(output_file),
             "--output_path", str(temporary)],
            check=True,
        )
        computed = read_json(temporary)
        if len(computed) != 400 or any(
            type(x.get("score")) not in (float, int) or
            not math.isfinite(x["score"]) or not 0 <= x["score"] <= 1
            for x in computed
        ):
            raise ValueError(f"Official evaluator returned invalid scores: {temporary}")
        temporary.rename(scored_file)

    comparison_dir = folder / "comparison"
    if comparison_dir.exists():
        raise FileExistsError("Comparison already exists; refusing to overwrite")
    bootstrap_file = ROOT / "scripts/bootstrap_kie.py"
    spec = importlib.util.spec_from_file_location("original_bootstrap_kie", bootstrap_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    expected = {module.identity(x) for x in docs}
    aligned_ids = [module.identity(x) for x in docs]
    sources = {EXPERIMENT_ID: scored_file}
    for name in COMPARATORS:
        original = next(m for m in batch["runs"] if m["id"] == name)
        sources[name] = (batch_path.parents[2] /
                         original["run_id"] / "predictions_scored.json")
    import numpy as np
    scores = {}
    for name, path in sources.items():
        scored = module.read_scored(path, expected)
        scores[name] = np.asarray([scored[key] for key in aligned_ids], dtype=float)
    models, pairs, aligned = module.make_results(
        aligned_ids, docs, scores, NEW_MODELS,
        draws=draws, seed=2026, confidence=0.95, stratified=True,
    )
    comparison_dir.mkdir()
    module.write_csv(comparison_dir / "model_bootstrap_ci.csv", models)
    module.write_csv(comparison_dir / "paired_bootstrap_ci.csv", pairs)
    module.write_csv(comparison_dir / "paired_document_scores.csv", aligned)
    descriptive = []
    for other in COMPARATORS:
        new = scores[EXPERIMENT_ID]
        old = scores[other]
        descriptive.append({
            "comparison": f"{EXPERIMENT_ID} - {other}",
            "spatial_mean": float(new.mean()),
            "other_mean": float(old.mean()),
            "mean_difference": float((new - old).mean()),
            "spatial_higher": int((new > old).sum()),
            "tie": int((new == old).sum()),
            "other_higher": int((new < old).sum()),
        })
    with (comparison_dir / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(descriptive[0]))
        writer.writeheader()
        writer.writerows(descriptive)
    write_json(comparison_dir / "methodology.json", {
        "new_experiment": EXPERIMENT_ID,
        "input": FORMAT,
        "source_batch": batch["batch_id"],
        "dataset_manifest": batch["manifest_sha256"],
        "source_predictions": {k: str(v) for k, v in sources.items()},
        "official_evaluator": batch["evaluator"],
        "bootstrap_draws": draws,
        "bootstrap_seed": 2026,
        "paired_comparison_models": list(NEW_MODELS),
        "familywise_correction": "Bonferroni across six comparisons of four OCR-LM pipelines",
        "timing_warning": "OCR time reused from saved LaOCR cache; new LM time measured in this run; not end-to-end wall time",
        "gpu_memory_warning": "Torch-only memory; does not include PaddleOCR",
    })
    print(f"Official mean KIE score: {scores[EXPERIMENT_ID].mean():.4f}")
    for row in descriptive:
        print(row)
    print(f"Saved official scores and paired comparison: {comparison_dir}")
    print("Original six scored runs and bootstrap results were not modified.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "infer", "score"))
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if args.n_bootstrap < 100:
        parser.error("--n-bootstrap must be at least 100")
    batch, batch_path, docs, folder = get_paths(args.batch)
    if args.action == "prepare":
        prepare(batch, batch_path, docs, folder)
    elif args.action == "infer":
        infer(batch, docs, folder)
    else:
        score(batch, batch_path, docs, folder, args.n_bootstrap)


if __name__ == "__main__":
    main()
