#!/usr/bin/env python3
"""Build a self-contained seven-model OCRBench KIE analysis from ALREADY SCORED runs.

CPU-only: no OCR, no model inference, no Slurm submission. Does not modify source runs.
Includes a purposefully selected 16-case qualitative review, not a random sample.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import bootstrap_kie as boot
import kie_error_analysis as diagnostics
from ulm_extractor.storage import key_for

SIX = (
    "qwen35_vlm", "minicpm_vlm", "qwen35_ocr", "qwen25_ocr",
    "qwen35_laocr", "qwen35_ppstructure",
)
SPATIAL = "qwen35_spatial_ocr"
MODELS = (*SIX, SPATIAL)
DATASETS = ("FUNSD", "POIE", "Receipt", "SROIE")


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows available for {path}")
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_runs(batch_path: Path):
    """Validate every original run and the spatial run against the SAME 400 identities."""
    batch_path = boot.as_batch_path(batch_path)
    batch = read_json(batch_path)
    identities, docs, arrays, details, manifest = boot.collect_models([batch_path], SIX)
    if manifest != batch["manifest_sha256"]:
        raise ValueError("Original six-model manifest mismatch")
    run_root = batch_path.parents[2]
    spec = {x["id"]: x for x in batch["runs"]}
    if set(spec) != set(SIX):
        raise ValueError(f"Expected six specific original model IDs, found {sorted(spec)}")
    folders = {m: run_root / spec[m]["run_id"] for m in SIX}
    spatial_dir = run_root / "analysis" / f"spatial_ocr_lm_{batch['batch_id']}"
    cache_path = spatial_dir / "spatial_cache.json"
    meta_path = spatial_dir / "metadata.json"
    if not cache_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError("Missing spatial cache or metadata; finish the seventh experiment first")
    cache, meta = read_json(cache_path), read_json(meta_path)
    for label, entry in (("spatial_cache", cache), ("spatial_metadata", meta)):
        if entry.get("manifest_sha256") != manifest:
            raise ValueError(f"{label} manifest is not identical to original six runs")
    if cache.get("original_batch_id") != batch["batch_id"]:
        raise ValueError("Spatial cache references another batch")
    expected = set(identities)
    scored_path = spatial_dir / "predictions_scored.json"
    spatial_scores = boot.read_scored(scored_path, expected)
    arrays[SPATIAL] = np.asarray([spatial_scores[k] for k in identities], dtype=float)
    folders[SPATIAL] = spatial_dir
    # Keep full official rows keyed by the exact OCRBench document identity.
    row_maps = {}
    for model, folder in folders.items():
        rows = read_json(folder / "predictions_scored.json")
        indexed = {boot.identity(row): row for row in rows}
        if len(rows) != 400 or set(indexed) != expected:
            raise ValueError(f"Scored row identities do not match: {model}")
        row_maps[model] = indexed
    return batch_path, batch, identities, docs, arrays, folders, row_maps, details


def cases_from_file(path: Path, docs: list[dict]):
    """Accept CSV/JSON/TXT with exactly 16 document_key or dataset + OCRBench id entries."""
    if path.suffix.lower() == ".json":
        items = read_json(path)
        if isinstance(items, dict):
            items = items.get("cases", items.get("selected_cases", []))
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            items = list(csv.DictReader(f))
    else:
        items = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not isinstance(items, list) or len(items) != 16:
        raise ValueError("Case selection file must contain exactly 16 entries")
    selected, used = [], set()
    for item in items:
        if isinstance(item, dict):
            key = str(item.get("document_key", item.get("key", ""))).strip()
            ident = str(item.get("id", "")).strip()
            dataset = str(item.get("dataset_name", item.get("dataset", ""))).strip()
        else:
            key, ident, dataset = str(item).strip(), str(item).strip(), ""
        matches = [
            i for i, doc in enumerate(docs)
            if (key and key_for(doc) == key)
            or (ident and str(doc["id"]) == ident
                and (not dataset or str(doc["dataset_name"]).casefold() == dataset.casefold()))
        ]
        if len(matches) != 1 or matches[0] in used:
            raise ValueError(f"Ambiguous, duplicate or missing case identity: {item!r}")
        selected.append((matches[0], "user_supplied_previous_case"))
        used.add(matches[0])
    return selected


def choose_cases(docs: list[dict], scores: dict, selection_file: Path | None):
    if selection_file:
        return cases_from_file(selection_file, docs)
    # Purposive, deterministic: 4 per dataset = 16. Do not interpret this as sampling.
    chosen = []
    vectors = {m: np.asarray(v) for m, v in scores.items()}
    for dataset in DATASETS:
        candidates = [i for i, x in enumerate(docs) if str(x["dataset_name"]).casefold() == dataset.casefold()]
        if not candidates:
            raise ValueError(f"Dataset {dataset} has no records")
        spatial = vectors[SPATIAL]
        flat = vectors["qwen35_ocr"]
        layout = vectors["qwen35_laocr"]
        pp = vectors["qwen35_ppstructure"]
        baselines = np.vstack([vectors[m] for m in SIX])
        priorities = [
            ("spatial_gain_vs_flat", lambda i: spatial[i] - flat[i]),
            ("spatial_loss_vs_flat", lambda i: flat[i] - spatial[i]),
            ("spatial_vs_layout_divergence", lambda i: abs(spatial[i] - layout[i])),
            ("spatial_vs_pp_and_model_divergence",
             lambda i: abs(spatial[i] - pp[i]) + float(np.std(baselines[:, i]))),
        ]
        seen = set()
        for label, criterion in priorities:
            ranks = sorted(candidates, key=lambda i: (-float(criterion(i)), str(docs[i]["id"]), i))
            pick = next((i for i in ranks if i not in seen), None)
            if pick is None:
                raise ValueError(f"Not enough unique cases in {dataset}")
            chosen.append((pick, label))
            seen.add(pick)
    return chosen


def make_figures(folder: Path, docs: list[dict], scores: dict, models_ci: list[dict],
                 pair_ci: list[dict], selected: list[tuple], efficiency: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, name):
        fig.tight_layout()
        for extension in (".png", ".pdf"):
            fig.savefig(folder / f"{name}{extension}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    all_rows = {r["experiment_id"]: r for r in models_ci if r["dataset"] == "ALL"}
    y = np.arange(len(MODELS))
    means = np.array([all_rows[m]["official_mean_score"] for m in MODELS])
    lo = np.array([all_rows[m]["ci_low"] for m in MODELS])
    hi = np.array([all_rows[m]["ci_high"] for m in MODELS])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.errorbar(means, y, xerr=[means - lo, hi - means], fmt="o", capsize=4)
    ax.set_yticks(y, MODELS)
    ax.invert_yaxis()
    ax.set_xlabel("Mean official OCRBench KIE item score (95% bootstrap CI)")
    ax.set_title("Seven-model comparison on 400 matched questions")
    ax.grid(axis="x", alpha=0.25)
    save(fig, "01_seven_model_overall_ci")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(DATASETS))
    width = 0.11
    for j, m in enumerate(MODELS):
        vals = [float(np.mean([scores[m][i] for i, d in enumerate(docs)
                               if str(d["dataset_name"]).casefold() == ds.casefold()]))
                for ds in DATASETS]
        ax.bar(x + (j - 3) * width, vals, width, label=m)
    ax.set_xticks(x, DATASETS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean official KIE item score")
    ax.set_title("All seven models by OCRBench subset")
    ax.legend(ncol=2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, "02_model_scores_by_dataset")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    baselines = ("qwen35_ocr", "qwen35_laocr", "qwen35_ppstructure", "qwen35_vlm")
    differences = [scores[SPATIAL] - scores[m] for m in baselines]
    ax.boxplot(differences, tick_labels=baselines, showmeans=True)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_ylabel("Spatial score minus comparator, per question")
    ax.set_title("Paired document-level score differences")
    ax.tick_params(axis="x", rotation=15)
    save(fig, "03_spatial_paired_differences")

    fig, ax = plt.subplots(figsize=(10, 6))
    difference = np.array([[float(np.mean(scores[a] - scores[b])) for b in MODELS] for a in MODELS])
    bound = max(float(np.max(abs(difference))), 0.01)
    im = ax.imshow(difference, vmin=-bound, vmax=bound, cmap="coolwarm")
    ax.set_xticks(range(len(MODELS)), MODELS, rotation=45, ha="right")
    ax.set_yticks(range(len(MODELS)), MODELS)
    fig.colorbar(im, ax=ax, label="Mean difference (row minus column)")
    ax.set_title("Full seven-model mean-score difference matrix")
    save(fig, "04_pairwise_difference_heatmap")

    fig, ax = plt.subplots(figsize=(12, 6))
    selected_indices = [i for i, _ in selected]
    matrix = np.array([[scores[m][i] for m in MODELS] for i in selected_indices])
    im = ax.imshow(matrix, vmin=0, vmax=1, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(MODELS)), MODELS, rotation=35, ha="right")
    ax.set_yticks(range(16), [f"{j+1:02d} {docs[i]['dataset_name']} #{docs[i]['id']}"
                                  for j, i in enumerate(selected_indices)])
    fig.colorbar(im, ax=ax, label="Official item score")
    ax.set_title("16 purposively selected qualitative cases")
    save(fig, "05_qualitative_case_score_heatmap")

    fig, ax = plt.subplots(figsize=(11, 6))
    for m in MODELS:
        row = next(e for e in efficiency if e["model"] == m)
        t = row["mean_component_time_s"]
        if isinstance(t, (float, int)):
            ax.scatter([t], [row["mean_official_score"]], s=75)
            ax.annotate(m, (t, row["mean_official_score"]), xytext=(5, 5),
                        textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean available component time per question (seconds)")
    ax.set_ylabel("Mean official KIE item score")
    ax.set_title("Accuracy versus measured component time (not end-to-end wall time)")
    ax.grid(alpha=0.2)
    save(fig, "06_accuracy_vs_component_time")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([scores[m] for m in MODELS], tick_labels=MODELS, showmeans=True)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Official item-level KIE score")
    ax.set_title("Distribution of item scores for the seven models")
    ax.tick_params(axis="x", rotation=35)
    save(fig, "07_item_score_distributions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        help="Default: runs/analysis/ULM_KIE_Full_Analysis_<UTC>")
    parser.add_argument("--case-ids-file", type=Path,
                        help="CSV/JSON/TXT containing the same EXACT 16 cases as a previous review")
    parser.add_argument("--existing-review", type=Path,
                        help="Optional prior qualitative folder copied as archival evidence")
    parser.add_argument("--image-root", type=Path,
                        help="Optional OCRBench image root; copies only selected 16 source images")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    if args.n_bootstrap < 100:
        parser.error("At least 100 bootstrap draws are required")
    batch_path, batch, identities, docs, scores, folders, records, details = load_runs(args.batch)
    selected = choose_cases(docs, scores, args.case_ids_file)
    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out = (args.output_dir or ROOT / "runs" / "analysis" / f"ULM_KIE_Full_Analysis_{now}").expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite an existing folder: {out}")
    if args.dry_run:
        print(f"Validated {len(MODELS)} complete scored runs on {len(docs)} matched questions.")
        print(f"Output planned: {out}")
        print("Selected cases:", [(docs[i]["dataset_name"], docs[i]["id"], label) for i, label in selected])
        print("DRY RUN: no files created; no GPU work.")
        return

    out.mkdir(parents=True)
    tables, figures, cases_dir = out / "tables", out / "figures", out / "qualitative_16_cases"
    for folder in (tables, figures, cases_dir, out / "model_runs", out / "provenance"):
        folder.mkdir(parents=True, exist_ok=True)

    models_ci, pair_ci, aligned = boot.make_results(
        identities, docs, scores, MODELS, draws=args.n_bootstrap,
        seed=args.seed, confidence=0.95, stratified=True
    )
    write_csv(tables / "seven_model_bootstrap_ci.csv", models_ci)
    write_csv(tables / "seven_model_paired_bootstrap_ci.csv", pair_ci)
    write_csv(tables / "seven_model_document_scores.csv", aligned)

    overall, subsets, efficiency, model_paths = [], [], [], {}
    for model in MODELS:
        arr = scores[model]
        overall.append({
            "model": model, "documents": 400, "official_mean_score": float(arr.mean()),
            "median_item_score": float(np.median(arr)), "score_std": float(arr.std(ddof=1)),
            "zero_score_count": int(np.sum(arr == 0)),
            "perfect_score_count": int(np.sum(arr == 1)),
        })
        for ds in DATASETS:
            index = [i for i, d in enumerate(docs) if d["dataset_name"].casefold() == ds.casefold()]
            subsets.append({"model": model, "dataset": ds, "documents": len(index),
                            "official_mean_score": float(arr[index].mean())})
        folder = folders[model]
        destination = out / "model_runs" / model
        destination.mkdir(parents=True)
        copied = {}
        for filename in ("predictions.json", "predictions_scored.json", "metrics.json",
                         "metadata.json", "run.json"):
            original = folder / filename
            if original.is_file():
                shutil.copy2(original, destination / filename)
                copied[filename] = {"source": str(original), "sha256": digest(original)}
        if "predictions_scored.json" not in copied:
            raise FileNotFoundError(f"Missing scored source in {folder}")
        model_paths[model] = copied
        durations = [float(r["inference_seconds"]) for r in records[model].values()
                     if type(r.get("inference_seconds")) in (float, int)]
        if not durations:
            durations = [float(r["ocr_seconds"] + r["lm_seconds"])
                         for r in records[model].values()
                         if type(r.get("ocr_seconds")) in (int, float)
                         and type(r.get("lm_seconds")) in (int, float)]
        metrics = read_json(folder / "metrics.json") if (folder / "metrics.json").is_file() else {}
        component_time = metrics.get("mean_latency_s")
        if component_time is None and durations:
            component_time = float(np.mean(durations))
        efficiency.append({
            "model": model, "mean_official_score": float(arr.mean()),
            "mean_component_time_s": component_time,
            "time_source": "saved_metrics" if metrics.get("mean_latency_s") is not None else
                           ("saved_prediction_fields" if durations else "unavailable"),
            "mean_ocr_s": metrics.get("mean_ocr_time_s"),
            "mean_lm_s": metrics.get("mean_lm_time_s"),
            "peak_torch_memory_gib": metrics.get("peak_torch_memory_gib"),
            "timing_note": "Spatial reuses earlier OCR timing; not full end-to-end wall time"
                           if model == SPATIAL else "Components recorded in original run; not job wall time",
        })

    write_csv(tables / "overall_model_summary.csv", overall)
    write_csv(tables / "scores_by_dataset.csv", subsets)
    write_csv(tables / "efficiency.csv", efficiency)

    paired = []
    for other in SIX:
        diffs = scores[SPATIAL] - scores[other]
        paired.append({
            "spatial_minus_model": other,
            "mean_difference": float(diffs.mean()),
            "spatial_wins": int((diffs > 0).sum()),
            "ties": int((diffs == 0).sum()),
            "spatial_losses": int((diffs < 0).sum()),
        })
    write_csv(tables / "spatial_head_to_head.csv", paired)

    selection_rows, field_rows = [], []
    for position, (index, reason) in enumerate(selected, start=1):
        doc, identity = docs[index], identities[index]
        key = key_for(doc)
        safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(doc["id"]))[:40]
        case_folder = cases_dir / f"case_{position:02d}_{doc['dataset_name']}_{safe_id}"
        case_folder.mkdir()
        case_rows = {}
        parsed = {}
        for model in MODELS:
            row = records[model][identity]
            case_rows[model] = {
                "official_score": float(row["score"]),
                "prediction": row.get("predict"),
                "hit_token_limit": row.get("hit_token_limit"),
                "inference_seconds": row.get("inference_seconds"),
                "ocr_seconds": row.get("ocr_seconds"),
                "lm_seconds": row.get("lm_seconds"),
            }
            parsed[model] = diagnostics.parse_prediction(row.get("predict"))
        try:
            reference = diagnostics.reference_values(doc)
        except Exception as exc:
            raise ValueError(f"Could not parse gold fields for case {key}: {exc}") from exc
        all_fields = list(reference) + sorted(set().union(
            *(set(x) for x in parsed.values())) - set(reference))
        for field in all_fields:
            vals = reference.get(field, [])
            for model in MODELS:
                guess = parsed[model].get(field)
                field_rows.append({
                    "case_number": position, "dataset": doc["dataset_name"], "id": doc["id"],
                    "field": field, "model": model,
                    "reference_values": json.dumps(vals, ensure_ascii=False),
                    "prediction": guess if guess is not None else "<MISSING>",
                    "heuristic_exact_normalized_match": int(
                        guess is not None and any(
                            diagnostics.normalize(guess) == diagnostics.normalize(v) for v in vals
                        )
                    ) if vals else "",
                    "extra_key": field not in reference,
                })
        case_data = {
            "case_number": position, "selection_reason": reason,
            "document_key": key, "dataset": doc["dataset_name"],
            "id": str(doc["id"]), "image_path": doc["image_path"],
            "question": doc["question"], "reference": reference, "models": case_rows,
            "review_template": {
                "error_type": "", "likely_mechanism": "", "human_interpretation": "",
                "notes": "Human judgement required. Do not infer mechanism from score alone."
            },
        }
        write_json(case_folder / "case.json", case_data)
        lines = [
            f"# Case {position:02d}: {doc['dataset_name']} / {doc['id']}",
            "", f"**Selection reason:** {reason}",
            "", f"**OCRBench question:** {doc['question']}",
            "", f"**Image path:** `{doc['image_path']}`",
            "", "## Ground truth", "", "```json",
            json.dumps(reference, indent=2, ensure_ascii=False), "```",
            "", "## Seven scored predictions",
        ]
        for model in MODELS:
            lines.extend([
                "", f"### {model} — official score {case_rows[model]['official_score']:.4f}",
                "", "```text", str(case_rows[model]["prediction"] or ""), "```",
            ])
        lines.extend([
            "", "## Manual qualitative interpretation",
            "", "- [ ] Check source document and requested keys",
            "- [ ] Identify OCR recognition errors versus spatial/layout alignment errors",
            "- [ ] Check missing fields, hallucinated values and incorrect associations",
            "- [ ] Describe any success/failure difference across seven models",
            "- [ ] Note uncertainty; do not infer causality from one example",
        ])
        (case_folder / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        spatial_text = folders[SPATIAL] / "spatial_text" / f"{key}.txt"
        if spatial_text.is_file():
            shutil.copy2(spatial_text, case_folder / "spatial_text.txt")
        # Save the exact previously cached OCR inputs for close reading, no new OCR.
        for engine in ("paddleocr", "paddleocr_layout", "ppstructurev3"):
            index_path = batch_path.parent / "ocr" / engine / "index.json"
            if not index_path.is_file():
                continue
            ocr_index = read_json(index_path)
            if ocr_index.get("manifest_sha256") != batch["manifest_sha256"]:
                raise ValueError(f"Cache manifest mismatch for {engine}")
            cache_entry = ocr_index.get("records", {}).get(key, {})
            for source_field in ("text_path", "raw_path", "markdown_path", "json_path"):
                value = cache_entry.get(source_field)
                if not isinstance(value, str) or not value:
                    continue
                candidate = Path(value)
                original = next(
                    (p for p in (candidate, ROOT / candidate, batch_path.parent / candidate)
                     if p.is_file()), None
                )
                if original is None or original.suffix.lower() not in (".txt", ".md", ".json"):
                    continue
                if original.stat().st_size > 5_000_000:
                    continue
                input_dir = case_folder / "ocr_inputs"
                input_dir.mkdir(exist_ok=True)
                shutil.copy2(original, input_dir / f"{engine}_{source_field}{original.suffix.lower()}")
        if args.image_root:
            root = args.image_root.expanduser().resolve()
            original = Path(str(doc["image_path"]))
            image = original if original.is_absolute() else root / original
            if not image.is_file() and not original.is_absolute():
                image = root / original.name
            if image.is_file() and image.resolve().is_relative_to(root) and image.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                shutil.copy2(image, case_folder / f"source_image{image.suffix.lower()}")
        selection_rows.append({
            "case_number": position, "selection_reason": reason,
            "document_key": key, "dataset": doc["dataset_name"], "id": doc["id"],
            "image_path": doc["image_path"],
            **{f"{model}_score": float(scores[model][index]) for model in MODELS},
        })
    write_csv(tables / "qualitative_16_cases_selection.csv", selection_rows)
    write_csv(tables / "qualitative_field_diagnostics.csv", field_rows)

    if args.existing_review:
        previous = args.existing_review.expanduser().resolve()
        if not previous.is_dir() or previous == out or out.is_relative_to(previous):
            raise ValueError("Existing review must be a separate existing folder")
        shutil.copytree(previous, out / "archived_previous_qualitative_review")

    make_figures(figures, docs, scores, models_ci, pair_ci, selected, efficiency)
    # Preserve the original 6-model finalizer and the spatial comparison, where present.
    snapshots = out / "previous_statistical_analysis"
    sources = (
        ROOT / "runs" / "analysis" / batch["batch_id"],
        folders[SPATIAL] / "comparison",
        ROOT / "runs" / "analysis" / "seven_model_release",
    )
    for source in sources:
        if source.is_dir():
            dest = snapshots / source.name
            # Restrict to generated summaries, not raw predictions or oversized caches.
            for file in source.rglob("*"):
                if file.is_file() and file.suffix.lower() in (".csv", ".json", ".md", ".png", ".pdf", ".txt"):
                    target = dest / file.relative_to(source)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file, target)

    src_git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=False)
    write_json(out / "provenance" / "manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch["batch_id"], "batch_sha256": digest(batch_path),
        "manifest_sha256": batch["manifest_sha256"],
        "source_git_commit": src_git.stdout.strip() if src_git.returncode == 0 else None,
        "models": list(MODELS), "questions": 400, "qualitative_cases": 16,
        "case_selection": "user supplied 16 identities" if args.case_ids_file else
                          "purposive diagnostic selection: four per dataset; deterministic score contrasts",
        "bootstrap": {"resamples": args.n_bootstrap, "seed": args.seed,
                      "method": "paired, dataset-stratified percentile; same documents across models",
                      "pairwise_tests": 21, "familywise": "Bonferroni within model pair panel"},
        "sources": model_paths,
        "caveats": [
            "Official scores are OCRBench's own per-item KIE metrics, not field-level micro-F1.",
            "16 cases are selected for diagnostic discussion, not representative inference.",
            "Heuristic field matching is for qualitative diagnostics only, not official scoring.",
            "Spatial reuses saved LaOCR timing; only LM time is newly measured.",
            "PyTorch peak allocated memory omits the separate Paddle OCR allocations.",
            "Original six benchmark runs, evaluator outputs, and source analysis are never edited.",
        ],
    })
    report = [
        f"# Complete OCRBench v2 English KIE analysis — {batch['batch_id']}",
        "", f"Generated UTC: {now}. This is a **new analysis of saved results**, not a new model run.",
        "", "## Official seven-model results", "",
        "| Model | Mean official item score | Zero-score items | Perfect-score items |",
        "|---|---:|---:|---:|",
    ]
    report += [
        f"| {row['model']} | {row['official_mean_score']:.4f} | {row['zero_score_count']} | {row['perfect_score_count']} |"
        for row in overall
    ]
    report += [
        "", "## How to read this folder", "",
        "- `model_runs/`: frozen scored predictions, available raw predictions, model metadata and metrics.",
        "- `tables/`: official-score analyses, seven-model paired bootstrap, efficiency, 16-case selections and diagnostic fields.",
        "- `figures/`: seven publication-oriented charts as PNG and PDF.",
        "- `qualitative_16_cases/`: 16 case folders, each with question, reference, all seven predictions, manual-review checklist and spatial text when available.",
        "- `previous_statistical_analysis/`: copies of existing six-model, spatial and seven-model statistical summaries when present.",
        "- `provenance/`: exact input checksums, selection policy, caveats and output checksums.",
        "", "## Interpretation boundaries", "",
        "These 16 cases were purposefully selected to illustrate differences; they are not a representative sample.",
        "Qualitative interpretations must be written by a reviewer after checking the actual document image.",
        "Official item scores are separate from the diagnostic normalized field-match columns.",
        "Spatial OCR time was taken from saved LaOCR; its inference time is not a fresh end-to-end OCR+LM run.",
        "Do not publicly commit or publish these raw scored predictions or dataset images without permission.",
    ]
    (out / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    # Static, locally browsable visual report; no external links or scripting.
    from html import escape
    tr = lambda cells: "<tr>" + "".join("<td>" + escape(str(x)) + "</td>" for x in cells) + "</tr>"
    hrows = "\n".join(tr([r["model"], f"{r['official_mean_score']:.4f}",
                             r["zero_score_count"], r["perfect_score_count"]]) for r in overall)
    links = "\n".join(
        f'<li><a href="qualitative_16_cases/{escape((cases_dir / ("case_" + str(k).zfill(2) + "_" + docs[i]["dataset_name"] + "_" + re.sub(r"[^a-zA-Z0-9_-]+", "_", str(docs[i]["id"]))[:40]) / "review.md").relative_to(cases_dir).as_posix())}">'
        f'Case {k}: {escape(str(docs[i]["dataset_name"]))} #{escape(str(docs[i]["id"]))}</a> — {escape(reason)}</li>'
        for k, (i, reason) in enumerate(selected, 1)
    )
    chart_html = "\n".join(
        f'<figure><img src="figures/{escape(p.name)}" alt="{escape(p.stem)}"><figcaption>{escape(p.stem.replace("_", " "))}</figcaption></figure>'
        for p in sorted(figures.glob("*.png"))
    )
    (out / "INDEX.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>ULM KIE Complete Analysis</title>'
        '<style>body{font:16px system-ui,sans-serif;max-width:1080px;margin:40px auto;padding:0 20px;'
        'line-height:1.55;color:#222}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;'
        'padding:8px;text-align:left}figure{margin:30px 0}img{max-width:100%;height:auto;border:1px solid #ddd}'
        'a{color:#175a93}small{color:#555}</style>'
        f'<h1>Seven-model OCRBench v2 KIE analysis</h1><p>Batch: {escape(batch["batch_id"])} — '
        '400 English KIE questions, 16 purposively chosen case reviews.</p>'
        '<p><strong>Method:</strong> reuse saved official predictions and scores; no new GPU inference. '
        'Spatial OCR timing is a sum of reused OCR and freshly measured LM components, not end-to-end wall time.</p>'
        '<h2>Official model means</h2><table><tr><th>Model</th><th>Mean</th><th>Zero scores</th>'
        f'<th>Perfect scores</th></tr>{hrows}</table>'
        '<h2>Figures</h2>' + chart_html +
        '<h2>Qualitative cases</h2><p>Diagnostic, nonrepresentative selection. Human interpretation is required.</p>'
        f'<ol>{links}</ol><h2>Data files</h2>'
        '<p>See <a href="README.md">README</a>, <a href="tables/seven_model_bootstrap_ci.csv">model confidence intervals</a>, '
        '<a href="tables/seven_model_paired_bootstrap_ci.csv">pairwise intervals</a>, '
        '<a href="tables/qualitative_field_diagnostics.csv">field diagnostics</a>, '
        '<a href="tables/seven_model_document_scores.csv">all 400 matched scores</a>, '
        'and <a href="provenance/manifest.json">provenance</a>.</p>'
        '<small>Raw predictions and any images are intended only for authorized collaborators.</small></html>',
        encoding="utf-8",
    )
    checksums = [
        {"file": str(file.relative_to(out)), "bytes": file.stat().st_size, "sha256": digest(file)}
        for file in sorted(out.rglob("*")) if file.is_file()
    ]
    write_csv(out / "provenance" / "output_checksums.csv", checksums)

    if not args.no_zip:
        archive = out.parent / (out.name + ".zip")
        if archive.exists():
            raise FileExistsError(f"Archive already exists: {archive}")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for file in sorted(out.rglob("*")):
                if file.is_file():
                    z.write(file, arcname=str(file.relative_to(out.parent)))
        print(f"Shareable archive: {archive}")
    print(f"Complete seven-model analysis: {out}")
    print(f"Validated seven models x 400 scored documents; 16 qualitative case folders.")
    print(f"Spatial official mean: {np.mean(scores[SPATIAL]):.4f}")
    print("This command did not run inference or change any original run.")


if __name__ == "__main__":
    main()
