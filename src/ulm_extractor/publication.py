"""GPU-free, reproducible publication analysis of seven scored OCRBench v2 KIE runs.

Primary inference: stratified *paired* percentile bootstrap, resampling 400
matched questions with replacement. A secondary CSV preserves recorded
timing metrics; no OCR/PyTorch inference is executed here.
"""
import argparse
import base64
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "analysis" / "scored_documents.json"
REGISTRY = ROOT / "analysis" / "experiments.csv"
EFFICIENCY = ROOT / "analysis" / "efficiency_observations.csv"
DEFAULT_OUT = ROOT / "results" / "final"
DATASET_COUNTS = {"FUNSD": 48, "POIE": 177, "Receipt": 25, "SROIE": 150}
PRIMARY = ("qwen35_vlm", "minicpm_vlm", "qwen35_ocr", "qwen25_ocr")
EXPANDED = (*PRIMARY, "qwen35_laocr", "qwen35_ppstructure_v2")
RECOVERY = ("qwen35_ppstructure_v2", "qwen35_ppstructure_v1")


def load_panel(path=SCORES):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get("schema") != "ocrbench-v2-kie-document-scores-v1":
        raise ValueError("Unsupported scored-document schema")
    n = data.get("n_documents")
    names = data.get("model_columns")
    levels = data.get("score_dictionary")
    if n != 400 or names != list((*EXPANDED, "qwen35_ppstructure_v1")):
        raise ValueError("Expected seven named experiments with 400 aligned documents")
    if not isinstance(levels, list) or len(levels) > 256 or not all(
        type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1
        for x in levels
    ):
        raise ValueError("Invalid numeric score dictionary")
    values = base64.b64decode(data["score_indices_b64"], validate=True)
    packed = base64.b64decode(data["subset_indices_2bit_b64"], validate=True)
    if len(values) != n * len(names) or len(packed) != n // 4:
        raise ValueError("Scored-document encoding has an incorrect length")
    if any(idx >= len(levels) for idx in values):
        raise ValueError("Invalid score dictionary index")
    subset_codes = [(byte >> (2 * part)) & 3
                    for byte in packed for part in range(4)]
    datasets = [data["subset_names"][idx] for idx in subset_codes]
    if dict(Counter(datasets)) != DATASET_COUNTS:
        raise ValueError("Incorrect FUNSD/POIE/Receipt/SROIE composition")
    if data["first_document_id"] != 900:
        raise ValueError("Unexpected OCRBench item numbering")
    by_model = {
        name: np.asarray([levels[values[i * len(names) + j]]
                          for i in range(n)], dtype=np.float64)
        for j, name in enumerate(names)
    }
    rows = [
        {"dataset_name": datasets[i], "id": str(900 + i),
         **{name: float(by_model[name][i]) for name in names}}
        for i in range(n)
    ]
    if len({(r["dataset_name"], r["id"]) for r in rows}) != n:
        raise ValueError("Document identity collision")
    return rows, by_model, data, hashlib.sha256(raw).hexdigest()


def load_registry(path=REGISTRY):
    with Path(path).open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    registry = {}
    for row in records:
        name = row["publication_label"]
        if name in registry:
            raise ValueError(f"Duplicate publication label: {name}")
        if len(row["model_revision"]) != 40 or len(row["scored_predictions_sha256"]) != 64:
            raise ValueError(f"Invalid revision or checksum: {name}")
        registry[name] = row
    if set(registry) != set((*EXPANDED, "qwen35_ppstructure_v1")):
        raise ValueError("Registry must identify the same seven scored models")
    return registry


def validate_archive(archive, rows, score, registry, manifest_hash):
    """Optional strict verification against the full original runs(3).zip."""
    archive = Path(archive)
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        original_document_identities = None
        for name, meta in registry.items():
            batch, run = meta["batch_id"], meta["run_id"]
            base = f"runs/batches/{batch}/"
            config_path = base + "batch.json"
            doc_path = base + "documents.json"
            scored_path = f"runs/{run}/predictions_scored.json"
            for path in (config_path, doc_path, scored_path, f"runs/{run}/run.json"):
                if path not in names:
                    raise ValueError(f"Missing file in original archive: {path}")
            batch_config = json.loads(zf.read(config_path))
            if batch_config["batch_id"] != batch or batch_config["manifest_sha256"] != manifest_hash:
                raise ValueError(f"Batch provenance mismatch for {name}")
            doclist = json.loads(zf.read(doc_path))
            identities = [(str(r["dataset_name"]), str(r["id"]), str(r["image_path"]),
                           str(r["question"])) for r in doclist]
            if len(identities) != 400 or len(set(identities)) != 400:
                raise ValueError(f"Bad original manifest for {name}")
            if original_document_identities is None:
                original_document_identities = set(identities)
            elif set(identities) != original_document_identities:
                raise ValueError(f"Different document identities for {name}")
            status = json.loads(zf.read(f"runs/{run}/run.json"))
            if status["status"] != "scored" or status["batch_id"] != batch:
                raise ValueError(f"Run status mismatch: {name}")
            raw = zf.read(scored_path)
            if hashlib.sha256(raw).hexdigest() != meta["scored_predictions_sha256"]:
                raise ValueError(f"Original scored file SHA256 mismatch: {name}")
            saved = json.loads(raw)
            if len(saved) != 400:
                raise ValueError(f"Not 400 predictions for {name}")
            seen = set()
            for pred in saved:
                key = (str(pred["dataset_name"]), str(pred["id"]),
                       str(pred["image_path"]), str(pred["question"]))
                if key not in original_document_identities or key in seen or pred.get("error"):
                    raise ValueError(f"Unmatched, duplicate, or failed document in {name}")
                seen.add(key)
                offset = int(pred["id"]) - 900
                if not 0 <= offset < 400 or rows[offset]["dataset_name"] != str(pred["dataset_name"]):
                    raise ValueError(f"Document alignment mismatch in {name}")
                if not math.isclose(float(pred["score"]), float(score[name][offset]),
                                    rel_tol=0, abs_tol=1e-15):
                    raise ValueError(f"Published score disagrees with original for {name}, {key}")
            if name.startswith("qwen35_ppstructure"):
                index = json.loads(zf.read(base + "ocr/ppstructurev3/index.json"))
                if index["format"] != meta["ocr_representation"]:
                    raise ValueError(f"Incorrect PP Markdown version for {name}")
        return {
            "archive": str(archive), "archive_sha256": file_sha256(archive),
            "checked_scored_runs": 7, "checked_document_predictions": 2800,
            "document_identity_sets_match": True,
            "scored_file_sha256s_match": True,
            "pp_v1_v2_ocr_formats_match": True,
        }


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(scores_path=SCORES, archive=None):
    rows, scores, info, input_hash = load_panel(scores_path)
    registry = load_registry()
    for name, metadata in registry.items():
        actual = float(scores[name].mean())
        expected = float(metadata["official_kie_f1"])
        if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError(
                f"Published score mean disagrees with official recorded F1 for "
                f"{name}: {actual} != {expected}"
            )
    evidence = {
        "validated": True,
        "documents": len(rows),
        "scored_runs": len(registry),
        "dataset_counts": dict(Counter(row["dataset_name"] for row in rows)),
        "dataset_manifest_sha256": info["dataset_manifest_sha256"],
        "published_scores_file_sha256": input_hash,
        "source_score_matrix_csv_sha256": info["source_sha256"],
        "source_repository_commit": "4aff4fb97a5f818884584720bd2e607c7a1f4245",
        "original_scored_predictions_sha256": {
            name: record["scored_predictions_sha256"]
            for name, record in registry.items()
        },
        "official_means_match": True,
        "archive_verification": (
            validate_archive(archive, rows, scores, registry, info["dataset_manifest_sha256"])
            if archive else "not requested; supply --archive runs(3).zip"
        ),
    }
    return rows, scores, evidence


def percentile_interval(draws, confidence):
    alpha = (1 - confidence) / 2
    q = np.quantile(draws, (alpha, 1 - alpha))
    return float(q[0]), float(q[1])


def bootstrap_panel(rows, scores, selected, draws=10000, seed=2026, confidence=0.95):
    if not (0 < confidence < 1) or draws < 100:
        raise ValueError("Invalid bootstrap draw count or confidence level")
    strata = {
        dataset: np.array([i for i, row in enumerate(rows)
                           if row["dataset_name"] == dataset], dtype=np.int32)
        for dataset in DATASET_COUNTS
    }
    rng = np.random.default_rng(seed)
    # This exact RNG order matches the archived independent 2026 analysis.
    all_indices = np.empty((draws, 400), dtype=np.int32)
    for indices in strata.values():
        all_indices[:, indices] = rng.choice(
            indices, size=(draws, len(indices)), replace=True
        )
    scopes = [("ALL", np.arange(400, dtype=np.int32), all_indices)]
    for dataset, indices in strata.items():
        scopes.append((dataset, indices, rng.choice(
            indices, size=(draws, len(indices)), replace=True
        )))
    model_results, pair_results = [], []
    npairs = len(selected) * (len(selected) - 1) // 2
    adjusted = 1 - (1 - confidence) / npairs
    for subset, positions, sampled_positions in scopes:
        boot_means = {}
        for name in selected:
            sample_means = scores[name][sampled_positions].mean(axis=1)
            boot_means[name] = sample_means
            low, high = percentile_interval(sample_means, confidence)
            model_results.append({
                "dataset": subset, "experiment_id": name,
                "n_documents": len(positions),
                "official_kie_f1": float(scores[name][positions].mean()),
                "ci95_low": low, "ci95_high": high, "draws": draws, "seed": seed,
            })
        for a, b in itertools.combinations(selected, 2):
            differences = scores[a][positions] - scores[b][positions]
            boot_differences = boot_means[a] - boot_means[b]
            low, high = percentile_interval(boot_differences, confidence)
            corrected_low, corrected_high = percentile_interval(
                boot_differences, adjusted
            )
            pair_results.append({
                "dataset": subset, "model_a": a, "model_b": b,
                "n_documents": len(positions),
                "mean_f1_difference_a_minus_b": float(differences.mean()),
                "ci95_low": low, "ci95_high": high,
                "ci95_excludes_zero": low > 0 or high < 0,
                "bonferroni95_low": corrected_low,
                "bonferroni95_high": corrected_high,
                "bonferroni95_excludes_zero": corrected_low > 0 or corrected_high < 0,
                "a_wins": int(np.sum(differences > 1e-12)),
                "b_wins": int(np.sum(differences < -1e-12)),
                "ties": int(np.sum(abs(differences) <= 1e-12)),
                "n_pairs_in_family": npairs, "draws": draws, "seed": seed,
            })
    return model_results, pair_results


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write empty analysis table")
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_figures(out, all_models, primary_pairs, pp_pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [r for r in all_models if r["dataset"] == "ALL"]
    names = [r["experiment_id"] for r in models]
    y = np.arange(len(models))
    values = np.array([100 * r["official_kie_f1"] for r in models])
    lows = np.array([100 * r["ci95_low"] for r in models])
    highs = np.array([100 * r["ci95_high"] for r in models])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(values, y, xerr=[values-lows, highs-values], fmt="o", capsize=3)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("Mean official document-level KIE score (%)")
    ax.set_title("OCRBench v2 English KIE: score and 95% bootstrap CI")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "figure_model_accuracy.png", dpi=180)
    plt.close(fig)

    def pairplot(rows, file, title):
        scope = [r for r in rows if r["dataset"] == "ALL"]
        labels = [f'{r["model_a"]} - {r["model_b"]}' for r in scope]
        d = np.array([100 * r["mean_f1_difference_a_minus_b"] for r in scope])
        lo = np.array([100 * r["ci95_low"] for r in scope])
        hi = np.array([100 * r["ci95_high"] for r in scope])
        fig, ax = plt.subplots(figsize=(10, max(2.5, 0.58*len(scope)+1.5)))
        yy = np.arange(len(scope))
        ax.errorbar(d, yy, xerr=[d-lo, hi-d], fmt="o", capsize=3)
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_yticks(yy, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Paired mean score difference (percentage points)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / file, dpi=180)
        plt.close(fig)
    pairplot(primary_pairs, "figure_primary_pairwise.png",
             "Original four architectures: paired 95% bootstrap CI")
    pairplot(pp_pairs, "figure_pp_recovery.png",
             "PP-StructureV3 Markdown recovery: paired 95% bootstrap CI")


def build(out=DEFAULT_OUT, draws=10000, seed=2026, archive=None, figures=True):
    rows, scores, evidence = validate(archive=archive)
    out = Path(out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    panels = {
        "primary": (PRIMARY, seed),
        "expanded": (EXPANDED, seed + 1),
        "pp_recovery": (RECOVERY, seed + 2),
    }
    generated = {}
    for title, (models, panel_seed) in panels.items():
        m, p = bootstrap_panel(rows, scores, models, draws=draws, seed=panel_seed)
        write_csv(out / f"{title}_model_confidence_intervals.csv", m)
        write_csv(out / f"{title}_paired_differences.csv", p)
        generated[title] = (m, p)
    write_csv(out / "all_seven_document_scores.csv", rows)
    with EFFICIENCY.open(newline="", encoding="utf-8") as f:
        efficiency = list(csv.DictReader(f))
    if {r["publication_label"] for r in efficiency} != set(scores):
        raise ValueError("Efficiency observations do not map to the same seven runs")
    write_csv(out / "efficiency.csv", efficiency)
    evidence.update({
        "statistics": "paired, subset-stratified percentile bootstrap",
        "sampling_unit": "matched document/question, with replacement",
        "fixed_subset_counts": DATASET_COUNTS,
        "bootstrap_draws": draws,
        "confidence_level": 0.95,
        "primary_seed": seed, "expanded_seed": seed + 1,
        "pp_recovery_seed": seed + 2,
        "multiple_comparisons": "Bonferroni adjustment across model pairs within each subset only",
        "limitations": (
            "Benchmark conditional uncertainty; no evidence of generalization to "
            "unobserved domains or stochastic decoding variability. Efficiency "
            "metrics mix A100 40GB/80GB workers, measure torch memory not Paddle "
            "memory, and combine separately recorded OCR+LM processing times."
        ),
    })
    (out / "validation_report.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )
    if figures:
        draw_figures(out, generated["expanded"][0],
                     generated["primary"][1], generated["pp_recovery"][1])
    print(f"PASS: {len(rows)} documents, {len(scores)} scored runs, "
          f"{draws} paired bootstrap draws per panel")
    print(f"Results: {out}")
    for panel, (model_result, pair_result) in generated.items():
        print(f"{panel}: {len(model_result)} model/subset estimates, "
              f"{len(pair_result)} paired/subset estimates")
    return out


def build_main(argv=None):
    p = argparse.ArgumentParser(description="Reproduce final paper tables and figures (CPU only)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--draws", type=int, default=10000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--archive", type=Path, default=None,
                   help="Optional original runs(3).zip for raw prediction SHA and score audit")
    p.add_argument("--no-figures", action="store_true")
    a = p.parse_args(argv)
    build(out=a.out, draws=a.draws, seed=a.seed, archive=a.archive,
          figures=not a.no_figures)


def validate_main(argv=None):
    p = argparse.ArgumentParser(description="Validate published seven-run score panel")
    p.add_argument("--archive", type=Path, default=None,
                   help="Optional original runs(3).zip to verify raw scored file SHA256")
    a = p.parse_args(argv)
    rows, scores, evidence = validate(archive=a.archive)
    print(f"PASS: {len(rows)} matched questions, {len(scores)} official models")
    for k in scores:
        print(f"  {k:25s}: {scores[k].mean():.12f}")
    print(f"Source manifest: {evidence['dataset_manifest_sha256']}")
    if a.archive:
        print(f"Full original archive verified: {evidence['archive_verification']['archive_sha256']}")
    else:
        print("Original archive not checked (pass --archive for full provenance validation)")
