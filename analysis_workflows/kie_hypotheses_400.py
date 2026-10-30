import ast
import csv
import json
import re
from pathlib import Path

from ulm_extractor.storage import key_for

batch_id = "cbdd58de-531a-4c9c-9778-5bc8b0b0be48"
batch_dir = Path("runs/batches") / batch_id
out = Path("runs/analysis/hypotheses_400")

models = (
    "qwen35_vlm", "minicpm_vlm", "qwen35_ocr",
    "qwen25_ocr", "qwen35_laocr", "qwen35_ppstructure"
)

comparisons = (
    ("H1", "qwen35_ocr", "qwen25_ocr"),
    ("H2", "qwen35_vlm", "qwen35_ocr"),
    ("Layout_LaOCR", "qwen35_ocr", "qwen35_laocr"),
    ("Layout_PP", "qwen35_ocr", "qwen35_ppstructure")
)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def export(name, rows):
    path = out / name
    if not rows:
        raise ValueError(f"Empty output: {name}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{name}: {len(rows)} rows")

def normalize(value):
    return str(value).lower().strip().replace("\n", " ").replace(" ", "")

def parse_reference(doc):
    value = doc["answers"][0]
    for _ in range(3):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    if not isinstance(value, dict):
        raise ValueError(f"Unparseable reference: {doc['id']}")
    return {
        str(k): [str(x) for x in (v if isinstance(v, list) else [v])]
        for k, v in value.items()
    }

def parse_prediction(raw):
    if not isinstance(raw, str):
        return {}, "unparseable"
    match = re.search(r"```(?:python|json)?\n(.*?)\n```", raw, re.S | re.I)
    value = match.group(1) if match else raw.strip()
    try:
        data = json.loads(value)
        if isinstance(data, dict):
            return {str(k).strip(): str(v).strip() for k, v in data.items()}, "json"
    except (ValueError, TypeError):
        pass
    try:
        data = ast.literal_eval(value)
        if isinstance(data, dict):
            return {str(k).strip(): str(v).strip() for k, v in data.items()}, "python_dict"
    except (ValueError, SyntaxError, TypeError):
        pass
    pattern = r'["\']?([\w\s]+)["\']?\s*[:=]\s*["\']?([^\n,"\'{}]+)["\']?'
    data = {k.strip(): v.strip() for k, v in re.findall(pattern, value)}
    return data, "regex_fallback" if data else "unparseable"

batch = read_json(batch_dir / "batch.json")
docs = read_json(batch_dir / "documents.json")

if len(docs) != 400 or len({key_for(d) for d in docs}) != 400:
    raise ValueError("Expected 400 unique questions")

run_root = batch_dir.parents[1]
run_ids = {s["id"]: s["run_id"] for s in batch["runs"]}
predictions, metrics = {}, {}

for model in models:
    run_dir = run_root / run_ids[model]
    records = read_json(run_dir / "predictions_scored.json")
    keyed = {key_for(r): r for r in records}
    if len(records) != 400 or set(keyed) != {key_for(d) for d in docs}:
        raise ValueError(f"Unmatched records: {model}")
    predictions[model] = keyed
    metrics[model] = read_json(run_dir / "metrics.json")

caches = {}
for engine in ("paddleocr", "paddleocr_layout", "ppstructurev3"):
    caches[engine] = read_json(
        batch_dir / "ocr" / engine / "index.json"
    )["records"]

questions, fields = [], []

for doc in docs:
    key = key_for(doc)
    ref = parse_reference(doc)
    observed = {m: predictions[m][key] for m in models}
    parsed = {m: parse_prediction(observed[m].get("predict")) for m in models}

    flat = Path(caches["paddleocr"][key]["text_path"]).read_text(encoding="utf-8")
    layout = Path(caches["paddleocr_layout"][key]["text_path"]).read_text(encoding="utf-8")
    lines = json.loads(layout)["lines"]

    pp = caches["ppstructurev3"][key]
    markdown = Path(pp["text_path"]).read_text(encoding="utf-8")
    audit = read_json(pp["audit_path"])

    q = {
        "document_key": key,
        "dataset": doc["dataset_name"],
        "id": doc["id"],
        "image_path": doc["image_path"],
        "reference_fields": len(ref),
        "plain_laocr_same_text": int(
            flat == "\n".join(str(x["text"]) for x in lines)
        ),
        "plain_ocr_chars": len(flat),
        "laocr_json_chars": len(layout),
        "pp_markdown_chars": len(markdown),
        "pp_unplaced_lines": sum(
            p.get("additional_ocr_lines", 0)
            for p in audit.get("pages", [])
        ),
        "pp_table_rows": len(re.findall(r"<tr\b", markdown, re.I)),
        "pp_table_cells": len(re.findall(r"<td\b", markdown, re.I))
    }

    for model in models:
        prediction, kind = parsed[model]
        correct = {
            field for field, variants in ref.items()
            if field in prediction and any(
                normalize(prediction[field]) == normalize(v)
                for v in variants
            )
        }
        wrong = set(ref) - correct
        swaps = sum(
            field in prediction and any(
                normalize(prediction[field]) == normalize(candidate)
                for other, variants in ref.items()
                if other != field
                for candidate in variants
            )
            for field in wrong
        )

        q.update({
            f"{model}_score": observed[model]["score"],
            f"{model}_parse": kind,
            f"{model}_token_limit": int(
                bool(observed[model].get("hit_token_limit"))
            ),
            f"{model}_output_tokens": observed[model].get("output_tokens", ""),
            f"{model}_correct_fields": len(correct),
            f"{model}_missing_fields": len(set(ref) - set(prediction)),
            f"{model}_extra_fields": len(set(prediction) - set(ref)),
            f"{model}_candidate_swaps": swaps
        })

    questions.append(q)

    names = list(ref) + sorted(
        set().union(*(set(parsed[m][0]) for m in models)) - set(ref)
    )

    for field in names:
        variants = ref.get(field, [])
        f = {
            "document_key": key,
            "dataset": doc["dataset_name"],
            "id": doc["id"],
            "field": field,
            "reference_values": json.dumps(variants, ensure_ascii=False),
            "reference_field": int(field in ref)
        }

        for model in models:
            pred = parsed[model][0]
            f[f"{model}_value"] = pred.get(field, "<MISSING>")
            f[f"{model}_correct"] = int(
                field in pred and any(
                    normalize(pred[field]) == normalize(v)
                    for v in variants
                )
            )

        fields.append(f)

pairs = []

for hypothesis, a, b in comparisons:
    for dataset in ("ALL", "FUNSD", "POIE", "SROIE", "Receipt"):
        selection = [
            q for q in questions
            if dataset == "ALL" or q["dataset"] == dataset
        ]
        diff = [
            float(q[f"{a}_score"]) - float(q[f"{b}_score"])
            for q in selection
        ]

        pairs.append({
            "hypothesis": hypothesis,
            "dataset": dataset,
            "model_a": a,
            "model_b": b,
            "questions": len(diff),
            "mean_a_minus_b": sum(diff) / len(diff),
            "a_higher": sum(d > 1e-10 for d in diff),
            "tied": sum(abs(d) <= 1e-10 for d in diff),
            "b_higher": sum(d < -1e-10 for d in diff)
        })

summaries = []

for dataset in ("ALL", "FUNSD", "POIE", "SROIE", "Receipt"):
    subset = [
        q for q in questions
        if dataset == "ALL" or q["dataset"] == dataset
    ]

    for model in models:
        summaries.append({
            "dataset": dataset,
            "model": model,
            "questions": len(subset),
            "mean_score": sum(
                float(q[f"{model}_score"]) for q in subset
            ) / len(subset),
            "perfect_questions": sum(
                float(q[f"{model}_score"]) == 1 for q in subset
            ),
            "zero_questions": sum(
                float(q[f"{model}_score"]) == 0 for q in subset
            ),
            "unparseable": sum(
                q[f"{model}_parse"] == "unparseable" for q in subset
            ),
            "token_limit_hits": sum(
                q[f"{model}_token_limit"] for q in subset
            ),
            "correct_fields": sum(
                q[f"{model}_correct_fields"] for q in subset
            ),
            "missing_fields": sum(
                q[f"{model}_missing_fields"] for q in subset
            ),
            "candidate_swaps": sum(
                q[f"{model}_candidate_swaps"] for q in subset
            )
        })

efficiency = []

for model in models:
    m = metrics[model]
    efficiency.append({
        "model": model,
        "mean_processing_s": m.get("mean_latency_s"),
        "mean_ocr_s": m.get("mean_ocr_time_s"),
        "mean_lm_s": m.get("mean_lm_time_s"),
        "peak_torch_gib": m.get("peak_torch_memory_gib"),
        "mean_output_tokens": m.get("mean_output_tokens"),
        "token_limit_hits": sum(
            q[f"{model}_token_limit"] for q in questions
        )
    })

if out.exists() and any(out.iterdir()):
    raise FileExistsError(f"Output directory is not empty: {out}")

out.mkdir(parents=True, exist_ok=True)

export("kie_400_questions.csv", questions)
export("kie_400_fields.csv", fields)
export("hypothesis_pairwise_descriptive.csv", pairs)
export("model_error_summary.csv", summaries)
export("efficiency_descriptive.csv", efficiency)

print("Saved analysis to", out)
