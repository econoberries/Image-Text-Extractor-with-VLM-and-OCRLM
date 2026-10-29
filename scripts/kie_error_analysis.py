"""Export question- and field-level KIE error comparisons from saved model results.

Diagnostic only: this does not rerun inference or modify official scores.
Field matching follows the OCRBench v2 case/whitespace normalization, but
is separate from the official question-level scoring implementation.
"""
import argparse
import ast
import csv
import json
from pathlib import Path
import re

MODELS = (
    ("qwen35_vlm", "vlm"),
    ("qwen35_ocr", "ocr35"),
    ("qwen25_ocr", "ocr25"),
)


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def identity(row):
    return (str(row["dataset_name"]), str(row["id"]),
            str(row["image_path"]), str(row["question"]))


def parse_prediction(text):
    """Mirror OCRBench v2's convert_str_to_dict, including its regex fallback."""
    if not isinstance(text, str):
        return {}
    match = re.search(r"[\x60]{3}(?:python|json)?\n(.*?)\n[\x60]{3}", text, re.DOTALL | re.IGNORECASE)
    content = match.group(1) if match else text.strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(content)
            if not isinstance(data, dict):
                raise ValueError("Expected a dictionary")
        except (ValueError, SyntaxError):
            data = {}
            pattern = r'["\']?([\w\s]+)["\']?\s*[:=]\s*["\']?([^\n,"\'{}]+)["\']?'
            for key, value in re.findall(pattern, content):
                data[key.strip()] = value.strip()
    try:
        return {key.strip(): str(value).strip() for key, value in data.items()}
    except (AttributeError, TypeError, ValueError):
        return {}


def reference_values(row):
    raw = row["answers"]
    if not isinstance(raw, list) or len(raw) != 1:
        raise ValueError(f"Expected one reference for question {row['id']}")
    gold = raw[0]
    for _ in range(2):
        if not isinstance(gold, str):
            break
        try:
            gold = ast.literal_eval(gold)
        except (SyntaxError, ValueError):
            gold = json.loads(gold)
    if not isinstance(gold, dict):
        raise ValueError(f"Expected reference dictionary for question {row['id']}")
    return {str(key): [str(v) for v in (values if isinstance(values, list) else [values])]
            for key, values in gold.items()}


def normalize(value):
    return str(value).lower().strip().replace("\n", " ").replace(" ", "")


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def make_reports(batch_path, output_dir):
    batch_path = Path(batch_path).expanduser().resolve()
    batch = read_json(batch_path)
    batch_root = batch_path.parent
    run_root = batch_root.parents[1]
    specs = {item["id"]: item for item in batch["runs"]}
    documents = read_json(batch_root / "documents.json")
    if len(documents) != 400:
        raise ValueError("Expected 400 benchmark questions")
    document_ids = [identity(row) for row in documents]
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("Duplicate benchmark question identity")

    predictions = {}
    for model_id, alias in MODELS:
        entries = read_json(run_root / specs[model_id]["run_id"] / "predictions_scored.json")
        indexed = {identity(row): row for row in entries}
        if len(entries) != 400 or set(indexed) != set(document_ids):
            raise ValueError(f"Unmatched or duplicate predictions for {model_id}")
        predictions[alias] = indexed

    question_rows, field_rows = [], []
    for document, key in zip(documents, document_ids):
        row_models = {alias: predictions[alias][key] for _, alias in MODELS}
        refs = reference_values(document)
        parsed = {alias: parse_prediction(row_models[alias].get("predict"))
                  for _, alias in MODELS}
        score = {alias: float(row_models[alias]["score"]) for _, alias in MODELS}
        question_rows.append({
            "id": str(document["id"]),
            "dataset": str(document["dataset_name"]),
            "image_path": str(document["image_path"]),
            "question": str(document["question"]),
            "answers": json.dumps(refs, ensure_ascii=False),
            **{f"{alias}_score": score[alias] for _, alias in MODELS},
            "vlm_minus_ocr35": score["vlm"] - score["ocr35"],
            "ocr35_minus_ocr25": score["ocr35"] - score["ocr25"],
            **{f"{alias}_prediction": str(row_models[alias].get("predict", ""))
               for _, alias in MODELS},
            **{f"{alias}_hit_token_limit": bool(row_models[alias].get("hit_token_limit"))
               for _, alias in MODELS},
        })
        extra = sorted(set().union(*(set(d) for d in parsed.values())) - set(refs))
        for field in [*refs, *extra]:
            choices = refs.get(field, [])
            result = {
                "id": str(document["id"]),
                "dataset": str(document["dataset_name"]),
                "image_path": str(document["image_path"]),
                "key": field,
                "references": " | ".join(choices) if choices else "<EXTRA KEY>",
            }
            for _, alias in MODELS:
                pred = parsed[alias]
                result[f"{alias}_prediction"] = pred.get(field, "<MISSING>")
                result[f"{alias}_correct"] = int(
                    field in pred and any(normalize(pred[field]) == normalize(g) for g in choices)
                )
            field_rows.append(result)

    output_dir = Path(output_dir).expanduser().resolve()
    question_path = output_dir / "kie_question_comparison.csv"
    field_path = output_dir / "kie_field_comparison.csv"
    if question_path.exists() or field_path.exists():
        raise FileExistsError("Refusing to overwrite existing KIE diagnostic CSV files")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(question_path, question_rows, list(question_rows[0]))
    write_csv(field_path, field_rows, list(field_rows[0]))
    print(f"Saved question- and field-level KIE comparisons: {output_dir}")
    return question_path, field_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make_reports(args.batch, args.output_dir)


if __name__ == "__main__":
    main()
