#!/usr/bin/env python3
"""
Aggregate contents of individual model-architecture runs into one csv.
"""
import pandas as pd
from pathlib import Path
import json

def aggregate_model_json(df):
    results = []

    # Find predictions_scored.json for each run in benchmark_summary
    for _, row in df.iterrows():
        run_id = row["run_id"]

        # Model name: text after the first "/" + pipeline
        model = str(row["model"]).split("/", 1)[-1]
        pipeline = str(row["pipeline"])
        model_name = f"{model}_{pipeline}"

        json_path = "data" / Path(run_id) / "predictions_scored.json"

        if not json_path.exists():
            print(f"Warning: File not found: {json_path}")
            continue

        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = [data]

        for record in data:
            results.append({
                "model_name": model_name,
                "run_id": run_id,
                **record,
            })

    return pd.DataFrame(results)

def main():
    benchmark_summary = pd.read_csv("data/batches/cbdd58de-531a-4c9c-9778-5bc8b0b0be48/benchmark_summary.csv")
    agg_json = aggregate_model_json(benchmark_summary)
    agg_json.to_csv("data/aggregated_predictions_scored.csv")
    print("success")

if __name__ == "__main__":
    main()
