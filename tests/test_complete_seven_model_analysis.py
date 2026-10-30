"""Offline tests for case selection (no benchmark data or GPU required)."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "complete_seven_analysis", ROOT / "scripts/build_complete_seven_model_analysis.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class CompleteSevenModelAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            {
                "id": str(i), "dataset_name": dataset,
                "image_path": f"{dataset}/{i}.png", "question": f"Extract {i}",
            }
            for dataset, count in [("FUNSD", 48), ("POIE", 177), ("Receipt", 25), ("SROIE", 150)]
            for i in range(count)
        ]
        assert len(self.docs) == 400
        self.scores = {
            model: np.linspace(0.05, 0.95, 400) for model in module.MODELS
        }
        self.scores["qwen35_spatial_ocr"] = np.linspace(0.96, 0.06, 400)

    def test_selects_exactly_four_per_dataset_deterministically(self):
        a = module.choose_cases(self.docs, self.scores, None)
        b = module.choose_cases(self.docs, self.scores, None)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 16)
        self.assertEqual(len(set(i for i, reason in a)), 16)
        for dataset in module.DATASETS:
            self.assertEqual(sum(self.docs[i]["dataset_name"] == dataset for i, _ in a), 4)

    def test_can_preserve_exact_preselected_case_ids(self):
        import csv
        chosen = [0, 1, 2, 3, 48, 49, 50, 51, 225, 226, 227, 228, 250, 251, 252, 253]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "16_cases.csv"
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["dataset_name", "id"])
                writer.writeheader()
                writer.writerows([{
                    "dataset_name": self.docs[i]["dataset_name"],
                    "id": self.docs[i]["id"],
                } for i in chosen])
            selected = module.choose_cases(self.docs, self.scores, path)
            self.assertEqual([i for i, _ in selected], chosen)

    def test_rejects_duplicate_or_missing_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.txt"
            path.write_text("\n".join(["nonexistent"] * 16), encoding="utf-8")
            with self.assertRaises(ValueError):
                module.choose_cases(self.docs, self.scores, path)


if __name__ == "__main__":
    unittest.main()
