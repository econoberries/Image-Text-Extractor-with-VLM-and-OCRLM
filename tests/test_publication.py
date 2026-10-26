"""Publication data-integrity, statistical reproducibility and CPU-only tests."""
from pathlib import Path
import csv
import json
import math
import tempfile
import unittest

import numpy as np
from ulm_extractor.publication import (
    PRIMARY, EXPANDED, RECOVERY, DATASET_COUNTS,
    bootstrap_panel, build, load_panel, load_registry, validate,
)


class PublicationReleaseTests(unittest.TestCase):
    def test_seven_models_and_all_four_strata(self):
        rows, scores, info, sha = load_panel()
        self.assertEqual(len(rows), 400)
        self.assertEqual(len(scores), 7)
        self.assertEqual(list(scores), [*EXPANDED, "qwen35_ppstructure_v1"])
        self.assertEqual({k: sum(x["dataset_name"] == k for x in rows)
                          for k in DATASET_COUNTS}, DATASET_COUNTS)
        self.assertEqual([r["id"] for r in rows], [str(i) for i in range(900, 1300)])
        self.assertEqual(len(sha), 64)

    def test_exact_official_means_and_provenance(self):
        rows, scores, evidence = validate()
        registry = load_registry()
        self.assertTrue(evidence["official_means_match"])
        self.assertEqual(evidence["scored_runs"], 7)
        for name in scores:
            self.assertAlmostEqual(float(scores[name].mean()),
                                   float(registry[name]["official_kie_f1"]), places=12)
        self.assertEqual(registry["qwen35_ppstructure_v1"]["ocr_representation"],
                         "ppstructurev3-markdown-v1")
        self.assertEqual(registry["qwen35_ppstructure_v2"]["ocr_representation"],
                         "ppstructurev3-markdown-recovered-v2")
        self.assertNotEqual(registry["qwen35_ppstructure_v1"]["run_id"],
                            registry["qwen35_ppstructure_v2"]["run_id"])

    def test_exact_primary_bootstrap_statistics_are_deterministic(self):
        rows, scores, _ = validate()
        a, b = bootstrap_panel(rows, scores, PRIMARY, draws=1000, seed=2026)
        c, d = bootstrap_panel(rows, scores, PRIMARY, draws=1000, seed=2026)
        self.assertEqual(a, c)
        self.assertEqual(b, d)
        comparison = next(r for r in b if r["dataset"] == "ALL"
                          and r["model_a"] == "qwen35_vlm"
                          and r["model_b"] == "qwen35_ocr")
        self.assertAlmostEqual(comparison["mean_f1_difference_a_minus_b"],
                               0.1028310425277335, places=9)
        self.assertEqual(comparison["a_wins"], 160)
        self.assertEqual(comparison["b_wins"], 20)
        self.assertEqual(comparison["ties"], 220)
        self.assertGreater(comparison["ci95_low"], 0)
        self.assertLess(comparison["ci95_high"], 1)

    def test_original_and_recovered_pp_are_paired(self):
        rows, scores, _ = validate()
        _, paired = bootstrap_panel(rows, scores, RECOVERY, draws=1000, seed=2028)
        comparison = next(row for row in paired if row["dataset"] == "ALL")
        self.assertAlmostEqual(comparison["mean_f1_difference_a_minus_b"],
                               0.13282167906984083, places=12)
        self.assertEqual((comparison["a_wins"],
                          comparison["b_wins"], comparison["ties"]), (155, 29, 216))

    def test_build_generates_reproducible_files(self):
        with tempfile.TemporaryDirectory() as directory:
            out = build(out=Path(directory) / "results", draws=300,
                        seed=2026, figures=False)
            expected = [
                "primary_model_confidence_intervals.csv",
                "primary_paired_differences.csv",
                "expanded_model_confidence_intervals.csv",
                "expanded_paired_differences.csv",
                "pp_recovery_model_confidence_intervals.csv",
                "pp_recovery_paired_differences.csv",
                "all_seven_document_scores.csv",
                "efficiency.csv", "validation_report.json"
            ]
            for name in expected:
                self.assertTrue((out / name).is_file(), name)
            with (out / "primary_paired_differences.csv").open() as f:
                pairs = list(csv.DictReader(f))
            self.assertEqual(len(pairs), 30)  # 6 pairs × 5 scopes
            with (out / "expanded_paired_differences.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 75)
            with (out / "all_seven_document_scores.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 400)
            with (out / "validation_report.json").open() as f:
                report = json.load(f)
            self.assertTrue(report["validated"])


if __name__ == "__main__":
    unittest.main()
