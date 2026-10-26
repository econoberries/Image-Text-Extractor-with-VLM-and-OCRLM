"""Fast offline smoke tests for a fresh six-model, same-GPU benchmark."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from ulm_extractor.storage import write_json

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "finalize_benchmark.py"
s = importlib.util.spec_from_file_location("finalize_benchmark", SCRIPT)
finalizer = importlib.util.module_from_spec(s)
s.loader.exec_module(finalizer)


class FinalizeTests(unittest.TestCase):
    def make_run(self, root):
        runroot = root / "runs"
        batchdir = runroot / "batches" / "new-batch"
        batchdir.mkdir(parents=True)
        counts = (("FUNSD", 48), ("POIE", 177), ("Receipt", 25), ("SROIE", 150))
        docs = []
        for dataset, count in counts:
            for i in range(count):
                docs.append({
                    "dataset_name": dataset, "id": str(len(docs)+900),
                    "image_path": f"OCRBench_v2/EN_part/{dataset}/{i}.png",
                    "question": f"Extract {i} from {dataset}",
                })
        write_json(batchdir / "documents.json", docs)
        specs = []
        for n, name in enumerate(finalizer.EXTENDED):
            run_id = f"new-run-{n}"
            folder = runroot / run_id
            folder.mkdir()
            score = 0.9 - n * 0.1
            write_json(folder / "predictions_scored.json",
                       [{**d, "predict": "{}", "score": score} for d in docs])
            write_json(folder / "run.json",
                       {"status": "scored", "batch_id": "new-batch"})
            write_json(folder / "metadata.json",
                       {"gpu_names": ["NVIDIA A100-SXM4-80GB"], "gpu_memory_gib": [79.33]})
            write_json(folder / "metrics.json", {
                "status": "scored", "completed_documents": 400,
                "official_kie_f1": score, "mean_latency_s": 4.5,
                "mean_ocr_time_s": 0.3 if n > 1 else None,
                "mean_lm_time_s": 4.2, "peak_torch_memory_gib": 20.0,
            })
            specs.append({"id": name, "run_id": run_id, "checkpoint": f"test/{name}",
                          "revision": "a"*40, "pipeline": "vlm" if n < 2 else "ocr_lm",
                          "ocr": None if n < 2 else "paddleocr"})
        for engine in finalizer.ENGINES:
            write_json(batchdir / "ocr" / engine / "index.json", {
                "manifest_sha256": "a"*64,
                "records": {str(i): {} for i in range(400)},
                "gpu_hardware": {"name": "NVIDIA A100-SXM4-80GB", "memory_mib": 81000}
            })
        batch = batchdir / "batch.json"
        write_json(batch, {
            "batch_id": "new-batch", "manifest_sha256": "a"*64,
            "gpu_family": "A100",
            "gpu_gres": "gpu:A100:1,gpumem_per_gpu:70GB",
            "min_gpu_memory_gib": 70,
            "runs": specs, "output_root": str(runroot),
        })
        return batch

    def test_fresh_six_model_analysis(self):
        with tempfile.TemporaryDirectory() as temp:
            b = self.make_run(Path(temp))
            out = finalizer.finalize(b, draws=150, seed=2026, figures=False)
            self.assertTrue((out / "six_model_paired_ci.csv").is_file())
            self.assertTrue((out / "primary_model_ci.csv").is_file())
            self.assertTrue((out / "provenance.json").is_file())
            with self.assertRaises(FileExistsError):
                finalizer.finalize(b, draws=150, seed=2026, figures=False)

    def test_fails_on_wrong_variant(self):
        with tempfile.TemporaryDirectory() as temp:
            b = self.make_run(Path(temp))
            root = Path(temp) / "runs"
            meta = root / "new-run-0" / "metadata.json"
            m = json.loads(meta.read_text())
            m["gpu_names"] = ["NVIDIA A100-SXM4-40GB"]
            write_json(meta, m)
            with self.assertRaisesRegex(ValueError, "wrong GPU variant"):
                finalizer.inspect_batch(b)

    def test_fails_on_ocr_hardware_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            b = self.make_run(Path(temp))
            cache = b.parent / "ocr" / "paddleocr" / "index.json"
            data = json.loads(cache.read_text())
            data["gpu_hardware"]["name"] = "NVIDIA A100-SXM4-40GB"
            write_json(cache, data)
            with self.assertRaisesRegex(ValueError, "preprocessing was not verified"):
                finalizer.inspect_batch(b)

    def test_wrong_original_gres_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            b = self.make_run(Path(temp))
            data = json.loads(b.read_text())
            data["gpu_gres"] = "gpu:A100:1"
            write_json(b, data)
            with self.assertRaisesRegex(ValueError, "A100 80GB-class"):
                finalizer.inspect_batch(b)


if __name__ == "__main__":
    unittest.main()
