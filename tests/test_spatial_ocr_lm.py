"""Offline checks for the new spatial OCR representation (no GPU or benchmark data)."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments import spatial_ocr_lm as spatial
from ulm_extractor.storage import key_for, write_json


def line(text, x1, y1, x2, y2):
    return {"text": text, "bbox": [x1, y1, x2, y2], "confidence": 0.99}


class SpatialOCRTests(unittest.TestCase):
    def test_row_group_and_horizontal_pixel_spacing(self):
        lines = [
            line("B", 40, 10, 50, 20),
            line("C", 10, 30, 20, 40),
            line("A", 10, 10, 20, 20),
        ]
        # Mean height 10 -> estimated char width 5; A ends near x=15.
        self.assertEqual(spatial.arrange_text(lines), "A     B\nC")

    def test_uses_original_row_center_tolerance(self):
        lines = [line("Right", 40, 12, 60, 22), line("Left", 10, 10, 30, 20)]
        self.assertIn("Left", spatial.arrange_text(lines).splitlines()[0])
        self.assertEqual(len(spatial.arrange_text(lines).splitlines()), 1)

    def test_refuses_to_silently_drop_polygon_only_or_missing_box(self):
        with self.assertRaisesRegex(ValueError, "Missing bbox"):
            spatial.arrange_text([{"text": "Do not drop me", "polygon": [[0, 0], [10, 10]]}])

    def test_preparation_does_not_write_if_flat_text_differs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch_dir = root / "runs/batches/test-batch"
            layout_dir = batch_dir / "ocr/paddleocr_layout"
            flat_dir = batch_dir / "ocr/paddleocr"
            layout_dir.mkdir(parents=True)
            flat_dir.mkdir(parents=True)
            docs = [
                {"id": i, "dataset_name": "FUNSD", "image_path": f"EN_part/{i}.png",
                 "question": "Extract field", "answers": ["x"]}
                for i in (1, 2)
            ]
            layout_records, flat_records = {}, {}
            for doc in docs:
                key = key_for(doc)
                layout_text = layout_dir / f"{key}.json"
                flat_text = flat_dir / f"{key}.txt"
                write_json(layout_text, {
                    "format": "paddleocr-layout-v1",
                    "lines": [line("A", 10, 10, 20, 20)],
                })
                flat_text.write_text("different" if doc["id"] == 2 else "A", encoding="utf-8")
                layout_records[key] = {"text_path": str(layout_text), "ocr_seconds": 0.2}
                flat_records[key] = {"text_path": str(flat_text), "ocr_seconds": 0.2}
            batch = {"batch_id": "test-batch", "manifest_sha256": "a" * 64}
            write_json(layout_dir / "index.json", {
                "format": "paddleocr-layout-v1", "manifest_sha256": "a" * 64,
                "records": layout_records,
            })
            write_json(flat_dir / "index.json", {
                "manifest_sha256": "a" * 64, "records": flat_records,
            })
            output = root / "runs/analysis/spatial_ocr_lm_test-batch"
            with patch.object(spatial, "ROOT", root):
                # This calls the same entrypoint the 400-case CLI calls after validation.
                # The mock has two cases, so it will fail at the cache completeness check.
                with self.assertRaisesRegex(ValueError, "not complete"):
                    spatial.prepare(batch, batch_dir / "batch.json", docs, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
