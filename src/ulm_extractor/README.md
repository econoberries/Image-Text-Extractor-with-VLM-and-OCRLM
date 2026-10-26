# Core Python package

The code implementing our experiment lives here. The key flow is `config.py` / `dataset.py` → `scheduler.py` → `ocr.py` / `ppstructure.py` → `adapters.py` / `worker.py` → `aggregate.py`. `resume.py` repairs interrupted jobs; `gpu_hardware.py` records actual preprocessing hardware.

Use the `extractor` CLI rather than invoking package files directly. See the **root README** for complete commands, metrics and explanations.
