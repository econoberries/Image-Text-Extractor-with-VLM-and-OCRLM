# Comparing Vision-Language Models and OCR → Language-Model Pipelines

**OCRBench v2 English Key Information Extraction (KIE)**

Is it better to give a document image directly to a vision-language model (VLM), or to first extract the text with OCR and then ask a language model (LM) the same question? We run both approaches on the same 400 documents and compare extraction accuracy, processing time and compute requirements.


## The six setups

Every setup answers the **original, unchanged** OCRBench question. No model is trained or fine-tuned; checkpoints are pinned in `src/ulm_extractor/config.py`.

| ID | Model input | Model |
| --- | --- | --- |
| `qwen35_vlm` | Document image + question | Qwen3.5-9B |
| `minicpm_vlm` | Document image + question | MiniCPM-V-4.6 |
| `qwen35_ocr` | Flat PaddleOCR text + question | Qwen3.5-9B |
| `qwen25_ocr` | Flat PaddleOCR text + question | Qwen2.5-14B-Instruct-1M |
| `qwen35_laocr` | PaddleOCR text **with bounding boxes** (JSON) + question | Qwen3.5-9B |
| `qwen35_ppstructure` | PP-StructureV3 Markdown with OCR text recovery + question | Qwen3.5-9B |

The **main comparison** is `qwen35_vlm` vs. `qwen35_ocr`: same model, different input. The other setups change the model or the kind of OCR input.

## Dataset and evaluation

- **Data:** OCRBench v2, task `key information extraction en`: 400 questions (FUNSD 48, POIE 177, Receipt 25, SROIE 150). The dataset is **not included** in this repository; download it from [OCRBench v2](https://github.com/Yuliang-Liu/MultimodalOCR/blob/main/OCRBench_v2/README.md).
- **Scoring:** the unmodified official evaluator (`eval_scripts/eval.py`, commit `4e363b3`). We report the mean of the 400 per-item KIE scores, overall and per subset.
- **Decoding:** greedy, `max_new_tokens=384`, batch size 1.
- **Statistics:** a paired, stratified bootstrap over documents (10,000 resamples) gives 95% confidence intervals for each model and for every pairwise difference. Models are not re-run during the bootstrap.
- **Efficiency:** per-item OCR and LM time, model load time, output tokens, and peak PyTorch GPU memory. All GPU jobs run on the same GPU class (A100 80 GB).

## Repository layout

```text
├── model.yml               ← All six setups (full benchmark)
├── model-*.yml             ← Smaller subsets
├── src/ulm_extractor/      ← Package and `extractor` CLI (dataset, OCR, models, scheduling, scoring)
├── scripts/                ← Bootstrap, final analysis, dependency checks
├── slurm/                  ← GPU worker, CPU scoring collector, smoke tests
├── requirements/           ← Pinned CUDA/Paddle/analysis requirements
└── tests/                  ← Offline tests (run in GitHub CI)
```

Generated data (`data/`, `runs/`, `.venv*`) is not committed to Git.

## Quick start

The experiment runs on a Slurm GPU cluster with Python 3.12. PyTorch and PaddlePaddle conflict over CUDA libraries, so each gets its own environment: `.venv` for the LMs and scoring, plus `.venv-ocr`, `.venv-la-ocr` and `.venv-ppstructure` for the three OCR pipelines. Exact install commands are in [README_full.md §5](README_full.md#5-set-up-the-project-on-helix).

```bash
# 1. Check environments and run the offline tests
.venv/bin/python scripts/check_dependencies.py --stage all
.venv/bin/python -m unittest discover -s tests -v

# 2. Preview the jobs (3 OCR + 6 model + 1 scoring), then run without --dry-run
.venv/bin/extractor benchmark --input "$EN_PART" --models model.yml \
  --partition gpu-single --collector-partition cpu-single \
  --gres "gpu:A100:1,gpumem_per_gpu:70GB" --gpu-family A100 --min-gpu-memory-gib 70 \
  --cpus 4 --mem 64G --time 06:00:00 --dry-run

# 3. Resume a failed batch (keeps the same batch/run IDs)
.venv/bin/extractor resume --batch runs/batches/<BATCH_ID>/batch.json

# 4. After all six runs are scored: confidence intervals, tables, figures
.venv-analysis/bin/python scripts/finalize_benchmark.py \
  --batch runs/batches/<BATCH_ID>/batch.json --n-bootstrap 10000 --seed 2026
```

## Where the results are

| Path | Contents |
| --- | --- |
| `runs/batches/<BATCH_ID>/benchmark_summary.csv` | **Start here:** one row per model with scores, timing and hardware |
| `runs/<RUN_ID>/` | Each model's predictions, official per-item scores and metrics |
| `runs/analysis/<BATCH_ID>/` | Bootstrap CIs (`six_model_ci.csv`, `six_model_paired_ci.csv`), `efficiency.csv`, figures, `provenance.json` |

For **when each output is generated**, how to find your batch and run IDs, and where to inspect the files, see [README_full.md — When each result is generated](README_full.md#when-each-result-is-generated). The full guide also covers archiving and downloading the results to your computer.

## Limitations

- Only 400 English KIE questions; results may not carry over to other document types.
- Some comparisons change both the input pipeline and the model family.
- OCR and LM time are measured in separate jobs, so the totals are component times, not end-to-end wall-clock time.
- GPU memory figures cover PyTorch allocations only, not Paddle.
- PP-Structure with text recovery is evaluated as one complete pipeline, not as a layout-only ablation.

## References

- [OCRBench v2: data, code and official evaluator](https://github.com/Yuliang-Liu/MultimodalOCR) · [paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/8c2e6bb15be1894b8fb4e0f9bcad1739-Abstract-Datasets_and_Benchmarks_Track.html)
- [PaddleOCR documentation](https://www.paddleocr.ai/latest/en/index.html)



