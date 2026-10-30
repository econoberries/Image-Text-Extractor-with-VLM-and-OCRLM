# Vision-Language Models (VLM) vs. OCR-LM Pipelines for Key Information Extraction

**OCRBench v2 English Key Information Extraction (KIE)**

This repository contains the code for our course project, which investigates whether structured information extraction from images can be performed more efficiently, in terms of time and computational cost, with low-resource open-source vision-language models (VLMs) or with pipelines that combine optical character recognition (OCR) and a language model (OCR-LM). Both approaches are evaluated on the same 400 English KIE questions from OCRBench v2 and compared on extraction accuracy, processing time and GPU memory. We test three hypotheses: (1) larger models achieve higher accuracy, (2) VLM and OCR-LM approaches do not differ significantly in accuracy, and (3) OCR-LM pipelines have lower computational costs.

This repository is an integrated version of our original development repository [econoberries/ULM-VLM-OCR-IE](https://github.com/econoberries/ULM-VLM-OCR-IE), where we first developed and tested the individual pipelines. It documents our working process and preliminary runs.

To compare all approaches fairly, we brought the pipelines together here in a single evaluation framework. All six pipelines run in one batch on the same 400 English OCRBench v2 questions, with the same unmodified questions, GPU type, and official OCRBench v2 scorer. In this process, some implementation details were standardized across pipelines. The results reported in this repository are therefore our final results, while the preliminary runs in the development repository used earlier settings.

## The six setups

Every setup answers the **original, unchanged** OCRBench question. No model is trained or fine-tuned, and the exact model versions are pinned in `src/ulm_extractor/config.py`.

| ID | Model input | Model |
| --- | --- | --- |
| `qwen35_vlm` | Document image + question | Qwen3.5-9B |
| `minicpm_vlm` | Document image + question | MiniCPM-V-4.6 |
| `qwen35_ocr` | Flat PaddleOCR text + question | Qwen3.5-9B |
| `qwen25_ocr` | Flat PaddleOCR text + question | Qwen2.5-14B-Instruct-1M |
| `qwen35_laocr` | PaddleOCR text **with bounding boxes** (JSON) + question | Qwen3.5-9B |
| `qwen35_ppstructure` | PP-StructureV3 Markdown with OCR text recovery + question | Qwen3.5-9B |

The **main comparison** is `qwen35_vlm` vs. `qwen35_ocr`, which use the same model with different input. The other setups change the model or the kind of OCR input.

An **optional seventh comparison**, Spatial OCR–LM (`qwen35_spatial_ocr`), is kept separate from the original six-model benchmark. See the [spatial experiment instructions](https://github.com/econoberries/Image-Text-Extractor-with-VLM-and-OCRLM/blob/experiment/spatial-ocr-lm/experiments/README.md) to reuse the saved OCR caches, produce its official scores, and compare them with the six scored models on the same 400 questions.

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

- [OCRBench v2](https://github.com/Yuliang-Liu/MultimodalOCR)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- [MiniCPM-V 4.6](https://github.com/OpenBMB/MiniCPM-V)
- [Qwen2.5-14B-Instruct-1M](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-1M)
