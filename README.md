# Comparing Vision-Language Models and OCR → Language-Model Pipelines

**A University of Tübingen student project · OCRBench v2 English Key Information Extraction (KIE)**

We wanted to find out whether it is better to give a document image directly to a vision-language model (VLM), or to first extract the text using OCR and then ask a language model (LM) to answer the same question. We built several versions of both approaches, ran them on the same documents, and compared extraction accuracy, processing time and compute requirements.

This repository contains **the code to run a six-model benchmark on bwForCluster Helix**, evaluate predictions with the official OCRBench v2 scorer, and calculate statistical comparisons. These instructions assume a first-time setup, with measured results to be added after the benchmark.

**Start here:** The short roadmap below shows the whole workflow. The numbered sections afterward explain each step, including the commands, metrics, statistical comparisons and troubleshooting. Folder READMEs are only brief navigation aids.

## How to reproduce our experiment — the full workflow at a glance

This is the order to follow on a **fresh Helix installation**. You do not need any files or results from another project. Each step links to the detailed instructions below.

| Step | What to do | Where to find the commands |
| --- | --- | --- |
| **1. Clone** | Log in to Helix, check Git / Slurm / Python 3.12, and clone this repository. | [Setup, §5](#5-set-up-the-project-on-helix) |
| **2. Install** | Create four separate Python environments: one for Torch/LMs and three for PaddleOCR-based pipelines. | [Setup, §5](#5-set-up-the-project-on-helix) |
| **3. Prepare the benchmark** | Download the official OCRBench v2 images and annotations, then clone the pinned official evaluator. These are **not provided in this repository**. | [Setup, §5](#5-set-up-the-project-on-helix) |
| **4. Check dependencies** | Verify the four environments, run offline tests, and confirm that the loader can select the 400 English KIE questions. | [Validation, §6](#6-check-gpu-allocation-and-dependencies) |
| **5. Check GPU execution** | Test the A100 80 GB-class Slurm request; run the Torch, Paddle and single-image/model smoke tests. | [Validation, §6](#6-check-gpu-allocation-and-dependencies) |
| **6. Preview the experiment** | Use `extractor benchmark --dry-run` with `model.yml` to check the **3 OCR + 6 model + 1 collector** jobs without submitting them. | [Execution, §7](#7-run-all-six-models) |
| **7. Submit once** | Run the same benchmark command without `--dry-run`. Save the printed batch ID and Slurm job IDs. | [Execution, §7](#7-run-all-six-models) |
| **8. Check and score** | Monitor jobs and inspect the batch-specific `benchmark_summary.csv`. The CPU collector applies the official KIE scorer; resume or rescore only if needed. | [Outputs, §8](#8-monitor-recover-and-score) |
| **9. Compare the models** | After all six runs have 400 scored documents, run `finalize_benchmark.py` to produce paired-bootstrap confidence intervals, efficiency tables and figures. | [Metrics, §9](#9-how-we-measure-accuracy-and-efficiency), [bootstrap, §10](#10-paired-bootstrap-statistical-comparisons), [analysis, §11](#11-finalize-the-fresh-benchmark) |
| **10. Preserve the results** | Check the generated tables, figures, original predictions, checksums and configuration; archive the exact code commit and batch. | [Final checks, §11](#11-finalize-the-fresh-benchmark) |

**In short:** clone → install → obtain data/evaluator → validate → GPU smoke test → dry run → submit → score → bootstrap → archive.

The full benchmark produces **six model run folders**, one batch summary, and a separate final analysis folder. All generated data, model downloads and results stay outside Git; the repository contains the implementation and reproduction instructions.

## Contents

1. [What we are testing](#1-what-we-are-testing)
2. [What is in this repository?](#2-what-is-in-this-repository)
3. [Dataset and scoring](#3-dataset-and-scoring)
4. [Models and experiment settings](#4-models-and-experiment-settings)
5. [Set up the project on Helix](#5-set-up-the-project-on-helix)
6. [Check GPU allocation and dependencies](#6-check-gpu-allocation-and-dependencies)
7. [Run all six models](#7-run-all-six-models)
8. [Monitor, recover and score](#8-monitor-recover-and-score)
9. [How we measure accuracy and efficiency](#9-how-we-measure-accuracy-and-efficiency)
10. [Paired bootstrap: statistical comparisons](#10-paired-bootstrap-statistical-comparisons)
11. [Finalize the fresh benchmark](#11-finalize-the-fresh-benchmark)
12. [Troubleshooting](#12-troubleshooting)
13. [Reproducibility, limitations and references](#13-reproducibility-limitations-and-references)

## 1. What we are testing

We compare the following **six** setups. They all answer the **original, unchanged** OCRBench question for each document.

| ID | What the model receives | Model |
| --- | --- | --- |
| `qwen35_vlm` | Original document image + question | Qwen3.5-9B |
| `minicpm_vlm` | Original document image + question | MiniCPM-V-4.6 |
| `qwen35_ocr` | Flat PaddleOCR text + question | Qwen3.5-9B |
| `qwen25_ocr` | Flat PaddleOCR text + question | Qwen2.5-14B-Instruct-1M |
| `qwen35_laocr` | PaddleOCR recognized text **with bounding-box coordinates** + question | Qwen3.5-9B |
| `qwen35_ppstructure` | PP-StructureV3 Markdown **with OCR text recovery** + question | Qwen3.5-9B |

Our **main comparison** is direct Qwen3.5 versus flat PaddleOCR → the *same* Qwen3.5 checkpoint. This comes closest to comparing the effect of the input pipeline while keeping the model fixed. Comparisons with MiniCPM and Qwen2.5 also change the underlying model.

For the OCR pipelines, the LM never receives the original image. Flat OCR preserves recognized lines but does not explicitly retain positions; coordinate-aware OCR serializes each line, its pixel bounding box and confidence into JSON; PP-StructureV3 produces document-structure Markdown and recovers missing OCR text. The PP-Structure model in this benchmark uses **recovered Markdown** as its text representation.

We did **not** train or fine-tune models. We use pretrained checkpoints and document their exact model revisions in `src/ulm_extractor/config.py`.

## 2. What is in this repository?

```text
Image-Text-Extractor-with-VLM-and-OCRLM/
├── README.md                      ← Full guide: you are here
├── CITATION.cff                   ← Project citation metadata
├── .github/workflows/tests.yml   ← Offline tests on every GitHub push
├── pyproject.toml                 ← Python package, dependencies and CLI
├── .python-version               ← Python version
│
├── model.yml                     ← All six models (full benchmark)
├── model-baselines.yml           ← Primary four-model comparison subset
├── model-laocr.yml               ← Only coordinate-aware OCR → Qwen
├── model-ppstructure.yml         ← Only recovered PP Markdown → Qwen
│
├── src/ulm_extractor/            ← Main Python source code
├── scripts/                      ← Bootstrap, final analysis, diagnostic scripts
├── slurm/                        ← GPU worker, CPU collector, GPU smoke tests
├── requirements/                 ← Python/CUDA installation requirements
├── tests/                        ← Offline regression and integrity tests
└── analysis/                     ← Brief analysis/navigation notes
```

**Important files and their jobs:**

| File | Purpose |
| --- | --- |
| `src/ulm_extractor/config.py` | Checks YAML models, supported pipelines, and frozen checkpoint revisions |
| `dataset.py` | Selects and validates 400 English KIE questions and images |
| `cli.py` | Provides `extractor benchmark` and `extractor resume` |
| `scheduler.py` | Submits OCR jobs, model jobs and CPU scoring with Slurm dependencies |
| `worker.py` | Runs inference and saves one prediction per question |
| `adapters.py` | Loads the model, formats its input, generates and decodes answers |
| `ocr.py`, `layout.py` | Flat OCR and coordinate-preserving JSON OCR |
| `ppstructure.py`, `markdown_recovery.py` | Document parser and text-recovery procedure |
| `aggregate.py` | Calls the official evaluator and builds per-run accuracy/efficiency summaries |
| `resume.py` | Safely retries failed work without changing batch/run IDs |
| `storage.py` | Atomic JSON writes and document identity hashing |
| `scripts/bootstrap_kie.py` | Document-paired, stratified bootstrap analysis |
| `scripts/finalize_benchmark.py` | **After all new runs are scored:** hardware/score checks, statistics, tables and figures |
| `slurm/worker.sbatch` | Routes Torch and Paddle jobs to separate environments and validates GPU VRAM |
| `slurm/collector.sbatch` | CPU scoring and summary, without rerunning models |
| `slurm/check_environment.sbatch` | Single-job tests for GPU, OCR and model inference |

The package uses `src/` layout, so the `src/ulm_extractor/*.py` files are **not** scripts you usually run by hand. Use the `extractor` CLI and the `scripts/` entry points.

**Generated directories do not belong in Git:**

- `data/`: optional local dataset files; the official benchmark must be obtained separately.
- `runs/`: new run UUIDs, predictions, OCR caches, Slurm logs, scoring results and final statistical analysis.
- `.venv*`: separate Python environments.

These paths are ignored by `.gitignore`. Do not delete `runs/` to fix a failed job: our resume procedure reuses its partial results.

## 3. Dataset and scoring

We use **OCRBench v2**, restricted to the task `key information extraction en`.

| Source subset | Questions |
| --- | ---: |
| FUNSD | 48 |
| POIE | 177 |
| Receipt | 25 |
| SROIE | 150 |
| **Total** | **400** |

The evaluation input comes from `OCRBench_v2.json`; the images are in `OCRBench_v2/EN_part/`. The loader checks the exact task type, images, identities and expected count, and writes a **dataset manifest SHA256** in each batch. All six new experiments should have *the same manifest hash* because we submit them in a single batch.

**Ground truth stays out of inference.** The model receives the OCRBench question and either the image or a text representation. The official answers are available only to the scoring procedure.

We use the upstream evaluator `MultimodalOCR/OCRBench_v2/eval_scripts/eval.py`, unmodified. Its KIE item-level scores appear as `score` in `predictions_scored.json`. Our collector reports **the arithmetic mean of the 400 official KIE item scores**, plus the mean separately within FUNSD, POIE, Receipt and SROIE. This is not a new token-level micro-F1 we compute ourselves.

We pin the reference evaluator repository to commit `4e363b3f499811b43d792ef0bb33932ef05eca27`. Record the actual evaluator commit if you intentionally use a different revision.

## 4. Models and experiment settings

`model.yml` specifies all six experiments. We use:

| Parameter | Setting |
| --- | --- |
| Prompt | Original benchmark-native question; no modified instructions |
| Batch size | 1 image/question at a time |
| `max_new_tokens` | 384 |
| `do_sample` | `false` (greedy decoding) |
| Model checkpoints | Pinned immutable 40-character revision IDs |
| New benchmark scope | All 400 English KIE items × all six models |
| Model run strategy | One complete inference run per architecture |

The model loader and chat processor live in `adapters.py`. Token-limit hits are recorded because truncation can affect extraction scores. We do not repeatedly generate new answers with sampling to estimate accuracy uncertainty; the paired bootstrap later operates on the saved **document scores**.

The two flat OCR → LM models share one cached OCR extraction, so OCR is not needlessly recomputed. Coordinate JSON and PP Markdown have independent caches. The recovered PP pipeline also runs English PaddleOCR over the image, can recognize PP image crops, and appends OCR lines missing from the base Markdown without inventing table cells. As a result, it is a more expensive *complete pipeline*, not a clean layout-only ablation.

## 5. Set up the project on Helix

Sections 5–8 assume a fresh installation on the **Helix login node**. Submit GPU workloads through Slurm using `sbatch`; do **not** execute GPU inference directly on a login node.

### Step 5.1 — Check Helix access and clone the project

Connect to Helix using your university account and the cluster's documented SSH login procedure. In the **Helix terminal**, check that Git, Slurm and Python 3.12 are available:

```bash
command -v git
command -v sbatch
/usr/bin/python3.12 --version
```

Clone this repository into your home directory:

```bash
cd "$HOME"
git clone https://github.com/econoberries/Image-Text-Extractor-with-VLM-and-OCRLM.git
cd Image-Text-Extractor-with-VLM-and-OCRLM
git rev-parse HEAD
```

Save the displayed commit SHA to identify the exact code used for the experiment. All the following commands assume you are in this repository root.

### Step 5.2 — Create the separate Python environments

We use Python **3.12**, with **four isolated environments** to avoid known CUDA library conflicts between PyTorch and PaddlePaddle:

| Environment | Used for |
| --- | --- |
| `.venv` | CLI, Qwen/MiniCPM inference and official scoring |
| `.venv-ocr` | Flat PaddleOCR |
| `.venv-la-ocr` | Coordinate-aware PaddleOCR |
| `.venv-ppstructure` | PP-StructureV3 and OCR text recovery |

**Install Torch/Transformers:**

```bash
test -x .venv/bin/python || /usr/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements/helix-pytorch-cu130.txt
.venv/bin/python -m pip install -e '.[vlm]'
.venv/bin/python -m pip install -r requirements/ocrbench-v2-kie-eval.txt
```

**Install flat PaddleOCR separately:**

```bash
test -x .venv-ocr/bin/python || /usr/bin/python3.12 -m venv .venv-ocr
.venv-ocr/bin/python -m pip install -e '.[ocr]'
.venv-ocr/bin/python -m pip install -r requirements/helix-paddle-cu130.txt
```

**Install coordinate OCR separately:**

```bash
test -x .venv-la-ocr/bin/python || /usr/bin/python3.12 -m venv .venv-la-ocr
.venv-la-ocr/bin/python -m pip install -e '.[ocr]'
.venv-la-ocr/bin/python -m pip install -r requirements/helix-paddle-cu130.txt
```

**Install PP-StructureV3 separately:**

```bash
test -x .venv-ppstructure/bin/python || /usr/bin/python3.12 -m venv .venv-ppstructure
.venv-ppstructure/bin/python -m pip install -e '.[ppstructure]'
.venv-ppstructure/bin/python -m pip install -r requirements/helix-paddle-cu130.txt
```

The Paddle GPU wheel is installed *last* within each Paddle environment. **Never install `paddlepaddle-gpu` into `.venv`.** We install CPU statistical-analysis dependencies (NumPy and matplotlib) into a separate `.venv-analysis` later or, for convenience, into the existing `.venv` after inference has finished.

### Step 5.3 — Download OCRBench v2 and clone the official evaluator

The benchmark dataset is **not included** in this GitHub repository. Download it from the authors' [OCRBench v2 dataset instructions](https://github.com/Yuliang-Liu/MultimodalOCR/blob/main/OCRBench_v2/README.md), extract the archive, and upload the required data to Helix using your institution's supported file-transfer method. **This project uses only the English data:** you need `EN_part/` and the original `OCRBench_v2.json`; the Chinese image directory `CN_part/` is not used and does not need to be uploaded.

The required directory structure is:

```text
$HOME/OCRBench_v2/
├── EN_part/
└── OCRBench_v2.json
```

Back in the **Helix terminal**, set the path and check that the files exist:

```bash
export EN_PART="$HOME/OCRBench_v2/EN_part"
test -d "$EN_PART" && echo "English images found"
test -f "$HOME/OCRBench_v2/OCRBench_v2.json" && echo "Annotations found"
```

`--input` must point to the **`EN_part` directory**, not its parent. The loader will verify the 400 English KIE items. Keep the official annotations and images unchanged.

Separately, clone the source code of the **official evaluator**. Cloning this repository does not download the benchmark images:

```bash
git clone https://github.com/Yuliang-Liu/MultimodalOCR.git "$HOME/MultimodalOCR"
git -C "$HOME/MultimodalOCR" checkout 4e363b3f499811b43d792ef0bb33932ef05eca27
test -f "$HOME/MultimodalOCR/OCRBench_v2/eval_scripts/eval.py" \
  && echo "Official evaluator found"
```

Both Git repositories are installed under `$HOME`, matching the evaluator path expected by the benchmark CLI.

## 6. Check GPU allocation and dependencies

### Why we insist on the same A100 variant

To make processing-time comparisons fairer, we request **one A100 with at least 70 GB GPU memory for every GPU job**—both model inference and OCR preprocessing. Requesting only an A100 without specifying VRAM could allow different A100 variants across jobs.

Helix's A100 nodes have 40 GB and 80 GB configurations. The Slurm request

```bash
--gres="gpu:A100:1,gpumem_per_gpu:70GB"
```

restricts allocation to the **A100 80 GB class**. The pipeline then checks the actual GPU **name and memory** at runtime and fails rather than silently switching to a 40 GB device. Helix documents `gpumem_per_gpu` as a GPU-memory selection filter. This is separate from `--mem=64G`, which reserves **system RAM**, not VRAM.

We still cannot make identical A100 nodes guarantee identical runtime behavior (concurrent load, caches and driver differences can matter), but standardizing the GPU variant removes an important source of confounding.

### Step 6.1 — Check dependencies, with no GPU needed

```bash
.venv/bin/python -m pip check
.venv-ocr/bin/python -m pip check
.venv-la-ocr/bin/python -m pip check
.venv-ppstructure/bin/python -m pip check

.venv/bin/python scripts/check_dependencies.py --stage all
.venv/bin/python -m unittest discover -s tests -v
```

On Helix login nodes, `libcuda.so.1` may be unavailable. A deferred Paddle GPU import in the dependency checker **does not establish a GPU failure**; verify it on an allocated GPU.

### Step 6.2 — Preview the 80 GB-class Slurm request

```bash
export GPU_GRES="gpu:A100:1,gpumem_per_gpu:70GB"
export GPU_EXPECT=A100
export GPU_MEMORY_MIB=71680

sinfo -N -o '%N %G' | grep A100
sbatch --test-only --partition=gpu-single --gres="$GPU_GRES" \
  slurm/check_environment.sbatch torch
```

The `--test-only` command checks schedulability without launching a job. If Helix rejects the combined typed-GPU/memory GRES syntax, inspect the current cluster configuration or contact Helix support **before launching a full benchmark**; don't simply remove the VRAM restriction.

### Step 6.3 — Run the four environment GPU checks

```bash
sbatch --gres="$GPU_GRES" \
  --export="ALL,EXTRACTOR_EXPECT_GPU=A100,EXTRACTOR_MIN_GPU_MEMORY_MIB=$GPU_MEMORY_MIB" \
  slurm/check_environment.sbatch torch

sbatch --gres="$GPU_GRES" \
  --export="ALL,EXTRACTOR_EXPECT_GPU=A100,EXTRACTOR_MIN_GPU_MEMORY_MIB=$GPU_MEMORY_MIB" \
  slurm/check_environment.sbatch paddle

sbatch --gres="$GPU_GRES" \
  --export="ALL,EXTRACTOR_EXPECT_GPU=A100,EXTRACTOR_MIN_GPU_MEMORY_MIB=$GPU_MEMORY_MIB" \
  slurm/check_environment.sbatch la-paddle

sbatch --gres="$GPU_GRES" \
  --export="ALL,EXTRACTOR_EXPECT_GPU=A100,EXTRACTOR_MIN_GPU_MEMORY_MIB=$GPU_MEMORY_MIB" \
  slurm/check_environment.sbatch pp-paddle
```

Look in `gpu-check-<JOB_ID>.out` for **GPU memory check passed**, and for **PyTorch CUDA: PASSED** or **PaddlePaddle CUDA: PASSED**. The logs should show an actual `A100-SXM4-80GB` device.

### Step 6.4 — Try a single example document

```bash
IMAGE="$(find "$EN_PART" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print -quit)"
test -n "$IMAGE"

sbatch --gres="$GPU_GRES" --mem=64G --time=00:45:00 \
  --export="ALL,EXTRACTOR_EXPECT_GPU=A100,EXTRACTOR_MIN_GPU_MEMORY_MIB=$GPU_MEMORY_MIB" \
  slurm/check_environment.sbatch ocr "$IMAGE"

sbatch --gres="$GPU_GRES" --mem=64G --time=00:45:00 \
  --export="ALL,EXTRACTOR_EXPECT_GPU=A100,EXTRACTOR_MIN_GPU_MEMORY_MIB=$GPU_MEMORY_MIB" \
  slurm/check_environment.sbatch la-ocr "$IMAGE"

sbatch --gres="$GPU_GRES" --mem=64G --time=00:45:00 \
  --export="ALL,EXTRACTOR_EXPECT_GPU=A100,EXTRACTOR_MIN_GPU_MEMORY_MIB=$GPU_MEMORY_MIB" \
  slurm/check_environment.sbatch ppstructure "$IMAGE"

sbatch --gres="$GPU_GRES" --mem=64G --time=00:45:00 \
  --export="ALL,EXTRACTOR_EXPECT_GPU=A100,EXTRACTOR_MIN_GPU_MEMORY_MIB=$GPU_MEMORY_MIB" \
  slurm/check_environment.sbatch model "$EN_PART" qwen35_vlm
```

These are **smoke tests**, not full evaluations; the model smoke test uses a reduced token budget. They do not produce publication benchmark scores.

## 7. Run all six models

### Step 7.1 — Check the complete Slurm submission first

Run this from the **repository root** after the smoke tests pass:

```bash
.venv/bin/extractor benchmark \
  --input "$EN_PART" \
  --models model.yml \
  --partition gpu-single \
  --collector-partition cpu-single \
  --gres "$GPU_GRES" \
  --gpu-family A100 \
  --min-gpu-memory-gib 70 \
  --cpus 4 --mem 64G --time 06:00:00 \
  --dry-run
```

`--dry-run` validates the dataset and six model settings, prints the planned Slurm jobs and dependencies, but **does not submit jobs or write results**.

### Step 7.2 — Submit the real benchmark

When the dry-run and GPU smoke tests pass, run the **same command without `--dry-run`**:

```bash
.venv/bin/extractor benchmark \
  --input "$EN_PART" \
  --models model.yml \
  --partition gpu-single \
  --collector-partition cpu-single \
  --gres "$GPU_GRES" \
  --gpu-family A100 \
  --min-gpu-memory-gib 70 \
  --cpus 4 --mem 64G --time 06:00:00
```

**This command submits real jobs and consumes cluster resources. Run it only once for the intended fresh batch.** The time limit can be changed if permitted by the partition and needed after checking earlier runtimes.

One complete six-model batch schedules:

1. Flat PaddleOCR worker (shared by two LM models).
2. Coordinate-aware PaddleOCR worker.
3. PP-StructureV3 Markdown + recovery worker.
4. Six independent language-model workers, each depending on its matching OCR worker where applicable.
5. One CPU-only collector that evaluates all completed predictions and writes the batch summary.

That is **10 Slurm jobs**. Dependencies are `afterany` so even failed jobs can be diagnosed; a submitted LM may still fail if its OCR dependency did not produce a complete cache.

The scheduler prints a **new batch UUID**. Save it:

```bash
BATCH="$PWD/runs/batches/<NEW_BATCH_UUID>/batch.json"
```

Replace `<NEW_BATCH_UUID>` with the actual identifier. Do not copy UUIDs from the older research repository.

## 8. Monitor, recover and score

### Where outputs are stored

```text
runs/
├── batches/<batch-id>/
│   ├── batch.json                ← Exact experiment settings, GPU request and hashes
│   ├── documents.json            ← Fixed 400 question manifest
│   ├── jobs.json                 ← Initial Slurm job IDs
│   ├── logs/                     ← Job logs
│   ├── ocr/
│   │   ├── paddleocr/            ← Flat recognized text + measured OCR time
│   │   ├── paddleocr_layout/     ← Coordinate JSON + measured OCR time
│   │   └── ppstructurev3/        ← Recovered Markdown, raw parser audit and timing
│   └── benchmark_summary.csv     ← Authoritative six-model summary
├── <run-uuid>/
│   ├── run.json                  ← submitted / running / failed / scored
│   ├── metadata.json             ← GPU name, total VRAM, model version/load time
│   ├── predictions.json          ← 400 predictions + timing + errors
│   ├── predictions_scored.json   ← Official per-item KIE scores
│   └── metrics.json              ← Per-run accuracy + efficiency metrics
├── analysis/<batch-id>/          ← Created by finalize_benchmark.py (after scoring)
└── benchmark_summary.csv         ← Convenience copy from most recent collector
```

### Step 8.1 — Watch Slurm

```bash
squeue -u "$USER"
find runs/batches -name benchmark_summary.csv -print
cat "runs/batches/<NEW_BATCH_UUID>/benchmark_summary.csv"
```

For an individual job:

```bash
sacct -j <SLURM_JOB_ID> --format=JobID,State,ExitCode,Elapsed
cat "runs/batches/<NEW_BATCH_UUID>/logs/<JOB_LOG_FILENAME>.out"
```

Check that each row in the **batch-specific** summary has `status=scored` and `completed_documents=400`. The top-level `runs/benchmark_summary.csv` can be replaced by subsequent batches, so it is **not** the source of truth.

### If submitting the batch stops partway through

Slurm accepts jobs one at a time, so a submission error does not imply earlier jobs were rejected. The batch folder now contains an incremental `jobs.json` ledger and a `submission.json` receipt. A receipt marked `partial_failure` means the original submission did not finish. Review the saved job IDs in those files and check the queue/accounting status before starting another batch. The scheduler does not automatically cancel jobs. The resume command rejects a batch that has not finished its initial submission.

### Step 8.2 — Recover a failed or interrupted batch

First verify that no old job for this batch is still pending or running:

```bash
squeue -u "$USER"
```

Then preview a resume:

```bash
.venv/bin/extractor resume --batch "$BATCH" --dry-run
```

After confirming the plan, submit the actual recovery:

```bash
.venv/bin/extractor resume --batch "$BATCH"
```

This keeps the **same original batch/run IDs**, resumes failed predictions, and preserves the batch's recorded GPU GRES and minimum VRAM requirement. Never start a brand-new `benchmark` to fix just one incomplete run. The `--ocr-only` option is available for deliberately excluding all direct VLM reruns.

### Step 8.3 — If inference succeeded but scoring failed

The official scorer runs in a CPU collector. You can rescore **without rerunning the GPU models**:

```bash
sbatch --partition=cpu-single --cpus-per-task=2 --mem=4G \
  --time=00:30:00 --job-name=ulm-rescore \
  slurm/collector.sbatch collect --batch "$BATCH"
```

Inspect each `scoring.log` if the evaluator raises an exception. Do not change the OCRBench evaluator to inflate or "fix" model scores.

## 9. How we measure accuracy and efficiency

The scoring scripts use saved model outputs from `predictions.json` and call the **official** `eval.py` to produce `predictions_scored.json`. The collector averages the resulting item-level scores.

| Reported metric | What it means | Important qualification |
| --- | --- | --- |
| **Official mean KIE score** | Mean official per-item score across 400 items | Not independently reconstructed token micro-F1 |
| **FUNSD/POIE/Receipt/SROIE mean** | Mean within each source subset | Receipt has only 25 items |
| **Mean/median/p95 processing seconds** | Distribution of measured per-item processing time | OCR + LM components were timed in different jobs |
| **Mean OCR seconds** | Paddle parsing/recognition time | Shared OCR cache can serve two LM runs |
| **Mean LM seconds** | Processing, token generation and decoding | Excludes initial model loading and result-file write |
| **Model load seconds** | Initialization of each language model | Includes filesystem/model-cache effects |
| **Output tokens / token-limit hits** | Generated text length and truncation frequency | All models have a 384-token maximum |
| **Peak Torch allocated GiB** | Maximum allocated GPU memory during LM inference | **Does not measure separate Paddle GPU memory** |
| **GPU name and available VRAM** | Actual physical accelerator for LM/OCR jobs | Checked separately for preprocessing and inference |

`inference_seconds` for an OCR pipeline is `ocr_seconds + lm_seconds`: an **estimated total component processing time**, not end-to-end elapsed wall time including Slurm scheduling, repeated cold starts and I/O. We should not call these controlled complete-system wall-clock measurements.

We standardize on A100 80 GB to improve comparability, but different model checkpoints and OCR architectures still use resources differently. The comparisons remain descriptive: we do not estimate energy consumption, and available GPU VRAM is not the same as peak GPU memory used.

## 10. Paired bootstrap: statistical comparisons

We already know the observed score on **these exact 400 documents** once inference and scoring finish. What we want to learn is how stable a model's apparent advantage is when we vary which *observed* benchmark documents get sampled.

The bootstrap does **not** rerun the neural networks.

1. We start with the 400 final item scores from each model, aligned by **the same document ID, image path and question**.
2. Within each dataset subset we sample documents **with replacement**, preserving the original counts 48 FUNSD / 177 POIE / 25 Receipt / 150 SROIE.
3. Every model uses the **same resampled document indices** (paired bootstrap).
4. We calculate each model's mean score and all pairwise differences for that resample.
5. We repeat this **10,000 times** using fixed seeds, then form percentile **95% confidence intervals**.

For models A and B, the paired comparison estimates:

```text
mean_difference(A,B)
    = mean over i=1..400 of [official_score(A,i) - official_score(B,i)]
```

We report both the unadjusted interval and a conservative **Bonferroni-adjusted familywise interval** across all pairwise comparisons within a given model panel and dataset scope.

We use two panels:

- **Primary:** original four architectures, six model pairs, seed 2026.
- **Expanded:** all six architectures, 15 model pairs, seed 2027.

The study compares only the **six pipelines in `model.yml`**. PP-Structure with text recovery is treated as one complete pipeline.

A bootstrap resample is **not a new document**, and 10,000 resamples are not 10,000 independent observations. The intervals quantify document-resampling uncertainty conditional on OCRBench's benchmark composition. They **do not establish** out-of-domain generalization or GPU/nondeterministic decoding variance.

The reusable implementation lives in `scripts/bootstrap_kie.py`; the single-batch finalizer below calls it automatically.

## 11. Finalize the fresh benchmark

**Only do this once all six jobs are officially scored and all the OCR indexes are complete.**

### Step 11.1 — Install CPU plotting dependencies

For example, create a lightweight CPU-analysis environment in the same checkout:

```bash
test -x .venv-analysis/bin/python || /usr/bin/python3.12 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r requirements/analysis-lock.txt
.venv-analysis/bin/python -m pip install -e .
```

The CPU-only analysis needs no PyTorch, PaddlePaddle or GPU allocation.

### Step 11.2 — Validate hardware and generate final results

```bash
.venv-analysis/bin/python scripts/finalize_benchmark.py \
  --batch "$BATCH" \
  --n-bootstrap 10000 \
  --seed 2026
```

The finalizer checks:

- Exactly the **six expected models**, scored on 400 matched records each.
- Dataset manifest agreement and original official mean-score agreement.
- Each LM worker used one **A100 80 GB** GPU with at least 70 GiB available.
- All **three OCR preprocessing caches** were also produced on the A100 80 GB class.
- All model runs were scored successfully.
- The output folder does not already contain an analysis that could be overwritten.

**Outputs** go to `runs/analysis/<NEW_BATCH_UUID>/`:

| File | Purpose |
| --- | --- |
| `primary_model_ci.csv` | Four primary model means and confidence intervals |
| `primary_paired_ci.csv` | Six paired four-model comparisons |
| `six_model_ci.csv` | Six model means and confidence intervals |
| `six_model_paired_ci.csv` | Fifteen six-model comparisons |
| `document_scores.csv` | The aligned 400-document score table |
| `efficiency.csv` | Official score, timing, GPU and model metadata per run |
| `provenance.json` | Model/run IDs, file checksums, GPU allocations and bootstrap parameters |
| `figure_primary.png`, `figure_six_models.png` | Accuracy plots with 95% intervals |

The script is deliberately **fail-closed**: if one GPU is wrong, scores are incomplete, or the run has a different manifest, we stop rather than publish misleading "final" numbers.

### Step 11.3 — Freeze what we actually ran

Before submission, keep the final `batch.json`, `documents.json`, per-run `predictions_scored.json`, `metrics.json`, OCR index files, the final analysis folder, and `git rev-parse HEAD` in a protected archive. Keep original predictions and intermediate OCR outputs when permissible. Avoid pushing OCRBench images/annotations into GitHub without reviewing the benchmark's redistribution terms.

After the figures and tables are checked against the paper, we can publish the measured outputs and tag the released code version. **Benchmark results are currently pending.**

## 12. Troubleshooting

| Symptom | What to check / do |
| --- | --- |
| Slurm rejects `gpumem_per_gpu` request | Verify `sinfo` and the live cluster GRES rules; use `sbatch --test-only`. Do not fall back silently to unrestricted A100 |
| Job lands on A100 40 GB | The worker will exit before inference because its VRAM check fails. Confirm the submitted `--gres` and exported minimum threshold |
| `libcuda.so.1` missing on login node | Test on a GPU-allocated Slurm node, not the login node |
| Torch/Paddle dependency conflict | Keep `.venv` and the three Paddle virtual environments separate; use the Torch repair requirements only when needed |
| `torchvision` or `paddleocr` missing | Reinstall into the matching environment, not system Python |
| CUDA/Paddle smoke test fails | Inspect `gpu-check-<id>.out`, GPU VRAM, `pip check`, PyTorch/Paddle versions and driver |
| OCR cache `index.json` missing | Read the separate OCR job log; LM jobs depend on it |
| PP Markdown is empty | Inspect PP parser job, recovered Markdown, raw JSON and audit output |
| Worker is `failed` | Inspect `runs/<run-id>/predictions.json` and logs, then preview `extractor resume` |
| Worker is `submitted` or `running` | Check `squeue` and `sacct` first; don't submit a competing duplicate worker |
| All predictions exist but scoring failed | Install scorer dependencies and rerun only the CPU collector |
| Saved bootstrap batch mixes models or documents | Select the single newly completed batch rather than old exploratory batches |
| `finalize_benchmark.py` reports hardware mismatch | Check both LM `metadata.json` and each OCR `index.json`; don't suppress the validation |
| A100 80 GB queue is long | Consider more wall-time headroom or waiting for allocation; changing to H200/40 GB would constitute a different hardware design |
| Output folder already exists | The finalizer protects existing results; preserve the folder and use a new output directory only for intentionally separate analysis |

## 13. Reproducibility, limitations and references

Our goal is for someone else to understand **what was measured**, **how it was measured**, and **how to repeat it**. Our code versions and model revisions are pinned; every new batch records its document manifest, model/run UUIDs and GPU request, and the worker records its actual allocated GPU. The finalizer keeps score and hardware provenance next to the statistical results.

Passing GitHub's **offline tests** proves that configuration, orchestration and analysis code satisfy the tested contracts. It does **not** execute Qwen/MiniCPM on a GPU. For genuine inference reproduction, the Helix smoke tests and fresh Slurm run described above are still required.

Important limitations remain even after standardizing hardware: the tested corpus contains only 400 English KIE questions; the OCR algorithms and model families differ across some arms; OCR/LM timing components are measured in separate jobs; GPU memory is PyTorch-only for LM allocations; and the bootstrap inference does not demonstrate performance in unseen document domains.

**External sources**

- [OCRBench v2 data, code and official evaluator](https://github.com/Yuliang-Liu/MultimodalOCR)
- [OCRBench v2 paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/8c2e6bb15be1894b8fb4e0f9bcad1739-Abstract-Datasets_and_Benchmarks_Track.html)
- [Helix GPU hardware](https://wiki.bwhpc.de/e/Helix/Hardware)
- [Helix Slurm GPU and memory requests](https://wiki.bwhpc.de/e/Helix/Slurm)
- [PaddleOCR documentation](https://www.paddleocr.ai/latest/en/index.html)

**Project team:** Khaira Abdillah, Hengyi Liu, and Madeline Miller. See `CITATION.cff` for citation metadata. The team should agree on code licensing before publishing a reuse license; upstream datasets, models and dependencies retain their own licenses.


