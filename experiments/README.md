# Seventh experiment: bounding-box spatial OCR → Qwen3.5-9B

**This is an optional follow-up experiment, separate from the original six-model benchmark.** It uses the same 400 OCRBench v2 English KIE questions, the same pinned Qwen3.5-9B model configuration as qwen35_ocr, and the same official OCRBench evaluator. It does not edit or rerun the original six model pipelines.

The goal is to test whether adding approximate document layout through spaces and line breaks helps the language model associate fields with values.

## Method

1. Reuse the saved LaOCR paddleocr_layout cache for all 400 questions. There is no fresh PaddleOCR inference.
2. Confirm its recognized text and original line order match the saved plain PaddleOCR cache, question by question. If not, stop rather than confound OCR recognition with layout differences.
3. Calculate the vertical center of each recognized line. Group lines whose centers are within half the average box height, updating row centers as detections are added.
4. Within each row, sort by the left x-coordinate. Estimate a character width as half the average box height. Insert spaces based on pixel gaps, and separate rows with newlines.
5. Feed this *plain text* plus the unchanged OCRBench question to the existing Qwen3.5-9B InferenceAdapter. Greedy decoding, max_new_tokens=384, and the original pinned checkpoint remain unchanged.
6. Apply the official scorer to the 400 new predictions. Bootstrap matched scores against three previously scored Qwen3.5 OCR-based methods: plain OCR, layout JSON and PP-StructureV3 Markdown.

The row/spacing calculation follows the supplied ocr_output.py arrange_text() algorithm. **Safety improvement:** if any recognized text has no rectangular bounding box, the experiment fails rather than silently dropping it.

This reuses our saved OCR recognition. It does not recreate the supplied script's separate PaddleOCR run, CPU/RSS metrics, image-folder traversal, or filename conventions.

## Run on Helix

From the repository root on the dedicated experiment branch:

### 1. Check and prepare the saved OCR documents (CPU)

~~~bash
BATCH="runs/batches/cbdd58de-531a-4c9c-9778-5bc8b0b0be48/batch.json"
.venv/bin/python experiments/spatial_ocr_lm.py prepare --batch "$BATCH"
~~~

This writes NEW outputs only under:

runs/analysis/spatial_ocr_lm_cbdd58de-531a-4c9c-9778-5bc8b0b0be48/

The command refuses to overwrite an existing experiment directory. Do not rerun prepare after it succeeds.

### 2. Submit ONE new Qwen3.5 inference job (A100 80GB)

~~~bash
sbatch --parsable \
  --partition gpu-single \
  --gres "gpu:A100:1,gpumem_per_gpu:70GB" \
  --cpus-per-task 4 \
  --mem 64G \
  --time 06:00:00 \
  --job-name ulm-spatial-ocr \
  --output "runs/analysis/spatial_ocr_lm_cbdd58de-531a-4c9c-9778-5bc8b0b0be48/spatial-%j.log" \
  slurm/spatial_ocr_lm.sbatch "$BATCH"
~~~

The experiment checks the allocated GPU class before loading Qwen and refuses to run inference outside a Slurm allocation. It writes fresh predictions.json and metadata.json. An interrupted inference can resume without regenerating saved predictions by submitting the same job after inspecting the failure.

Use squeue -u "$USER" to monitor the job. Wait for successful completion before scoring.

### 3. Official scoring and comparison (CPU)

~~~bash
.venv/bin/python experiments/spatial_ocr_lm.py score --batch "$BATCH"
~~~

This creates:
- predictions_scored.json — official KIE scores for spatial OCR–LM
- comparison/comparison_summary.csv — overall means, wins/ties/losses
- comparison/model_bootstrap_ci.csv — overall and dataset-specific 95% model CIs
- comparison/paired_bootstrap_ci.csv — paired CIs and Bonferroni familywise CIs across the four compared OCR–LM configurations
- comparison/paired_document_scores.csv — four models aligned to the same 400 questions
- comparison/methodology.json — method and caveats

The script does not update the original six-model batch or six-model bootstrap outputs.

## Timing caveat

This experiment reuses saved LaOCR OCR duration for each document, measures spatial text formatting on CPU, and measures new LM runtime in the Slurm job. Combined time is a sum of separately measured components, not full end-to-end wall time. Torch memory omits Paddle OCR allocations.

## Interpretation

This seventh exploratory configuration is a representation ablation. The model, original question, decoding settings and recognized text are held fixed relative to Qwen3.5 plain OCR. Spatial reconstruction differs from coordinate JSON in input encoding. PP-Structure also changes OCR recognition and recovery.

All new outputs live under Git-ignored runs/analysis/. Do not commit or publicly upload the benchmark images, OCR caches, or scored predictions.
