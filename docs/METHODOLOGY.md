# Methods and analysis protocol

## Research question and benchmark

We compare direct vision-language models (VLMs) and OCR-to-language-model (OCR→LM) pipelines on **the 400 English key-information-extraction items in OCRBench v2**. The four source collections are FUNSD (48), POIE (177), Receipt (25) and SROIE (150).

The official OCRBench v2 English KIE evaluator supplies a score in [0,1] per question. **The paper's overall KIE result is the arithmetic mean of these 400 official document-level scores**, not token micro-F1 reconstructed from generated strings.

## Models

The frozen experiment registry, [analysis/experiments.csv](../analysis/experiments.csv), identifies all seven historical scored inference runs. The main four are direct Qwen3.5-9B, direct MiniCPM-V-4.6, flat PaddleOCR→Qwen3.5-9B, and flat PaddleOCR→Qwen2.5-14B-Instruct-1M. Secondary analyses add coordinate-aware PaddleOCR→Qwen3.5 and PP-StructureV3→Qwen3.5 using **recovered** Markdown.

There is also a *historical* original PP Markdown result. It used the same `qwen35_ppstructure` inference experiment ID but the older `ppstructurev3-markdown-v1` cache. For publication we explicitly call the historical result `qwen35_ppstructure_v1` and the repaired pipeline `qwen35_ppstructure_v2`. They must never be silently pooled. In PP v2, full-image flat PaddleOCR, PP internal OCR, and region crops are combined to recover text not included in the original parser rendering; this changes both information content and compute. Therefore the PP v1→v2 difference is an **implementation/pipeline repair comparison**, not a pure isolating experiment of layout representation alone.

All archived jobs used the unmodified OCRBench questions, batch size 1, max 384 generated tokens and greedy decoding (`do_sample=False`). Checkpoint revisions are pinned in `src/ulm_extractor/config.py` and recorded in `analysis/experiments.csv`. The original upstream OCRBench evaluator is separate code and should be checked out at commit `4e363b3f499811b43d792ef0bb33932ef05eca27` for a full inference reproduction. The repository itself does not redistribute model weights or OCRBench media/annotations.

## Accuracy and paired bootstrap

Let `s[A,i]` and `s[B,i]` denote the **official KIE item scores** for models A and B on matched question i, i=1,...,400. The observed comparison is

`Delta(A,B) = mean_i(s[A,i] - s[B,i])`.

We use **10,000 stratified paired bootstrap resamples**. Each replicate samples documents *with replacement* separately within the original FUNSD, POIE, Receipt and SROIE strata; the stratum sample counts remain 48/177/25/150. Each model is evaluated using the **same sampled document indices** within that replicate, preserving the pairing. We recompute each model mean and pairwise difference and use the 2.5th and 97.5th percentiles as the 95% intervals.

We report:
- Primary four-model panel: seed 2026, 6 model pairs.
- Expanded six-model panel: seed 2027, 15 model pairs.
- Original versus repaired PP panel: seed 2028, 1 pair.

We apply Bonferroni adjustment across the model-pair family **within each reported scope**. The FUNSD, POIE, Receipt, and SROIE exploratory breakdowns are **not** jointly adjusted for every possible model-by-dataset family. Win/loss/tie counts use item-level score differences with tolerance 1e-12.

**Interpretation:** the uncertainty quantifies sensitivity to the empirical benchmark's observed documents under this resampling model. Reusing the same 400 documents cannot establish out-of-domain generalization, does not create new labeled data, and does not estimate variation caused by probabilistic decoding. The subset counts are fixed by design. Receipt's sample has only 25 documents, so its subgroup uncertainty is particularly high.

## Efficiency metrics and limitations

The saved inference runner records model-load time, generated-token counts, per-document LM processing time, OCR preprocessing time, total *estimated* per-document processing time (OCR + LM), and the PyTorch worker's peak allocated GPU memory. For OCR→LM workers OCR and LM execute in separate Slurm jobs, so the *sum* of their individual measured times is a per-item component estimate, not a directly observed serial full-pipeline wall clock with scheduling and I/O overhead.

Peak Torch memory does **not** include the separate PaddlePaddle OCR worker's peak allocation. A100 40 GB and A100 80 GB GPUs were mixed across archived experiments, which limits the fairness of latency/memory comparisons. We report each hardware variant in [analysis/efficiency_observations.csv](../analysis/efficiency_observations.csv) and avoid presenting these numbers as controlled hardware benchmarking or energy consumption.

## What was independently checked

The original score archive was previously checked for seven scored runs, 400 unique matched items each, the same manifest hash, successful scoring states and exact agreement of document means with official batch-level values. For publication the compact, lossless derivative score panel is checked on every CI run; optional `--archive` additionally verifies the original JSON score file SHA256 values, item identities, and PP cache format signatures.

To reproduce **all** published tables and figures from the compact score panel (CPU only), run:

```bash
python scripts/validate_final_results.py
python scripts/build_final_results.py --draws 10000 --seed 2026
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for environment setup and the stronger raw-archive verification.
