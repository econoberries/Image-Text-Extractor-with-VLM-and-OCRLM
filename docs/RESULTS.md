# Verified OCRBench v2 English KIE results

The numbers below are the *observed arithmetic means of the official KIE item scores* on the same 400 documents. They are not a new or unseen external validation dataset.

| Model / analysis label | Official mean KIE score | Documents |
| --- | ---: | ---: |
| Qwen3.5 direct VLM (`qwen35_vlm`) | **0.870673** | 400 |
| Flat PaddleOCR → Qwen3.5 (`qwen35_ocr`) | 0.767842 | 400 |
| Coordinate JSON PaddleOCR → Qwen3.5 (`qwen35_laocr`) | 0.742936 | 400 |
| PP-StructureV3 recovered Markdown → Qwen3.5 (`qwen35_ppstructure_v2`) | 0.722947 | 400 |
| Flat PaddleOCR → Qwen2.5 (`qwen25_ocr`) | 0.655764 | 400 |
| PP-StructureV3 original Markdown → Qwen3.5 (`qwen35_ppstructure_v1`) | 0.590125 | 400 |
| MiniCPM-V-4.6 direct VLM (`minicpm_vlm`) | 0.559314 | 400 |

**Primary contrast:** Qwen3.5 VLM minus flat OCR→Qwen3.5 is **0.102831** (10.283 percentage points). The completed paired 10,000-replicate, fixed-stratum bootstrap gives a 95% percentile interval of approximately **8.36–12.21 percentage points**. On the 400 paired items, direct Qwen scored higher on 160, flat OCR→Qwen on 20, and 220 tied. The interval excludes zero; this is conditional benchmark uncertainty, **not** evidence that this advantage necessarily transfers to different real-world document populations.

**PP implementation repair:** the recovered Markdown pipeline exceeds the historical original Markdown implementation by **0.132822** (13.282 percentage points), with an approximate 95% paired interval of **10.65–15.97 percentage points**. The same source model checkpoint was used, but the OCR reconstruction and runtime cost changed, so this is not a pure layout-only causal ablation.

For **exact** bootstrap confidence bounds, subgroup results and Bonferroni-adjusted intervals, generate the canonical versioned tables:

```bash
python scripts/build_final_results.py --draws 10000 --seed 2026
```

The three comparison panels use distinct and documented seeds: 2026 (primary 4), 2027 (expanded 6), and 2028 (PP v2 versus v1). Generated CSVs and figures are more authoritative for numerical reporting than these rounded README values.

## Compute-efficiency caution

`analysis/efficiency_observations.csv` preserves the originally measured processing and memory summaries with their **actual GPU device names**. These are not controlled matched-GPU performance tests: 40 GB and 80 GB A100s appear across architectures. For OCR→LM, `mean_processing_s` is the *sum of separately timed OCR + LM components*; it does not include Slurm queuing, model downloads, and other batch overhead. `peak_torch_memory_gib` excludes the independently executed PaddlePaddle OCR worker's memory. Do not interpret model load time as steady-state per-document latency.

## Why bootstrap instead of multiple deterministic model runs?

All experiments used `do_sample=False`. Repeating the same 400-item deterministic inference under unchanged settings does not provide independent document observations. We instead resample the **already official-scored matched items** 10,000 times with replacement, preserving subgroup counts, to assess document-sampling uncertainty. This technique does not improve model predictions or create any additional observed documents.

## Limitations for a paper

- The tested domain comprises only 400 OCRBench v2 English KIE items.
- Models differ in checkpoint family, size and preprocessing, so most pairwise comparisons are not single-variable model architecture ablations.
- PP recovered Markdown introduces extra OCR recovery; any improvement reflects a changed pipeline.
- GPU environments and hardware sizes vary, weakening cross-run efficiency interpretation.
- CIs condition on the empirical document distribution and assume resampling of benchmark items is meaningful for the target population.
