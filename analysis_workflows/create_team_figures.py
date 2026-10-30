#!/usr/bin/env python3
"""Create shareable research figures from completed OCRBench v2 CSV outputs only.

Run from repository root. No inference, scoring, bootstrap or existing result is changed.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = [
    'qwen35_vlm', 'qwen35_ocr', 'qwen35_laocr',
    'qwen35_ppstructure', 'qwen25_ocr', 'minicpm_vlm'
]
LABEL = {
    'qwen35_vlm': 'Qwen3.5-9B · direct VLM',
    'qwen35_ocr': 'Qwen3.5-9B · plain OCR',
    'qwen35_laocr': 'Qwen3.5-9B · layout JSON',
    'qwen35_ppstructure': 'Qwen3.5-9B · PP-Structure',
    'qwen25_ocr': 'Qwen2.5-14B · plain OCR',
    'minicpm_vlm': 'MiniCPM-V-4.6 · direct VLM',
}


def save(fig, directory, stem):
    fig.savefig(directory / (stem + '.png'), dpi=210, bbox_inches='tight')
    plt.close(fig)
    print('Saved', directory / (stem + '.png'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stat-dir', type=Path, default=Path('runs/analysis/kie_scores_0to1'))
    p.add_argument('--hyp-dir', type=Path, default=Path('runs/analysis/hypotheses_400'))
    p.add_argument('--output-dir', type=Path, default=Path('runs/analysis/team_figures'))
    args = p.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        p.error(f'Output folder already contains files; refusing overwrite: {out}')

    ci = pd.read_csv(args.stat_dir / 'six_model_ci.csv')
    pairs = pd.read_csv(args.hyp_dir / 'hypothesis_pairwise_descriptive.csv')
    efficiency = pd.read_csv(args.hyp_dir / 'efficiency_descriptive.csv')
    errors = pd.read_csv(args.hyp_dir / 'model_error_summary.csv')
    q = pd.read_csv(args.hyp_dir / 'kie_400_questions.csv')
    if len(q) != 400 or len(set(q['document_key'])) != 400:
        raise ValueError('Question-level data must contain 400 unique documents')
    if set(ci['experiment_id']) != set(MODELS):
        raise ValueError('Expected all six models in bootstrap table')

    out.mkdir(parents=True, exist_ok=True)

    overall = ci[ci['dataset'].eq('ALL')].set_index('experiment_id').loc[MODELS]
    y = np.arange(len(MODELS))
    values = overall['official_mean_score'].to_numpy()
    low = overall['ci_low'].to_numpy()
    high = overall['ci_high'].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 5.1))
    ax.errorbar(values, y, xerr=np.vstack([values - low, high - values]),
                fmt='o', capsize=4, linewidth=1.5)
    ax.set_yticks(y, [LABEL[m] for m in MODELS])
    ax.invert_yaxis()
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel('Mean official KIE item score (0–1)')
    ax.set_title('Accuracy across six configurations · 95% bootstrap CIs')
    ax.grid(axis='x', alpha=.22)
    for i, val in enumerate(values):
        ax.annotate(f'{val:.3f}', (val, i), xytext=(8, 5),
                    textcoords='offset points', fontsize=9)
    fig.tight_layout()
    save(fig, out, '01_accuracy_bootstrap')

    datasets = ['FUNSD', 'POIE', 'SROIE', 'Receipt']
    mat = np.array([
        ci[ci['dataset'].eq(ds)].set_index('experiment_id').loc[MODELS, 'official_mean_score'].to_numpy()
        for ds in datasets
    ])
    fig, ax = plt.subplots(figsize=(11, 4.0))
    im = ax.imshow(mat, aspect='auto', vmin=0, vmax=1)
    ax.set_yticks(np.arange(4), [f'{ds} (n={n})' for ds, n in zip(datasets, (48, 177, 150, 25))])
    ax.set_xticks(np.arange(6), [LABEL[m].replace(' · ', '\n') for m in MODELS])
    ax.tick_params(axis='x', labelsize=8)
    for i in range(4):
        for j in range(6):
            ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center',
                    color='white' if mat[i, j] < .5 else 'black', fontsize=10)
    ax.set_title('KIE accuracy by document dataset')
    fig.colorbar(im, ax=ax, label='Mean official KIE item score (0–1)', pad=.01)
    fig.tight_layout()
    save(fig, out, '02_accuracy_by_dataset')

    comparisons = [
        ('H1: 9B OCR vs 14B OCR', 'H1'),
        ('H2: direct VLM vs plain OCR', 'H2'),
        ('Plain OCR vs layout JSON', 'Layout_LaOCR'),
        ('Plain OCR vs PP-Structure', 'Layout_PP')
    ]
    p_all = pairs[pairs['dataset'].eq('ALL')].set_index('hypothesis')
    rows = [p_all.loc[key] for _, key in comparisons]
    pos = np.arange(len(rows))
    wins = np.array([r['a_higher'] for r in rows])
    ties = np.array([r['tied'] for r in rows])
    losses = np.array([r['b_higher'] for r in rows])
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    first = ax.barh(pos, wins, label='First approach higher')
    middle = ax.barh(pos, ties, left=wins, label='Equal score')
    last = ax.barh(pos, losses, left=wins + ties, label='Second approach higher')
    for bars, nums in [(first, wins), (middle, ties), (last, losses)]:
        for patch, n in zip(bars, nums):
            if n > 14:
                ax.text(patch.get_x() + patch.get_width() / 2,
                        patch.get_y() + patch.get_height() / 2,
                        str(int(n)), ha='center', va='center', fontsize=9)
    ax.set_yticks(pos, [label for label, _ in comparisons])
    ax.invert_yaxis()
    ax.set_xlim(0, 400)
    ax.set_xlabel('Number of questions (out of 400)')
    ax.set_title('Question-level paired comparisons')
    ax.legend(loc='lower center', bbox_to_anchor=(.5, -0.3), ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, out, '03_question_wins_ties_losses')

    score_by_model = overall['official_mean_score']
    eff = efficiency.set_index('model').loc[MODELS]
    for xcol, xlabel, filename, title in [
        ('mean_processing_s', 'Mean component processing time (s/question)',
         '04_accuracy_vs_processing_time', 'Accuracy versus measured processing time'),
        ('peak_torch_gib', 'Peak PyTorch allocated GPU memory (GiB)',
         '05_accuracy_vs_torch_memory', 'Accuracy versus allocated GPU memory'),
    ]:
        fig, ax = plt.subplots(figsize=(9.7, 5.0))
        x = eff[xcol].to_numpy()
        yscore = score_by_model.to_numpy()
        ax.scatter(x, yscore, s=65)
        for model, xx, yy in zip(MODELS, x, yscore):
            ax.annotate(LABEL[model], (xx, yy), xytext=(7, 5),
                        textcoords='offset points', fontsize=8.2)
        ax.margins(x=.25, y=.22)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Mean official KIE item score (0–1)')
        ax.set_title(title)
        ax.grid(alpha=.22)
        fig.tight_layout()
        save(fig, out, filename)

    e_all = errors[errors['dataset'].eq('ALL')].set_index('model').loc[MODELS]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    xs = np.arange(len(MODELS))
    values = e_all['token_limit_hits'].to_numpy()
    ax.bar(xs, values)
    ax.set_xticks(xs, [LABEL[m].replace(' · ', '\n') for m in MODELS], fontsize=8)
    for i, v in enumerate(values):
        ax.text(i, v + .25, str(int(v)), ha='center', fontsize=9)
    ax.set_ylabel('Questions hitting 384-token generation limit')
    ax.set_title('Generation-limit events across 400 questions')
    ax.set_ylim(0, max(values) * 1.2 + 1)
    fig.tight_layout()
    save(fig, out, '06_token_limit_hits')

    note = '''Research visualization package — OCRBench v2 English KIE (400 questions)\n\n01: Overall mean official KIE item scores with unadjusted 95% bootstrap model CIs.\n02: Accuracy broken down by FUNSD, POIE, SROIE and Receipt.\n03: Paired question wins/ties/losses (descriptive, not significance tests).\n04: Accuracy versus sum of separately measured OCR and LM processing times.\n05: Accuracy versus peak Torch allocated memory (excludes Paddle OCR memory).\n06: Observed hits at the 384-new-token generation cap.\n\nStatistics and official scoring were not recomputed by this script.\nFor paired inference, use the existing six_model_paired_ci.csv.\nThe qualitative case sample was purposively selected and is not representative.\n'''
    (out / 'README.txt').write_text(note, encoding='utf-8')
    print('Saved', out / 'README.txt')


if __name__ == '__main__':
    main()
