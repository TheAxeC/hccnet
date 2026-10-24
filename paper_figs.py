'''Regenerate the two composite manuscript figures from the fine-tune/baseline prediction
npz (offline, no retrain). The original figure code was not committed (see test.py note in
_cumulative_gain_mae), so these figures were previously not reproducible; this script closes
that gap and uses the SAME per-fold-ensemble statistic as gen_tables.py so the figure numbers
match the result tables exactly.

Per-fold ensemble = 50 models as 5 folds x 10 seeds; fold(row k) = k % 5; average the 10 seeds'
predicted probabilities within a fold, compute the metric on that ensemble, and report the mean
across the 5 folds (the ECE printed in the calibration legend equals the reliability table). All
50 models predict on the same fixed test set, so the per-fold ensembles share the label vector.

Outputs (written next to this script under a --out dir):
  calibration.png     - 3 modality rows x 4 variant columns; baseline vs fine-tuned reliability
                        bars per confidence bin, with the per-fold-ensemble ECE in each legend.
  cumulative_gain.png - 2 metric rows (AUPRC, AUROC) x 3 modality columns; one cumulative-gain
                        curve per variant over the 5 folds (ordered ascending) vs the diagonal,
                        with the fold-robustness MAE (Eq. in Section 4) in each legend.

The figures show all five variants (femto, pico, nano, tiny, compact) by default, matching the
five-architecture result tables. Pass --no-compact for the original four-variant layout.
'''
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score, average_precision_score

K = 5
NBINS = 10
PRED = os.environ.get('HCC_PRED_DIR', '/private/tmp/claude-502/scratch_pred')

MODS = [('dwi', 'DW-MRI'), ('t1dce', 'T1-DCE MRI'), ('t1iop', 'T1-IOP & T2 MRI')]
ARCHS4 = [('femto', 'HCCNet-Femto'), ('pico', 'HCCNet-Pico'),
          ('nano', 'HCCNet-Nano'), ('tiny', 'HCCNet-Tiny')]
COMPACT = ('compact', 'HCCNet-Compact')

FT_COLOR = '#1f1fd6'      # fine-tuned (blue)
BASE_COLOR = '#20b2aa'    # baseline (light sea green)
OPT_COLOR = '#f7b6bc'     # optimal-calibration triangle (pink)
# cumulative-gain line colours per variant, in ARCHS4 (+compact) order
CG_COLORS = ['#ffb0b8', '#1f1fd6', '#17becf', '#9c9b1e', '#7a5c00']


def foldsets(n):
    return [[k for k in range(n) if k % K == c] for c in range(K)]


def load(stem):
    f = os.path.join(PRED, f'{stem}.npz')
    if not os.path.isfile(f):
        return None
    d = np.load(f, allow_pickle=True)
    return np.asarray(d['probs'], float), np.asarray(d['labels']).ravel().astype(float)


def perfold_ensembles(probs, labels):
    '''Return the 5 per-fold ensemble probability vectors (each length N).'''
    return [probs[fold].mean(0) for fold in foldsets(probs.shape[0])]


def ece_of(p, y, nbins=NBINS):
    edges = np.linspace(0.0, 1.0, nbins + 1)
    ece, n = 0.0, len(p)
    for b in range(nbins):
        m = (p > edges[b]) & (p <= edges[b + 1])
        if b == 0:
            m |= (p == edges[b])
        if m.any():
            ece += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return ece


def perfold_mean_ece(probs, labels):
    '''Mean over folds of the per-fold-ensemble ECE (equals the reliability table).'''
    return float(np.mean([ece_of(pe, labels) for pe in perfold_ensembles(probs, labels)]))


def binned_frequency(probs, labels, nbins=NBINS):
    '''Observed frequency per confidence bin over the pooled per-fold ensembles.'''
    pooled_p = np.concatenate(perfold_ensembles(probs, labels))
    pooled_y = np.concatenate([labels for _ in range(K)])
    edges = np.linspace(0.0, 1.0, nbins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    freq = np.full(nbins, np.nan)
    for b in range(nbins):
        m = (pooled_p > edges[b]) & (pooled_p <= edges[b + 1])
        if b == 0:
            m |= (pooled_p == edges[b])
        if m.any():
            freq[b] = pooled_y[m].mean()
    return centers, freq


def perfold_metric(probs, labels, fn):
    return np.array([fn(labels, pe) for pe in perfold_ensembles(probs, labels)])


def cumulative_gain(per_fold):
    m = np.sort(np.asarray(per_fold, float))
    n = len(m)
    exp = np.arange(1, n + 1) / n
    if m.sum() == 0:
        return np.zeros(n), exp, 0.0
    obs = np.cumsum(m) / m.sum()
    return obs, exp, float(np.mean(np.abs(obs - exp)))


def make_calibration(archs, out_path):
    nrows, ncols = len(MODS), len(archs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)
    width = 1.0 / NBINS
    for r, (mst, ml) in enumerate(MODS):
        for c, (ast, al) in enumerate(archs):
            ax = axes[r][c]
            # optimal-calibration lower triangle + diagonal
            ax.fill_between([0, 1], [0, 1], color=OPT_COLOR, hatch='//',
                            edgecolor='#e79aa0', linewidth=0.0, zorder=0)
            ax.plot([0, 1], [0, 1], '--', color='0.5', linewidth=1.2, zorder=1)
            handles = []
            for stem_suf, col, name in [('paper_base', BASE_COLOR, 'base'),
                                        ('paper_ft', FT_COLOR, 'ft')]:
                res = load(f'{mst}_{ast}_{stem_suf}')
                if res is None:
                    continue
                probs, labels = res
                centers, freq = binned_frequency(probs, labels)
                ece = perfold_mean_ece(probs, labels)
                z = 3 if name == 'ft' else 2
                ax.bar(centers, np.nan_to_num(freq), width=width, color=col,
                       edgecolor='black', linewidth=0.5, alpha=0.85, zorder=z)
                handles.append(Patch(facecolor=col, edgecolor='black',
                                     label=f'ECE = {100 * ece:.1f}%'))
            ax.legend(handles=handles[::-1], fontsize=8, loc='upper left', framealpha=0.9)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            if r == 0:
                ax.set_title(al, fontsize=12)
            if c == 0:
                ax.set_ylabel(ml, fontsize=11)
    fig.supxlabel('Confidence', fontsize=14)
    fig.supylabel('Accuracy', fontsize=14)
    legend_items = [Patch(facecolor=OPT_COLOR, hatch='//', edgecolor='#e79aa0', label='Optimal Calibration'),
                    Patch(facecolor=FT_COLOR, edgecolor='black', label='Fine-Tuned Model'),
                    Patch(facecolor=BASE_COLOR, edgecolor='black', label='Baseline Model')]
    fig.legend(handles=legend_items, loc='lower center', ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0.02, 0.03, 1, 1))
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')


def make_cumulative_gain(archs, out_path):
    rows = [('AUPRC', average_precision_score), ('AUROC', roc_auc_score)]
    fig, axes = plt.subplots(2, len(MODS), figsize=(4.6 * len(MODS), 4.2 * 2), squeeze=False)
    for r, (mname, fn) in enumerate(rows):
        for c, (mst, ml) in enumerate(MODS):
            ax = axes[r][c]
            ax.plot([0, 1], [0, 1], '--', color='0.5', linewidth=1.6, zorder=1)
            for i, (ast, al) in enumerate(archs):
                res = load(f'{mst}_{ast}_paper_ft')
                if res is None:
                    continue
                probs, labels = res
                pf = perfold_metric(probs, labels, fn)
                obs, _, mae = cumulative_gain(pf)
                x = np.arange(1, len(obs) + 1) / len(obs)
                ax.plot(x, obs, marker='o', markersize=3, linewidth=1.8,
                        color=CG_COLORS[i], label=f'MAE = {100 * mae:.1f}%', zorder=2)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
            if r == 0:
                ax.set_title(ml, fontsize=12)
            ax.set_ylabel(f'Normalized {mname}', fontsize=11)
    fig.supylabel('Cumulative Gain', fontsize=14)
    fig.tight_layout(rect=(0.02, 0.10, 1, 1))
    fig.supxlabel('Cross-validation fold (ordered ascending)', fontsize=14, y=0.075)
    legend_items = [Line2D([0], [0], linestyle='--', color='0.5', label='Optimal Robustness')]
    legend_items += [Line2D([0], [0], color=CG_COLORS[i], linewidth=2, label=al)
                     for i, (ast, al) in enumerate(archs)]
    fig.legend(handles=legend_items, loc='lower center', ncol=len(archs) + 1, fontsize=11,
               bbox_to_anchor=(0.5, 0.0))
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='.', help='output directory for the two PNGs')
    ap.add_argument('--compact', action=argparse.BooleanOptionalAction, default=True,
                    help='include HCCNet-Compact as the fifth variant (default on; --no-compact for four)')
    ap.add_argument('--selfcheck', action='store_true',
                    help='print per-config per-fold-ensemble ECE for table cross-check')
    args = ap.parse_args()
    archs = ARCHS4 + ([COMPACT] if args.compact else [])
    os.makedirs(args.out, exist_ok=True)

    if args.selfcheck:
        print('per-fold-ensemble mean ECE (should equal the reliability table):')
        for mst, ml in MODS:
            for ast, al in ARCHS4 + [COMPACT]:
                for suf in ('paper_base', 'paper_ft'):
                    res = load(f'{mst}_{ast}_{suf}')
                    if res is not None:
                        print(f'  {ml:16s} {al:15s} {suf:10s} ECE {perfold_mean_ece(*res):.3f}')
        return

    make_calibration(archs, os.path.join(args.out, 'calibration.png'))
    make_cumulative_gain(archs, os.path.join(args.out, 'cumulative_gain.png'))


if __name__ == '__main__':
    main()
