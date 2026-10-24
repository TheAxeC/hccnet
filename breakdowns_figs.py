'''Meeting-quality figures from the HCCNet finetune predictions (offline, no retrain).

Reuses the per-fold ensemble machinery of breakdowns.py (fold(row k)=k%5; ensemble 10 seeds within
a fold; mean +/- std across the 5 folds). Produces:
  1. leadtime_bands.png  - AUROC vs prediction lead time (years) with +/- std bands + sample sizes
  2. etiology_bars.png   - AUROC by etiology (NASH / ASH; underpowered "Other" dropped) with error bars
  3. overview_panel.png  - both, side by side, for one slide
Delta is in years; sex/age unavailable; "Other" etiology has 1 positive so it is excluded here.
'''
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

K = 5
_HERE = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.environ.get('HCC_PRED_DIR', '/private/tmp/claude-502/scratch_pred')
LABELS_CSV = os.environ.get('HCC_LABELS_CSV', os.path.normpath(os.path.join(
    _HERE, os.pardir, 'notes', 'data', 'The moset recent labels-June 2024.csv')))
OUT = os.environ.get('HCC_BREAKDOWN_OUT', '/private/tmp/claude-502/breakdowns')
os.makedirs(OUT, exist_ok=True)
from sklearn.metrics import roc_auc_score

# Okabe-Ito colorblind-safe palette, one colour per modality (consistent across figures).
MODS = [('DW-MRI', 'dwi_pico_paper_ft', '#0072B2'),
        ('T1 DCE-MRI', 't1dce_pico_paper_ft', '#E69F00'),
        ('T1 IOP alone', 't1iop_only_pico_paper_ft', '#009E73'),
        ('T2 alone', 't2_only_pico_paper_ft', '#D55E00')]
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.25, 'figure.dpi': 150})


def etiology_map():
    m = {}
    with open(LABELS_CSV, newline='') as f:
        for r in csv.DictReader(f):
            nash = int(float(r.get('nash', 0) or 0)); ash = int(float(r.get('alcoholic_cirrosis', 0) or 0))
            m[r['id']] = 'DROP' if (nash and ash) else 'NASH' if nash else 'ASH' if ash else 'Other'
    return m


def load(stem):
    d = np.load(os.path.join(PRED_DIR, f'{stem}.npz'), allow_pickle=True)
    pid = np.array([str(u).rsplit('_', 1)[0] for u in d['uids']])
    return dict(probs=np.asarray(d['probs'], float), labels=np.asarray(d['labels']).ravel(),
                pid=pid, pfx=np.asarray(d['prefix_probs'], float), pdl=np.asarray(d['prefix_deltas'], float))


def perfold(cube, labels, idx):
    y = labels[idx]
    if len(idx) < 3 or len(set(y.tolist())) < 2:
        return None
    v = [roc_auc_score(y, cube[[k for k in range(cube.shape[0]) if k % K == c]][:, idx].mean(0)) for c in range(K)]
    return float(np.mean(v)), float(np.std(v))


def horizon(m, Y):
    pfx, pdl = m['pfx'], m['pdl']
    out = np.full(pfx.shape[:2], np.nan); keep = []
    for p in range(pfx.shape[1]):
        ok = np.where((~np.isnan(pdl[p])) & (pdl[p] >= Y))[0]
        if len(ok):
            out[:, p] = pfx[:, p, ok[pdl[p][ok].argmin()]]; keep.append(p)
    return out, np.array(keep, int)


def leadtime_ax(ax, data, years):
    for name, _, col in MODS:
        m, s, ns = [], [], []
        for Y in years:
            hc, keep = horizon(data[name], Y); v = perfold(hc, data[name]['labels'], keep)
            m.append(v[0] if v else np.nan); s.append(v[1] if v else np.nan); ns.append(len(keep))
        m, s = np.array(m), np.array(s)
        lw = 3.0 if name == 'DW-MRI' else 1.8
        ax.plot(years, m, marker='o', color=col, lw=lw, label=name, zorder=3)
        ax.fill_between(years, m - s, m + s, color=col, alpha=0.15, zorder=1)
    ax.axhline(0.5, ls='--', color='grey', lw=1)
    ax.set_xticks(years)
    ax.set_xticklabels([f'{Y}\nn={n}' for Y, n in zip(years, ns)])   # ns from last (all share denominator)
    ax.set_xlabel('prediction lead time  (scans $\\geq$ Y years before diagnosis)')
    ax.set_ylabel('AUROC  (per-fold ensemble $\\pm$ std)')
    ax.set_ylim(0.35, 1.0)
    ax.legend(loc='lower left', frameon=False)


def etiology_ax(ax, data):
    strata = ['NASH', 'ASH']; et = etiology_map()
    x = np.arange(len(strata)); w = 0.2
    pos_counts = {}
    for i, (name, _, col) in enumerate(MODS):
        m = data[name]; grp = np.array([et.get(p, 'Other') for p in m['pid']])
        mu, sd = [], []
        for s in strata:
            idx = np.where(grp == s)[0]; v = perfold(m['probs'], m['labels'], idx)
            mu.append(v[0] if v else np.nan); sd.append(v[1] if v else 0)
            pos_counts[s] = int(m['labels'][idx].sum())
        ax.bar(x + i * w, mu, w, yerr=sd, color=col, label=name, capsize=3, error_kw=dict(lw=1))
    ax.axhline(0.5, ls='--', color='grey', lw=1)
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels([f'{s}\n({pos_counts[s]} HCC+)' for s in strata])
    ax.set_ylabel('AUROC  (per-fold ensemble $\\pm$ std)'); ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', frameon=False, fontsize=9)


def main():
    data = {name: load(stem) for name, stem, _ in MODS}
    years = list(range(1, 8))

    fig, ax = plt.subplots(figsize=(8, 5.2)); leadtime_ax(ax, data, years)
    ax.set_title('HCCNet predicts HCC years in advance')
    fig.text(0.5, -0.13, 'Test set: 58 patients, 8 HCC+  |  DW-MRI and T2 hold AUROC ~0.9 up to 6-7 years out',
             ha='center', fontsize=9, color='#444')
    fig.savefig(os.path.join(OUT, 'leadtime_bands.png'), bbox_inches='tight'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.2)); etiology_ax(ax, data)
    ax.set_title('Robust across cirrhosis etiology')
    fig.text(0.5, -0.13, 'Per-patient etiology; dual NASH+ASH removed; "Other" (1 HCC+) omitted as underpowered',
             ha='center', fontsize=9, color='#444')
    fig.savefig(os.path.join(OUT, 'etiology_bars.png'), bbox_inches='tight'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.3))
    leadtime_ax(axes[0], data, years); etiology_ax(axes[1], data)
    axes[0].set_title('A  |  HCCNet predicts HCC years in advance', loc='left')
    axes[1].set_title('B  |  Robust across cirrhosis etiology', loc='left')
    fig.suptitle('HCCNet on longitudinal MRI - prediction lead time and etiology (pico, per-fold ensemble)', y=1.02)
    fig.savefig(os.path.join(OUT, 'overview_panel.png'), bbox_inches='tight'); plt.close(fig)
    print('wrote leadtime_bands.png, etiology_bars.png, overview_panel.png ->', OUT)


if __name__ == '__main__':
    main()
