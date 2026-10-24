'''Overall-performance + calibration figures for the meeting (offline, from finetune npz).

  1. pr_forest.png    - PR curves (ensemble, prevalence-aware) + AUROC/AUPRC forest (per-fold ens +/- std)
  2. calibration.png  - reliability diagram (ensemble) per modality, with ECE
Per-fold ensemble = 50 models as 5 folds x 10 seeds; fold(row k)=k%5; ensemble 10 seeds within a
fold; mean +/- std across the 5 folds. Small test set (58 patients, 8 HCC+) -> calibration is coarse.
'''
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

K = 5
PRED_DIR = os.environ.get('HCC_PRED_DIR', '/private/tmp/claude-502/scratch_pred')
OUT = os.environ.get('HCC_BREAKDOWN_OUT', '/private/tmp/claude-502/breakdowns')
os.makedirs(OUT, exist_ok=True)
MODS = [('DW-MRI', 'dwi_pico_paper_ft', '#0072B2'),
        ('T1 DCE-MRI', 't1dce_pico_paper_ft', '#E69F00'),
        ('T1 IOP alone', 't1iop_only_pico_paper_ft', '#009E73'),
        ('T2 alone', 't2_only_pico_paper_ft', '#D55E00')]
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.25, 'figure.dpi': 150})


def load(stem):
    d = np.load(os.path.join(PRED_DIR, f'{stem}.npz'), allow_pickle=True)
    return np.asarray(d['probs'], float), np.asarray(d['labels']).ravel()


def perfold(cube, labels, fn):
    return np.array([fn(labels, cube[[k for k in range(cube.shape[0]) if k % K == c]].mean(0)) for c in range(K)])


def main():
    data = {n: load(s) for n, s, _ in MODS}
    prev = np.mean(data['DW-MRI'][1])

    fig, (axp, axf) = plt.subplots(1, 2, figsize=(14, 5.4))

    # --- PR curves (all-50 ensemble) ---
    for n, _, col in MODS:
        probs, labels = data[n]
        ens = probs.mean(0)
        pr, rc, _ = precision_recall_curve(labels, ens)
        ap = average_precision_score(labels, ens)
        axp.plot(rc, pr, color=col, lw=2, label=f'{n}  (AP={ap:.2f})')
    axp.axhline(prev, ls='--', color='grey', lw=1, label=f'prevalence ({prev:.2f})')
    axp.set_xlabel('Recall'); axp.set_ylabel('Precision'); axp.set_xlim(0, 1); axp.set_ylim(0, 1.02)
    axp.set_title('A  |  Precision-Recall (ensemble)', loc='left'); axp.legend(fontsize=9, loc='upper right')

    # --- AUROC / AUPRC forest (per-fold ensemble mean +/- std) ---
    y = np.arange(len(MODS))[::-1]
    for i, (n, _, col) in enumerate(MODS):
        probs, labels = data[n]
        au = perfold(probs, labels, roc_auc_score); ap = perfold(probs, labels, average_precision_score)
        axf.errorbar(au.mean(), y[i] + 0.12, xerr=au.std(), fmt='o', color=col, capsize=4, ms=9)
        axf.errorbar(ap.mean(), y[i] - 0.12, xerr=ap.std(), fmt='s', color=col, capsize=4, ms=8, mfc='white')
    axf.axvline(0.5, ls='--', color='grey', lw=1)
    axf.set_yticks(y); axf.set_yticklabels([n for n, _, _ in MODS])
    axf.set_xlim(0, 1.02); axf.set_xlabel('score  (per-fold ensemble $\\pm$ std)')
    axf.set_title('B  |  AUROC (circle) & AUPRC (square)', loc='left')
    axf.plot([], [], 'ko', label='AUROC'); axf.plot([], [], 'ks', mfc='white', label='AUPRC')
    axf.legend(fontsize=9, loc='lower right')
    fig.suptitle('HCCNet overall performance (pico) - 58 patients, 8 HCC+', y=1.01)
    fig.savefig(os.path.join(OUT, 'pr_forest.png'), bbox_inches='tight'); plt.close(fig)

    # --- calibration / reliability (all-50 ensemble) ---
    nb = 5
    edges = np.linspace(0, 1, nb + 1); centers = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    for n, _, col in MODS:
        probs, labels = data[n]
        ens = probs.mean(0)
        freq = np.full(nb, np.nan); ece = 0.0
        for b in range(nb):
            m = (ens > edges[b]) & (ens <= edges[b + 1])
            if b == 0:
                m |= (ens == 0)
            if m.any():
                freq[b] = labels[m].mean()
                ece += m.sum() / len(ens) * abs(labels[m].mean() - ens[m].mean())
        ok = ~np.isnan(freq)
        ax.plot(centers[ok], freq[ok], marker='o', color=col, lw=2, label=f'{n}  (ECE={ece:.2f})')
    ax.plot([0, 1], [0, 1], '--', color='grey', lw=1, label='perfect')
    ax.set_xlabel('Mean predicted risk'); ax.set_ylabel('Observed HCC frequency')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_title('Calibration (ensemble, 5 bins)')
    ax.legend(fontsize=9, loc='upper left')
    fig.text(0.5, -0.02, 'Coarse at 8 HCC+ / 58 patients; illustrative of calibration trend', ha='center', fontsize=9, color='#444')
    fig.savefig(os.path.join(OUT, 'calibration.png'), bbox_inches='tight'); plt.close(fig)
    print('wrote pr_forest.png, calibration.png ->', OUT)


if __name__ == '__main__':
    main()
