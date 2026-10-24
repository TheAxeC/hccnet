'''Constituent comparison for the meeting: does combining T1 IOP + T2 beat either contrast alone?

Head-to-head of the COMBINED t1iop (T1 IOP + T2) vs its constituents T1 IOP alone and T2 alone,
aligned on their common patients (apples-to-apples). Per-fold ensemble (fold(row k)=k%5). Produces
constituent_comparison.png: (A) AUROC vs prediction lead time, (B) AUROC/AUPRC forest.
'''
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

K = 5
PRED_DIR = os.environ.get('HCC_PRED_DIR', '/private/tmp/claude-502/scratch_pred')
OUT = os.environ.get('HCC_BREAKDOWN_OUT', '/private/tmp/claude-502/breakdowns')
os.makedirs(OUT, exist_ok=True)
MODS = [('DW-MRI', 'dwi_pico_paper_ft', '#0072B2'),
        ('T1 DCE-MRI', 't1dce_pico_paper_ft', '#E69F00'),
        ('T1 IOP + T2 (combined)', 't1iop_pico_paper_ft', '#CC79A7'),
        ('T1 IOP alone', 't1iop_only_pico_paper_ft', '#009E73'),
        ('T2 alone', 't2_only_pico_paper_ft', '#D55E00')]
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.25, 'figure.dpi': 150})


def load(stem):
    d = np.load(os.path.join(PRED_DIR, f'{stem}.npz'), allow_pickle=True)
    return dict(probs=np.asarray(d['probs'], float), labels=np.asarray(d['labels']).ravel(),
                uid=np.array([str(u) for u in d['uids']]),
                pfx=np.asarray(d['prefix_probs'], float), pdl=np.asarray(d['prefix_deltas'], float))


def align(datasets):
    '''Restrict every config to the common patient uids, in one shared order (fair comparison).'''
    common = set(datasets[0]['uid'])
    for d in datasets[1:]:
        common &= set(d['uid'])
    order = [u for u in datasets[0]['uid'] if u in common]
    for d in datasets:
        pos = {u: i for i, u in enumerate(d['uid'])}
        idx = [pos[u] for u in order]
        d['probs'] = d['probs'][:, idx]; d['labels'] = d['labels'][idx]
        d['pfx'] = d['pfx'][:, idx, :]; d['pdl'] = d['pdl'][idx, :]; d['uid'] = np.array(order)
    return len(order)


def perfold(cube, labels, idx, fn):
    y = labels[idx]
    if len(idx) < 3 or len(set(y.tolist())) < 2:
        return None
    v = [fn(y, cube[[k for k in range(cube.shape[0]) if k % K == c]][:, idx].mean(0)) for c in range(K)]
    return float(np.mean(v)), float(np.std(v))


def horizon(m, Y):
    pfx, pdl = m['pfx'], m['pdl']
    out = np.full(pfx.shape[:2], np.nan); keep = []
    for p in range(pfx.shape[1]):
        ok = np.where((~np.isnan(pdl[p])) & (pdl[p] >= Y))[0]
        if len(ok):
            out[:, p] = pfx[:, p, ok[pdl[p][ok].argmin()]]; keep.append(p)
    return out, np.array(keep, int)


def main():
    data = {n: load(s) for n, s, _ in MODS}
    ncommon = align([data[n] for n, _, _ in MODS])
    npos = int(data[MODS[0][0]]['labels'].sum())
    years = list(range(1, 8))

    fig, (axl, axf) = plt.subplots(1, 2, figsize=(14, 5.4))

    # A: lead-time
    for n, _, col in MODS:
        m, s, ns = [], [], []
        for Y in years:
            hc, keep = horizon(data[n], Y); v = perfold(hc, data[n]['labels'], keep, roc_auc_score)
            m.append(v[0] if v else np.nan); s.append(v[1] if v else np.nan); ns.append(len(keep))
        m, s = np.array(m), np.array(s)
        lw = 3.0 if n in ('DW-MRI', 'T2 alone') else 1.7
        axl.plot(years, m, marker='o', color=col, lw=lw, label=n, zorder=3)
        axl.fill_between(years, m - s, m + s, color=col, alpha=0.15)
    axl.axhline(0.5, ls='--', color='grey', lw=1)
    axl.set_xticks(years); axl.set_xticklabels([f'{Y}\nn={n}' for Y, n in zip(years, ns)])
    axl.set_xlabel('prediction lead time  (scans $\\geq$ Y years before diagnosis)')
    axl.set_ylabel('AUROC  (per-fold ensemble $\\pm$ std)'); axl.set_ylim(0.35, 1.0)
    axl.set_title('A  |  Lead time', loc='left'); axl.legend(loc='lower left', frameon=False, fontsize=10)

    # B: overall forest
    y = np.arange(len(MODS))[::-1]
    allidx = np.arange(data[MODS[0][0]]['labels'].shape[0])
    for i, (n, _, col) in enumerate(MODS):
        au = perfold(data[n]['probs'], data[n]['labels'], allidx, roc_auc_score)
        ap = perfold(data[n]['probs'], data[n]['labels'], allidx, average_precision_score)
        axf.errorbar(au[0], y[i] + 0.12, xerr=au[1], fmt='o', color=col, capsize=4, ms=10)
        axf.errorbar(ap[0], y[i] - 0.12, xerr=ap[1], fmt='s', color=col, capsize=4, ms=9, mfc='white')
    axf.axvline(0.5, ls='--', color='grey', lw=1)
    axf.set_yticks(y); axf.set_yticklabels([n for n, _, _ in MODS]); axf.set_xlim(0, 1.02)
    axf.set_xlabel('score  (per-fold ensemble $\\pm$ std)')
    axf.set_title('B  |  Overall (AUROC circle, AUPRC square)', loc='left')
    axf.plot([], [], 'ko', label='AUROC'); axf.plot([], [], 'ks', mfc='white', label='AUPRC')
    axf.legend(fontsize=9, loc='lower right')

    fig.suptitle(f'HCCNet modality comparison (pico) - T2 alone rivals DW-MRI; combining T1 IOP + T2 does not help'
                 f'\n({ncommon} common patients, {npos} HCC+)', y=1.04, fontsize=13)
    fig.savefig(os.path.join(OUT, 'modality_comparison.png'), bbox_inches='tight'); plt.close(fig)
    print(f'wrote modality_comparison.png ({ncommon} common patients, {npos} HCC+) ->', OUT)


if __name__ == '__main__':
    main()
