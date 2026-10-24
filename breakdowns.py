'''Offline stratified breakdowns of HCCNet finetune predictions (no retrain, no GPU).

Reads the per-model predictions saved by test.py (probs + per-timepoint prefix_probs/prefix_deltas)
and the etiology label CSV, and reports per-fold ensemble metrics (50 models = 5 CV folds x 10
seeds; fold(row k) = k % 5; ensemble the 10 seeds within a fold, metric per fold, mean +/- std
across the 5 folds) for:
  1. GLOBAL lead-time (years) overview   -> AUROC vs "using only scans >= Y years before dx"
  2. GLOBAL etiology overview            -> AUROC/AUPRC per NASH / ASH / Other stratum
  3. lead-time x etiology                -> the year curve within each etiology stratum
Per modality, plus a naive cross-modality mean-ensemble ("Combined") on the common patients.
Delta is in YEARS (labels.csv); the manuscript's month columns need finer labels, so the axis
here is years. Sex/age are NOT in the label CSV, so male/female splits are not available.
'''
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

K = 5
_HERE = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.environ.get('HCC_PRED_DIR', '/private/tmp/claude-502/scratch_pred')
LABELS_CSV = os.environ.get('HCC_LABELS_CSV', os.path.normpath(os.path.join(
    _HERE, os.pardir, 'notes', 'data', 'The moset recent labels-June 2024.csv')))
OUT = os.environ.get('HCC_BREAKDOWN_OUT', '/private/tmp/claude-502/breakdowns')
os.makedirs(OUT, exist_ok=True)
MODS = [('DW-MRI', 'dwi_pico_paper_ft'),
        ('T1 DCE-MRI', 't1dce_pico_paper_ft'),
        ('T1 IOP alone', 't1iop_only_pico_paper_ft'),
        ('T2 alone', 't2_only_pico_paper_ft')]


def etiology_map():
    '''patient id -> "NASH" / "ASH" / "Other" / "DROP" (dual NASH+ASH). Etiology is per-patient.'''
    m = {}
    with open(LABELS_CSV, newline='') as f:
        for r in csv.DictReader(f):
            pid = r['id']
            nash = int(float(r.get('nash', 0) or 0))
            ash = int(float(r.get('alcoholic_cirrosis', 0) or 0))
            grp = 'DROP' if (nash and ash) else 'NASH' if nash else 'ASH' if ash else 'Other'
            m[pid] = grp
    return m


def load(stem):
    d = np.load(os.path.join(PRED_DIR, f'{stem}.npz'), allow_pickle=True)
    pid = np.array([str(u).rsplit('_', 1)[0] for u in d['uids']])   # PT_004_001 -> PT_004
    return dict(probs=np.asarray(d['probs'], float), labels=np.asarray(d['labels']).ravel(),
                pid=pid, pfx=np.asarray(d['prefix_probs'], float), pdl=np.asarray(d['prefix_deltas'], float))


def perfold(prob_cube, labels, idx, fn):
    '''per-fold ensemble metric over patients `idx`: fold=k%5, ensemble 10 seeds, metric per fold.'''
    y = labels[idx]
    if len(idx) < 3 or len(set(y.tolist())) < 2:
        return None
    vals = []
    for c in range(K):
        rows = [k for k in range(prob_cube.shape[0]) if k % K == c]
        ens = prob_cube[rows][:, idx].mean(0)
        vals.append(fn(y, ens))
    return float(np.mean(vals)), float(np.std(vals))


def horizon_cube(m, Y):
    '''[n_models, n_pat] prediction using only scans >= Y years before dx (as-of Y), + kept idx.'''
    pfx, pdl = m['pfx'], m['pdl']
    out = np.full((pfx.shape[0], pfx.shape[1]), np.nan)
    keep = []
    for p in range(pfx.shape[1]):
        dd = pdl[p]
        ok = np.where((~np.isnan(dd)) & (dd >= Y))[0]
        if len(ok):
            out[:, p] = pfx[:, p, ok[dd[ok].argmin()]]   # smallest delta >= Y
            keep.append(p)
    return out, np.array(keep, int)


def fmt(v):
    return f'{v[0]:.3f}+/-{v[1]:.3f}' if v else '   -   (n/a)'


def main():
    et = etiology_map()
    data = {name: load(stem) for name, stem in MODS}
    years = list(range(1, 11))

    # ---- 1. GLOBAL lead-time (years) ----
    print('\n===== 1. GLOBAL lead-time overview (per-fold ensemble AUROC; "scans >= Y yr before dx") =====')
    hdr = 'Y(yr) | ' + ' | '.join(f'{n:14s}' for n, _ in MODS) + ' |   n(kept)'
    print(hdr)
    curves = {n: [] for n, _ in MODS}
    for Y in years:
        row = f'{Y:5d} | '
        ns = []
        for n, _ in MODS:
            hc, keep = horizon_cube(data[n], Y)
            v = perfold(hc, data[n]['labels'], keep, roc_auc_score)
            curves[n].append(v[0] if v else np.nan)
            row += f'{fmt(v):14s} | '
            ns.append(len(keep))
        print(row + f'  {ns}')
    plt.figure(figsize=(7, 5))
    for n, _ in MODS:
        plt.plot(years, curves[n], marker='o', label=n)
    plt.xlabel('lead time: scans >= Y years before diagnosis'); plt.ylabel('per-fold ensemble AUROC')
    plt.title('HCCNet AUROC vs prediction lead time'); plt.legend(); plt.grid(alpha=.3); plt.ylim(0.4, 1.0)
    plt.savefig(os.path.join(OUT, 'leadtime_global.png'), dpi=150, bbox_inches='tight'); plt.close()

    # ---- 2. GLOBAL etiology ----
    print('\n===== 2. GLOBAL etiology overview (per-fold ensemble, full sequence) =====')
    strata = ['NASH', 'ASH', 'Other']
    for metric_name, fn in [('AUROC', roc_auc_score), ('AUPRC', average_precision_score)]:
        print(f'-- {metric_name} --')
        print('modality        | ' + ' | '.join(f'{s:16s}' for s in strata))
        for n, _ in MODS:
            m = data[n]
            grp = np.array([et.get(p, 'Other') for p in m['pid']])
            cells = []
            for s in strata:
                idx = np.where(grp == s)[0]
                v = perfold(m['probs'], m['labels'], idx, fn)
                npos = int(m['labels'][idx].sum())
                cells.append(f'{fmt(v)} (n{len(idx)}/p{npos})')
            print(f'{n:15s} | ' + ' | '.join(f'{c:16s}' for c in cells))

    # etiology AUROC bar (per modality x stratum)
    plt.figure(figsize=(8, 5))
    x = np.arange(len(strata)); w = 0.2
    for i, (n, _) in enumerate(MODS):
        m = data[n]; grp = np.array([et.get(p, 'Other') for p in m['pid']])
        vals = []
        for s in strata:
            idx = np.where(grp == s)[0]
            v = perfold(m['probs'], m['labels'], idx, roc_auc_score)
            vals.append(v[0] if v else np.nan)
        plt.bar(x + i * w, vals, w, label=n)
    plt.xticks(x + 1.5 * w, strata); plt.ylabel('per-fold ensemble AUROC'); plt.ylim(0, 1)
    plt.title('HCCNet AUROC by etiology'); plt.legend(); plt.grid(alpha=.3, axis='y')
    plt.savefig(os.path.join(OUT, 'etiology_global.png'), dpi=150, bbox_inches='tight'); plt.close()

    # ---- 3. lead-time x etiology ----
    print('\n===== 3. lead-time x etiology (per-fold ensemble AUROC) =====')
    fig, axes = plt.subplots(1, len(strata), figsize=(15, 4.5), sharey=True)
    for si, s in enumerate(strata):
        ax = axes[si]
        print(f'-- {s} --')
        for n, _ in MODS:
            m = data[n]; grp = np.array([et.get(p, 'Other') for p in m['pid']])
            yv = []
            for Y in years:
                hc, keep = horizon_cube(m, Y)
                keep = keep[np.isin(keep, np.where(grp == s)[0])]
                v = perfold(hc, m['labels'], keep, roc_auc_score)
                yv.append(v[0] if v else np.nan)
            ax.plot(years, yv, marker='o', label=n)
        ax.set_title(f'{s}'); ax.set_xlabel('>= Y yr before dx'); ax.grid(alpha=.3); ax.set_ylim(0.3, 1.02)
    axes[0].set_ylabel('per-fold ensemble AUROC'); axes[-1].legend(fontsize=8)
    plt.suptitle('AUROC vs lead time, per etiology'); plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'leadtime_by_etiology.png'), dpi=150, bbox_inches='tight'); plt.close()

    print(f'\nFigures -> {OUT}')
    print('NOTE: sex/age not in the label CSV -> male/female split unavailable (needs the clinical-DB mapping).')
    print('NOTE: combined t1iop (T1 IOP + T2) still training; add its row when the npz lands.')


if __name__ == '__main__':
    main()
