'''Regenerated values for the remaining result tables (offline, from the saved npz + etiology CSV).
Per-fold ensemble (fold=row k%5; ensemble the 10 seeds; metric on that ensemble; mean+/-std across
the 5 folds) = the manuscript statistic. Covers: reliability (ECE/MCE/Brier, baseline+finetune),
pretrain_ablation (None/CNN-only/Transformer-only/Both at Pico), base_vs_pre (derived), etiology
(NASH/ASH; Other dropped per Axel), months recast to YEAR horizons. Fusion is NOT here (HPC re-run).
'''
import os
import csv
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score)

K = 5
_HERE = os.path.dirname(os.path.abspath(__file__))
PRED = os.environ.get('HCC_PRED_DIR', '/private/tmp/claude-502/scratch_pred')
LABELS_CSV = os.environ.get('HCC_LABELS_CSV', os.path.normpath(os.path.join(
    _HERE, os.pardir, 'notes', 'data', 'The moset recent labels-June 2024.csv')))
ARCHS = [('femto', 'HCCNet-F'), ('compact', 'HCCNet-C'), ('pico', 'HCCNet-P'),
         ('nano', 'HCCNet-N'), ('tiny', 'HCCNet-T')]
MAIN = [('dwi', 'DW-MRI'), ('t1dce', 'T1 DCE-MRI'), ('t1iop', 'T1 IOP & T2-MRI')]
ALLM = MAIN + [('t1iop_only', 'T1 IOP-MRI'), ('t2_only', 'T2-MRI')]
YEARS = [1, 2, 3, 4, 5]


def foldsets(n):
    return [[k for k in range(n) if k % K == c] for c in range(K)]


def ms(x):
    return (float(np.mean(x)), float(np.std(x))) if len(x) else (float('nan'), float('nan'))


def calib(p, y, nb=10):
    p = np.asarray(p, float); y = np.asarray(y, float)
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0, 1, nb + 1); ece = mce = 0.0
    for b in range(nb):
        m = (p > edges[b]) & (p <= edges[b + 1])
        if b == 0:
            m |= (p == 0)
        if m.any():
            gap = abs(y[m].mean() - p[m].mean()); ece += m.sum() / len(p) * gap; mce = max(mce, gap)
    return ece, mce, brier


def load(stem):
    f = os.path.join(PRED, f'{stem}.npz')
    if not os.path.isfile(f):
        return None
    d = np.load(f, allow_pickle=True)
    out = dict(probs=np.asarray(d['probs'], float), labels=np.asarray(d['labels']).ravel(),
               uids=np.array([str(u) for u in d['uids']]))
    if 'prefix_probs' in d:
        out['pfx'] = np.asarray(d['prefix_probs'], float); out['pdl'] = np.asarray(d['prefix_deltas'], float)
    return out


def etiomap():
    m = {}
    with open(LABELS_CSV, newline='') as fh:
        for r in csv.DictReader(fh):
            nash = int(float(r.get('nash', 0) or 0)); ash = int(float(r.get('alcoholic_cirrosis', 0) or 0))
            m[r['id']] = 'DROP' if (nash and ash) else 'NASH' if nash else 'ASH' if ash else 'Other'
    return m


def perfold_metric(cube, labels, idx, fn):
    vals = []
    for fold in foldsets(cube.shape[0]):
        yy = labels[idx]
        if len(idx) < 2 or len(set(yy.tolist())) < 2:
            continue
        vals.append(fn(yy, cube[fold][:, idx].mean(0)))
    return ms(vals)


def main():
    print('##### RELIABILITY (per-fold ensemble ECE/MCE/Brier) - baseline | finetune #####')
    for mst, ml in ALLM:
        for ast, al in ARCHS:
            row = {}
            for suf, key in [('paper_base', 'base'), ('paper_ft', 'ft')]:
                d = load(f'{mst}_{ast}_{suf}')
                if d is None:
                    row[key] = None; continue
                e, mc, br = [], [], []
                for fold in foldsets(d['probs'].shape[0]):
                    ens = d['probs'][fold].mean(0)
                    a, b, c = calib(ens, d['labels']); e.append(a); mc.append(b); br.append(c)
                row[key] = (ms(e), ms(mc), ms(br))
            def fmt(t): return ' '.join(f'{v[0]:.3f}+/-{v[1]:.3f}' for v in t) if t else 'MISSING'
            print(f'  {ml:15s} {al:9s} | base {fmt(row["base"])} | ft {fmt(row["ft"])}')

    print('\n##### PRETRAIN ABLATION (Pico) - AUPRC / AUROC per-fold ensemble #####')
    for mst, ml in MAIN:
        arms = [('None (baseline)', f'{mst}_pico_paper_base'),
                ('CNN backbone only', f'{mst}_pico_paper_cnnonly'),
                ('Transformer encoder only', f'{mst}_pico_paper_enconly'),
                ('Both (full pre-training)', f'{mst}_pico_paper_ft')]
        for name, stem in arms:
            d = load(stem)
            if d is None:
                print(f'  {ml:15s} {name:26s} MISSING'); continue
            idx = np.arange(len(d['labels']))
            ap = perfold_metric(d['probs'], d['labels'], idx, average_precision_score)
            au = perfold_metric(d['probs'], d['labels'], idx, roc_auc_score)
            print(f'  {ml:15s} {name:26s} AUPRC {ap[0]:.3f}+/-{ap[1]:.3f}  AUROC {au[0]:.3f}+/-{au[1]:.3f}')

    print('\n##### BASE vs PRE (AUPRC/AUROC baseline | finetune | rel.change%) #####')
    for mst, ml in ALLM:
        for ast, al in ARCHS:
            b = load(f'{mst}_{ast}_paper_base'); f = load(f'{mst}_{ast}_paper_ft')
            if b is None or f is None:
                print(f'  {ml:15s} {al:9s} MISSING'); continue
            idx = np.arange(len(b['labels']))
            out = []
            for fn in (average_precision_score, roc_auc_score):
                bm = perfold_metric(b['probs'], b['labels'], idx, fn)
                fm = perfold_metric(f['probs'], f['labels'], idx, fn)
                rel = 100 * (fm[0] - bm[0]) / bm[0] if bm[0] else float('nan')
                out.append(f'{bm[0]:.3f}+/-{bm[1]:.3f} -> {fm[0]:.3f}+/-{fm[1]:.3f} ({rel:+.1f}%)')
            print(f'  {ml:15s} {al:9s} | AUPRC {out[0]} | AUROC {out[1]}')

    print('\n##### ETIOLOGY (finetune, NASH / ASH; Other DROPPED) - AUPRC / AUROC per-fold ensemble #####')
    et = etiomap()
    for mst, ml in MAIN:
        for ast, al in ARCHS:
            d = load(f'{mst}_{ast}_paper_ft')
            if d is None:
                print(f'  {ml:15s} {al:9s} MISSING'); continue
            grp = np.array([et.get(u.rsplit('_', 1)[0], 'Other') for u in d['uids']])
            cells = []
            for s in ['NASH', 'ASH']:
                idx = np.where(grp == s)[0]
                npos = int(d['labels'][idx].sum())
                ap = perfold_metric(d['probs'], d['labels'], idx, average_precision_score)
                au = perfold_metric(d['probs'], d['labels'], idx, roc_auc_score)
                cells.append(f'{s}(n{len(idx)}/p{npos}) AUPRC {ap[0]:.3f}+/-{ap[1]:.3f} AUROC {au[0]:.3f}+/-{au[1]:.3f}')
            print(f'  {ml:15s} {al:9s} | ' + ' | '.join(cells))

    print('\n##### MONTHS -> YEAR horizons (finetune, AUROC per-fold ensemble; scans >= Y yr before dx) #####')
    print('  header: ' + '  '.join(f'{y}yr' for y in YEARS))
    for mst, ml in MAIN:
        for ast, al in ARCHS:
            d = load(f'{mst}_{ast}_paper_ft')
            if d is None or 'pfx' not in d:
                print(f'  {ml:15s} {al:9s} (no prefix data)'); continue
            pfx, pdl = d['pfx'], d['pdl']
            cells = []
            for Y in YEARS:
                out = np.full(pfx.shape[:2], np.nan); keep = []
                for pi in range(pfx.shape[1]):
                    ok = np.where((~np.isnan(pdl[pi])) & (pdl[pi] >= Y))[0]
                    if len(ok):
                        out[:, pi] = pfx[:, pi, ok[pdl[pi][ok].argmin()]]; keep.append(pi)
                keep = np.array(keep, int)
                au = perfold_metric(out, d['labels'], keep, roc_auc_score)
                cells.append(f'{au[0]:.3f}+/-{au[1]:.3f}(n{len(keep)})')
            print(f'  {ml:15s} {al:9s} | ' + '  '.join(cells))


if __name__ == '__main__':
    main()
