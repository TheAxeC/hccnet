'''Full regenerated-vs-manuscript comparison for the finetune and baseline tables.

For every config (5 modalities x 5 archs), computes the per-fold ensemble AUROC/AUPRC (5 folds =
row k%5; ensemble the 10 seeds within a fold; mean +/- std across folds) - the same statistic the
manuscript reports - and diffs it against the value parsed from the manuscript .tex. Also reports
the arch-mean per modality (the headline). Offline, from the saved npz. No retrain.
'''
import os
import re
import glob
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

K = 5
PRED = os.environ.get('HCC_PRED_DIR', '/private/tmp/claude-502/scratch_pred')
TAB = os.path.join(os.path.dirname(__file__), '..', 'manuscript', 'tables')
ARCHS = [('femto', 'HCCNet-F'), ('compact', 'HCCNet-C'), ('pico', 'HCCNet-P'),
         ('nano', 'HCCNet-N'), ('tiny', 'HCCNet-T')]
MODS = [('dwi', 'DW-MRI'), ('t1dce', 'T1 DCE-MRI'), ('t1iop', 'T1 IOP \\& T2-MRI'),
        ('t1iop_only', 'T1 IOP-MRI'), ('t2_only', 'T2-MRI')]
MOD_LABEL = {'dwi': 'DW-MRI', 't1dce': 'T1 DCE-MRI', 't1iop': 'T1 IOP+T2',
             't1iop_only': 'T1 IOP alone', 't2_only': 'T2 alone'}


def perfold(cube, labels, fn):
    v = [fn(labels, cube[[k for k in range(cube.shape[0]) if k % K == c]].mean(0)) for c in range(K)]
    return float(np.mean(v)), float(np.std(v))


def regen(stem):
    d = np.load(os.path.join(PRED, f'{stem}.npz'), allow_pickle=True)
    probs, labels = np.asarray(d['probs'], float), np.asarray(d['labels']).ravel()
    return perfold(probs, labels, roc_auc_score), perfold(probs, labels, average_precision_score)


def parse_manuscript(texname):
    '''-> {(mod_stem, arch_stem): (auprc, auroc)} from the .tex (AUPRC=col6, AUROC=col7).'''
    txt = open(os.path.join(TAB, texname)).read()
    arch_map = {lab: st for st, lab in ARCHS}
    out, cur_mod = {}, None
    for line in txt.splitlines():
        if '$\\pm$' not in line or 'multicolumn' in line:
            continue
        cells = [c.strip() for c in re.split(r'(?<!\\)&', line)]
        # identify modality (col0, exact) + arch (col1); continuation rows have empty col0
        m0 = cells[0].strip()
        for st, lab in MODS:
            if m0 == lab:
                cur_mod = st
        arch = next((arch_map[a] for a in arch_map if a in cells[1]), None)
        if cur_mod is None or arch is None:
            continue
        nums = [float(re.search(r'-?\d+\.\d+', c).group()) for c in cells[-2:]]   # AUPRC, AUROC
        out[(cur_mod, arch)] = (nums[0], nums[1])
    return out


def main():
    for tag, texname, suffix in [('FINETUNE', 'results_finetune.tex', 'paper_ft'),
                                 ('BASELINE', 'results_baseline.tex', 'paper_base')]:
        man = parse_manuscript(texname)
        print(f'\n########## {tag}: regenerated (per-fold ensemble) vs manuscript ##########')
        print(f'{"modality":13s} {"arch":6s} | {"AUROC regen":16s} {"man":6s} {"d":7s} | {"AUPRC regen":16s} {"man":6s} {"d":7s}')
        arch_means = {}
        for mst, _ in MODS:
            regen_au, regen_ap = [], []
            for ast, _ in ARCHS:
                stem = f'{mst}_{ast}_{suffix}'
                if not os.path.isfile(os.path.join(PRED, f'{stem}.npz')):
                    print(f'{MOD_LABEL[mst]:13s} {ast:6s} | (npz missing)'); continue
                (au_m, au_s), (ap_m, ap_s) = regen(stem)
                mv = man.get((mst, ast), (np.nan, np.nan))
                dau, dap = au_m - mv[1], ap_m - mv[0]
                print(f'{MOD_LABEL[mst]:13s} {ast:6s} | {au_m:.3f}+/-{au_s:.3f}   {mv[1]:.3f}  {dau:+.3f} | '
                      f'{ap_m:.3f}+/-{ap_s:.3f}   {mv[0]:.3f}  {dap:+.3f}')
                regen_au.append(au_m); regen_ap.append(ap_m)
            if regen_au:
                man_au = np.mean([man.get((mst, a), (np.nan, np.nan))[1] for a, _ in ARCHS])
                man_ap = np.mean([man.get((mst, a), (np.nan, np.nan))[0] for a, _ in ARCHS])
                arch_means[mst] = (np.mean(regen_au), man_au, np.mean(regen_ap), man_ap)
        print(f'  --- arch-mean per modality (headline) ---')
        for mst, (rau, mau, rap, map_) in arch_means.items():
            print(f'  {MOD_LABEL[mst]:13s} AUROC {rau:.3f} vs {mau:.3f} ({rau-mau:+.3f})   '
                  f'AUPRC {rap:.3f} vs {map_:.3f} ({rap-map_:+.3f})')


if __name__ == '__main__':
    main()
