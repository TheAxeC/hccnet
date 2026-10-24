'''Regenerated finetune + baseline table values (per-fold ensemble mean +/- std, all six reported
metrics) for every config, formatted so the academic-paper revision can transcribe them into the
.tex tables. Offline, from the saved npz. Statistic matches the manuscript ("mean and std across
the five cross-validation folds"): fold = row k % 5; ensemble the 10 seeds within a fold; metric on
that ensemble prediction; mean +/- std across the 5 folds.
'''
import os
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score)

K = 5
PRED = os.environ.get('HCC_PRED_DIR', '/private/tmp/claude-502/scratch_pred')
ARCHS = [('femto', 'HCCNet-F'), ('compact', 'HCCNet-C'), ('pico', 'HCCNet-P'),
         ('nano', 'HCCNet-N'), ('tiny', 'HCCNet-T')]
MODS = [('dwi', 'DW-MRI'), ('t1dce', 'T1 DCE-MRI'), ('t1iop', 'T1 IOP & T2-MRI'),
        ('t1iop_only', 'T1 IOP-MRI'), ('t2_only', 'T2-MRI')]
METRICS = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUPRC', 'AUROC']


def perfold_all(cube, labels):
    rows = {m: [] for m in METRICS}
    for c in range(K):
        ens = cube[[k for k in range(cube.shape[0]) if k % K == c]].mean(0)
        pred = (ens >= 0.5).astype(int)
        rows['Accuracy'].append(accuracy_score(labels, pred))
        rows['Precision'].append(precision_score(labels, pred, zero_division=0))
        rows['Recall'].append(recall_score(labels, pred, zero_division=0))
        rows['F1-Score'].append(f1_score(labels, pred, zero_division=0))
        rows['AUPRC'].append(average_precision_score(labels, ens))
        rows['AUROC'].append(roc_auc_score(labels, ens))
    return {m: (float(np.mean(v)), float(np.std(v))) for m, v in rows.items()}


def main():
    for suffix, tag in [('paper_ft', 'FINETUNE  (tab:finetuned_results)'),
                        ('paper_base', 'BASELINE  (tab:baseline_results)')]:
        print(f'\n########## {tag} - regenerated per-fold ensemble ##########')
        print('modality        arch    ' + '  '.join(f'{m:16s}' for m in METRICS))
        for mst, mlab in MODS:
            for ast, alab in ARCHS:
                stem = f'{mst}_{ast}_{suffix}'
                p = os.path.join(PRED, f'{stem}.npz')
                if not os.path.isfile(p):
                    print(f'{mlab:15s} {alab:9s} (npz MISSING)'); continue
                d = np.load(p, allow_pickle=True)
                r = perfold_all(np.asarray(d['probs'], float), np.asarray(d['labels']).ravel())
                cells = '  '.join(f'{r[m][0]:.3f}+/-{r[m][1]:.3f}' for m in METRICS)
                print(f'{mlab:15s} {alab:9s} {cells}')


if __name__ == '__main__':
    main()
