'''Offline statistics from the saved per-model predictions (results/model_diagnostics/predictions/*.npz).

The 50 models are 5 CV folds x 10 seeds, flat-indexed as fold = seed*5 + cv (train.py:466), and
test.py stores probs rows in that flat order, so CV-fold(row k) = k % 5. Reports, per config:
  * per-model  mean +/- std   (cross-check vs the printed AUROC +/- line)
  * all-50 ensemble           (cross-check vs the printed ensemble line)
  * PER-FOLD ensemble mean +/- std  <- the manuscript statistic (ensemble 10 seeds within a fold,
                                       score each of the 5 folds, mean +/- std across folds)
No model, no GPU - pure numpy + sklearn on tiny arrays.
'''
import os
import glob
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

K_FOLDS = 5
PRED_DIR = os.environ.get('HCC_PRED_DIR',
                          os.path.join(os.environ['HOME'], 'projects/hccnet/results/model_diagnostics/predictions'))


def auc_pair(labels, p):
    return roc_auc_score(labels, p), average_precision_score(labels, p)


def main():
    files = sorted(glob.glob(os.path.join(PRED_DIR, '*.npz')))
    if not files:
        print(f'no npz under {PRED_DIR}'); return
    for f in files:
        d = np.load(f, allow_pickle=True)
        probs, labels = np.asarray(d['probs'], float), np.asarray(d['labels']).ravel()
        n_models = probs.shape[0]
        stem = os.path.basename(f)[:-4]

        per = np.array([auc_pair(labels, probs[k]) for k in range(n_models)])
        pm_au, pm_ap = per[:, 0], per[:, 1]

        ens = probs.mean(axis=0)
        e_au, e_ap = auc_pair(labels, ens)

        fold_au, fold_ap = [], []
        for c in range(K_FOLDS):
            rows = [k for k in range(n_models) if k % K_FOLDS == c]
            fe = probs[rows].mean(axis=0)
            a, p = auc_pair(labels, fe)
            fold_au.append(a); fold_ap.append(p)
        fold_au, fold_ap = np.array(fold_au), np.array(fold_ap)

        print(f'=== {stem}  ({n_models} models, {len(labels)} patients, {int(labels.sum())} pos) ===')
        print(f'  per-model     AUROC {pm_au.mean():.3f} +/- {pm_au.std():.3f}   '
              f'AUPRC {pm_ap.mean():.3f} +/- {pm_ap.std():.3f}')
        print(f'  all-50 ens    AUROC {e_au:.3f}          AUPRC {e_ap:.3f}   (single number, no +/-)')
        print(f'  PER-FOLD ens  AUROC {fold_au.mean():.3f} +/- {fold_au.std():.3f}   '
              f'AUPRC {fold_ap.mean():.3f} +/- {fold_ap.std():.3f}   [n=5 folds]')
        print(f'    fold AUROCs {np.round(fold_au, 3).tolist()}')
        print(f'    fold AUPRCs {np.round(fold_ap, 3).tolist()}')


if __name__ == '__main__':
    main()
