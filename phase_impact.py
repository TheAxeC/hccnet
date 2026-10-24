"""Paired held-out phase comparison; requires all five fully trained arms."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS = ['base', 'arterial', 'venous', 'late', 'all']
NAMES = ['Precontrast', 'Pre + arterial', 'Pre + venous', 'Pre + late', 'All DCE phases']
METRICS = {'AUROC': roc_auc_score, 'AUPRC': average_precision_score, 'Brier': brier_score_loss}


def analyse(root, n_boot=2000):
    sets = {}
    audits = {}
    for arm in ARMS:
        folder = root/arm
        audits[arm] = {stage: json.loads((folder/'cohort_audit'/f'{stage}.json').read_text()) for stage in ['pretrain','train','test']}
        with np.load(folder/'model_diagnostics/predictions'/f'phase_{arm}_pico_ft.npz', allow_pickle=True) as z:
            sets[arm] = {k:z[k].copy() for k in z.files}
        if sets[arm]['probs'].shape[0] != 50:
            raise ValueError(f'{arm}: expected 50 model predictions')
    base = sets['base']
    labels = base['labels'].ravel()
    for arm in ARMS:
        if audits[arm] != audits['base']:
            raise ValueError(f'{arm}: differing cohort or fold allocation')
        for key in ['uids','labels']:
            if not np.array_equal(sets[arm][key],base[key]):
                raise ValueError(f'{arm}: differing {key}')
        if not np.array_equal(sets[arm]['prefix_deltas'],base['prefix_deltas'],equal_nan=True):
            raise ValueError(f'{arm}: differing scan histories')
    risk = {a:sets[a]['probs'].mean(0) for a in ARMS}
    rng = np.random.default_rng(1234)
    pos, neg = np.flatnonzero(labels==1), np.flatnonzero(labels==0)
    samples = [np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)] for _ in range(n_boot)]
    summary = {'patients':len(labels),'hcc':len(pos),'models_per_arm':50,'bootstrap_samples':n_boot,'comparison':'paired stratified patient bootstrap; fixed Pico; matched acquired phases, visits, folds, seeds and training budgets','arms':{}}
    for arm in ARMS:
        summary['arms'][arm] = {}
        for name, fn in METRICS.items():
            bs = np.array([fn(labels[ix],risk[arm][ix]) for ix in samples])
            delta = np.array([fn(labels[ix],risk[arm][ix])-fn(labels[ix],risk['base'][ix]) for ix in samples])
            summary['arms'][arm][name] = {'estimate':float(fn(labels,risk[arm])),'ci95':np.quantile(bs,[.025,.975]).tolist(),'difference_from_base':float(fn(labels,risk[arm])-fn(labels,risk['base'])),'difference_ci95':np.quantile(delta,[.025,.975]).tolist()}
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig, axes = plt.subplots(1,3,figsize=(14,4))
    for ax,name in zip(axes,METRICS):
        vals = [summary['arms'][a][name] for a in ARMS]
        mid = np.array([v['estimate'] for v in vals]); ci=np.array([v['ci95'] for v in vals])
        for i,v in enumerate(vals):
            ax.plot([i,i],v['ci95'],color='#426a91'); ax.plot(i,v['estimate'],'o',color='#426a91')
        ax.set_xticks(range(5),NAMES,rotation=35,ha='right',fontsize=8); ax.set_title(name); ax.set_ylim(0,1); ax.grid(axis='y',alpha=.2)
    fig.suptitle(f'Matched DCE phase training | {len(pos)} HCC / {len(labels)} held-out patients | patient bootstrap 95% CI')
    fig.tight_layout(); fig.savefig(root/'phase_impact.png',dpi=160); plt.close(fig)
    for i in pos:
        pid=str(base['uids'][i]).rsplit('_',1)[0]
        fig,ax=plt.subplots(figsize=(8,4))
        for arm,label in zip(ARMS,NAMES):
            z=sets[arm]; d=z['prefix_deltas'][i]; valid=np.isfinite(d)
            p=z['prefix_probs'][:,i,valid].mean(0)
            ax.plot(-d[valid],p,'o-',label=label)
        ax.set(xlabel='Years relative to diagnosis (whole-year bins)',ylabel='Model probability',ylim=(-.03,1.03),title=pid+' | Matched phase-specific training')
        ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(root/f'{pid}_phase_trajectory.png',dpi=140);plt.close(fig)
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    args=parser.parse_args(); print(json.dumps(analyse(args.run),indent=2))
