'''
Evaluation entry point for HCCNet.

By default this evaluates the run identified by --arch / --suffix: it finds every
cross-validation/seed checkpoint that train.py saved
(model_weights/weights_fold*_<suffix>.pth), runs each on the held-out test set,
reports per-model AUROC/AUPRC (mean +/- std) and the probability-ensemble of all
models, and writes ROC + PR curves.

Pass --paper-sweep to instead reproduce the paper's table across all
architectures and the pretrained/random-init suffixes (see utils/paper_config.py),
loading from --weights-dir.
'''

from typing import List
import os
import glob
import re
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from monai.data import set_track_meta
from monai.utils import set_determinism
from models.mednet import MedNet
from utils.preprocessing import load_backbone, load_data
from utils.config import parse_args
from utils.paper_config import (
    PAPER_TEST_ARCHS,
    PAPER_TEST_SUFFIXES,
    HCC_TEST_PREVALENCE,
)
from utils.utils import prep_batch


def setup() -> None:
    dist.init_process_group(backend='nccl')


def cleanup() -> None:
    dist.destroy_process_group()


def build_model(args: argparse.Namespace, arch: str, num_classes: int, device: int) -> nn.Module:
    backbone = load_backbone(args, arch)
    model = MedNet(
        backbone,
        num_classes=num_classes,
        pretrain=False,
        max_len=12,
        num_layers=4 if any(arch in x for x in ['femto', 'pico', 'compact']) else 6,
        nhead=4 if arch == 'compact' else None,
        dropout=args.dropout,
        eps=args.epsilon,
        pe_mode=getattr(args, 'pe_mode', 'sqrt'))
    return model.to(device)


@torch.no_grad()
def predict(model: nn.Module, loader, device: int, amp: bool, meta: bool = False):
    '''Run one model over the test loader, returning (probs, labels) as 1D arrays. With
    meta=True also return per-sample patient uid + delta sequence (for saving predictions,
    from which etiology strata and month horizons are computed offline).'''
    model.eval()
    probs, labels, uids, deltas = [], [], [], []
    for batch in loader:
        inputs, label, delta, padding_mask = prep_batch(batch, batch_size=1, device=device)
        with autocast(enabled=amp):
            logits = model(inputs, pad_mask=padding_mask, pos=delta)
        probs.append(torch.sigmoid(logits.squeeze(-1)).float().cpu())
        labels.append(label.cpu())
        if meta:
            u = batch.get('uid', ['?'])
            uids.append(u[0] if isinstance(u, (list, tuple)) else str(u))
            deltas.append(np.asarray(delta.detach().cpu()).ravel())
    p = torch.cat(probs).numpy().ravel()
    l = torch.cat(labels).numpy().ravel()
    return (p, l, uids, deltas) if meta else (p, l)


def _transformer_head(model, feats, pad_mask, pos):
    '''MedNet from pre-extracted CNN features (skips the CNN) -> logits. Mirrors the
    pretrain=False path of MedNet.forward, so a truncation only re-runs the cheap transformer.'''
    x, pm, ps = model.add_cls_token(feats, pad_mask, pos)
    x = model.positional_encoding(x, ps)
    x = x.permute(1, 0, 2)
    x = model.transformer_encoder(x, src_key_padding_mask=pm)
    x = x.permute(1, 0, 2)
    x = model.pooler(x, padding_mask=pm)
    return model.head(x)


@torch.no_grad()
def predict_prefixes(model, loader, device, amp):
    '''For each patient, the model's prediction at every "as-of" truncation: keep only the scans
    at least `thr` before diagnosis (mask scans with delta < thr; padding is delta == 0), for thr
    over the patient's distinct real deltas, ordered oldest-only -> full sequence. Returns
    per-patient (probs, deltas) arrays. The CNN runs once per patient; each truncation re-runs only
    the transformer. UNIT-AGNOSTIC: the actual delta values are returned alongside the predictions,
    so the OFFLINE month-horizon step owns the exact horizon definition (horizon N -> per patient,
    the as-of point with the smallest delta >= N). The final entry (thr = smallest delta, all scans)
    equals the standard full-sequence prediction.'''
    model.eval()
    pfx_probs, pfx_deltas = [], []
    for batch in loader:
        inputs, _, delta, pad = prep_batch(batch, batch_size=1, device=device)
        with autocast(enabled=amp):
            feats = model.extract_features(inputs)
        d = delta.detach().cpu().reshape(-1).numpy()
        real = (pad.detach().cpu().reshape(-1).numpy() == 0.0)
        thrs = np.unique(d[real])                           # distinct real deltas, including zero
        pp, dd = [], []
        for thr in thrs[::-1]:                              # oldest-only first ... all scans last
            pad_thr = torch.where(delta < float(thr), torch.ones_like(pad), pad)
            with autocast(enabled=amp):
                logits = _transformer_head(model, feats, pad_thr, delta)
            pp.append(float(torch.sigmoid(logits.squeeze(-1)).reshape(-1)[0].cpu()))
            dd.append(float(thr))
        pfx_probs.append(np.asarray(pp, dtype=float))
        pfx_deltas.append(np.asarray(dd, dtype=float))
    return pfx_probs, pfx_deltas


def evaluate_stem(args, loader, device, arch, stem, weights_root, num_classes):
    '''Evaluate every weights_fold*_<stem>.pth checkpoint found under weights_root.

    Returns (probs_matrix [n_models, n_samples], labels [n_samples]) or (None, None)
    if no checkpoints match.
    '''
    # The fold wildcard has to stop at the fold number. A bare weights_fold*_<stem>.pth also
    # matches any checkpoint whose name merely ENDS with the stem, so the t1dce_pico stem picks
    # up fuse_dwi_t1dce_pico as well and the 8-channel fusion weights fail to load into the
    # 4-channel model. Anchor the name so only weights_fold<digits>_<stem>.pth is accepted.
    pattern = os.path.join(weights_root, 'model_weights', f'weights_fold*_{stem}.pth')
    exact = re.compile(r'^weights_fold(\d+)_' + re.escape(stem) + r'\.pth$')
    matched = [(m.group(1), path) for path in glob.glob(pattern)
               for m in [exact.match(os.path.basename(path))] if m]
    paths = [path for _, path in sorted(matched, key=lambda t: int(t[0]))]
    if not paths:
        print(f'[skip] no checkpoints matching {pattern}')
        return None, None, None, None, None, None
    all_probs, ref_labels, uids, deltas = [], None, None, None
    all_pfx, pfx_deltas = [], None
    for k, path in enumerate(paths):
        model = build_model(args, arch, num_classes, device)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        if k == 0:
            probs, labels, uids, deltas = predict(model, loader, device, args.amp, meta=True)
        else:
            probs, labels = predict(model, loader, device, args.amp)
        all_probs.append(probs)
        ref_labels = labels if ref_labels is None else ref_labels
        if getattr(args, 'save_prefixes', False):
            pp, pdd = predict_prefixes(model, loader, device, args.amp)
            all_pfx.append(pp)
            if pfx_deltas is None:
                pfx_deltas = pdd
    prefix_cube, prefix_deltas_arr = None, None
    if all_pfx:
        n_pat = len(all_pfx[0])
        max_t = max((len(a) for per_pat in all_pfx for a in per_pat), default=0)
        prefix_cube = np.full((len(all_pfx), n_pat, max_t), np.nan)  # [n_models, n_patients, max_asof]
        for mi, per_pat in enumerate(all_pfx):
            for pi, arr in enumerate(per_pat):
                prefix_cube[mi, pi, :len(arr)] = arr
        prefix_deltas_arr = np.full((n_pat, max_t), np.nan)          # as-of delta per column (model-independent)
        for pi, arr in enumerate(pfx_deltas):
            prefix_deltas_arr[pi, :len(arr)] = arr
    print(f'[ok] {stem}: ensembled {len(paths)} checkpoints over {len(ref_labels)} test patients')
    return np.vstack(all_probs), ref_labels, uids, deltas, prefix_cube, prefix_deltas_arr


def _threshold_metrics(probs, labels, threshold: float = 0.5):
    '''Accuracy / precision / recall / F1 at a probability threshold (default 0.5).'''
    pred = (np.asarray(probs) >= threshold).astype(int)
    return (metrics.accuracy_score(labels, pred),
            metrics.precision_score(labels, pred, zero_division=0),
            metrics.recall_score(labels, pred, zero_division=0),
            metrics.f1_score(labels, pred, zero_division=0))


def _calibration_metrics(probs, labels, n_bins: int = 10):
    '''Expected (ECE) and maximum (MCE) calibration error, and the Brier score,
    using equal-width confidence bins.'''
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    brier = float(np.mean((probs - labels) ** 2))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce, n = 0.0, 0.0, len(probs)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (probs > lo) & (probs <= hi)
        if b == 0:
            mask |= (probs == lo)
        if not mask.any():
            continue
        gap = abs(labels[mask].mean() - probs[mask].mean())
        ece += mask.sum() / n * gap
        mce = max(mce, gap)
    return float(ece), float(mce), brier


def _cumulative_gain_mae(per_fold):
    '''Fold-robustness diagnostic (Sec. 4 MAE). Folds are ordered by ascending
    metric; the normalized cumulative gain cumsum(sorted)/sum is compared against
    the i/n identity line and the MAE is the mean absolute deviation. ~0 means the
    metric is spread evenly across folds (robust); larger means a few folds carry
    the result. NOTE: this matches the "ordered ascending, follow the diagonal"
    description; the exact normalization in the paper's figure code (not in repo)
    should be confirmed before treating the absolute MAE as canonical.'''
    m = np.sort(np.asarray(per_fold, dtype=float))
    n = len(m)
    exp = np.arange(1, n + 1) / n
    if m.sum() == 0:
        return np.zeros(n), exp, 0.0
    obs = np.cumsum(m) / m.sum()
    return obs, exp, float(np.mean(np.abs(obs - exp)))


def summarise(label, probs_matrix, labels):
    '''Per-model mean +/- std for all reported metrics, the probability-ensemble
    metrics, and the cumulative-gain fold-robustness MAE.'''
    per_auroc, per_auprc = [], []
    per_acc, per_prec, per_rec, per_f1 = [], [], [], []
    per_ece, per_mce, per_brier = [], [], []
    for p in probs_matrix:
        per_auroc.append(metrics.roc_auc_score(labels, p))
        per_auprc.append(metrics.average_precision_score(labels, p))
        a, pr, rc, f = _threshold_metrics(p, labels)
        per_acc.append(a); per_prec.append(pr); per_rec.append(rc); per_f1.append(f)
        e, mc, br = _calibration_metrics(p, labels)
        per_ece.append(e); per_mce.append(mc); per_brier.append(br)
    ens = probs_matrix.mean(axis=0)
    ens_auroc = metrics.roc_auc_score(labels, ens)
    ens_auprc = metrics.average_precision_score(labels, ens)
    # Ensemble value for EVERY reported metric (threshold / calibrate the ensemble prediction),
    # so the manuscript can report the deployed-ensemble number alongside the per-model spread
    # for all columns, not just AUROC/AUPRC.
    ens_acc, ens_prec, ens_rec, ens_f1 = _threshold_metrics(ens, labels)
    ens_ece, ens_mce, ens_brier = _calibration_metrics(ens, labels)
    cg_auprc = _cumulative_gain_mae(per_auprc)
    cg_auroc = _cumulative_gain_mae(per_auroc)

    def ms(x):
        return f'{np.mean(x):.3f} +/- {np.std(x):.3f}'
    text = (
        f'=== {label} ({probs_matrix.shape[0]} models, {probs_matrix.shape[1]} test patients) ===\n'
        f'  Accuracy : {ms(per_acc)}    Precision: {ms(per_prec)}\n'
        f'  Recall   : {ms(per_rec)}    F1-Score : {ms(per_f1)}\n'
        f'  AUPRC    : {ms(per_auprc)}    AUROC    : {ms(per_auroc)}\n'
        f'  ECE      : {ms(per_ece)}    MCE      : {ms(per_mce)}    Brier: {ms(per_brier)}\n'
        f'  ensemble AUROC / AUPRC: {ens_auroc:.3f} / {ens_auprc:.3f}\n'
        f'  ensemble Accuracy/Precision/Recall/F1: {ens_acc:.3f} / {ens_prec:.3f} / {ens_rec:.3f} / {ens_f1:.3f}\n'
        f'  ensemble ECE/MCE/Brier: {ens_ece:.3f} / {ens_mce:.3f} / {ens_brier:.3f}\n'
        f'  cumulative-gain MAE (fold robustness): AUPRC {cg_auprc[2]:.3f}, AUROC {cg_auroc[2]:.3f}\n'
    )
    return {'label': label, 'ens': ens, 'labels': labels,
            'ens_auroc': ens_auroc, 'ens_auprc': ens_auprc,
            'ens_acc': ens_acc, 'ens_prec': ens_prec, 'ens_rec': ens_rec, 'ens_f1': ens_f1,
            'ens_ece': ens_ece, 'ens_mce': ens_mce, 'ens_brier': ens_brier,
            'per_auroc': per_auroc, 'per_auprc': per_auprc,
            'per_acc': per_acc, 'per_prec': per_prec, 'per_rec': per_rec, 'per_f1': per_f1,
            'per_ece': per_ece, 'per_mce': per_mce, 'per_brier': per_brier,
            'cg_auprc': cg_auprc, 'cg_auroc': cg_auroc, 'text': text}


def plot_curves(results: List[dict], metric_type: str, out_path: str):
    plt.figure(figsize=(6, 6))
    for res in results:
        labels, probs = res['labels'], res['ens']
        if metric_type == 'AUROC':
            x, y, _ = metrics.roc_curve(labels, probs)
            score = res['ens_auroc']
        else:
            y, x, _ = metrics.precision_recall_curve(labels, probs)
            score = res['ens_auprc']
        plt.plot(x, y, linewidth=1.5, label=f"{res['label']} (AUC={score:.3f})")
    if metric_type == 'AUROC':
        plt.plot([0, 1], [0, 1], '--', color='black', linewidth=1)
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    else:
        plt.plot([0, 1], [HCC_TEST_PREVALENCE, HCC_TEST_PREVALENCE], '--', color='black', linewidth=1)
        plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title(metric_type)
    plt.legend(fontsize=8, loc='lower right' if metric_type == 'AUROC' else 'upper right')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_reliability(results: List[dict], out_path: str, n_bins: int = 10):
    '''Reliability diagram: binned mean predicted probability vs observed frequency.'''
    plt.figure(figsize=(6, 6))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    for res in results:
        probs, labels = res['ens'], res['labels']
        freq = np.full(n_bins, np.nan)
        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            mask = (probs > lo) & (probs <= hi)
            if b == 0:
                mask |= (probs == lo)
            if mask.any():
                freq[b] = labels[mask].mean()
        plt.plot(centers, freq, marker='o', linewidth=1.5,
                 label=f"{res['label']} (ECE={np.mean(res['per_ece']):.3f})")
    plt.plot([0, 1], [0, 1], '--', color='black', linewidth=1)
    plt.xlabel('Mean predicted probability'); plt.ylabel('Observed frequency')
    plt.title('Reliability'); plt.legend(fontsize=8, loc='upper left')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_cumulative_gain(results: List[dict], metric_type: str, out_path: str):
    '''Normalized cumulative-gain curve (folds ordered ascending) vs the diagonal.'''
    plt.figure(figsize=(6, 6))
    for res in results:
        obs, _, mae = res['cg_auprc'] if metric_type == 'AUPRC' else res['cg_auroc']
        x = np.arange(1, len(obs) + 1) / len(obs)
        plt.plot(x, obs, marker='o', linewidth=1.5, label=f"{res['label']} (MAE={mae:.3f})")
    plt.plot([0, 1], [0, 1], '--', color='black', linewidth=1)
    plt.xlabel('Fold fraction (ordered ascending)')
    plt.ylabel(f'Normalized cumulative {metric_type}')
    plt.title(f'Cumulative gain ({metric_type})'); plt.legend(fontsize=8, loc='upper left')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def main(args: argparse.Namespace) -> None:
    set_determinism(seed=args.seed)
    if args.distributed:
        setup()
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    num_classes = args.num_classes if args.num_classes > 2 else 1

    dataloader, _ = load_data(args, device, phase='test', max_delta=args.max_delta)
    loader = dataloader['test'][0]
    set_track_meta(False)
    # Modality/group = everything before the architecture token, so multi-token groups
    # (t1iop_only, t2_only, fuse_*) are labelled correctly instead of collapsing to their first
    # token. The COMBINED t1iop group (T1 IOP + T2) is still shown as t1iop_t2, matching the
    # manuscript's "T1 IOP & T2-MRI" row; t1iop_only (T1 IOP alone) keeps its own name.
    _ARCHS = ('femto', 'compact', 'pico', 'nano', 'tiny')
    _toks = args.suffix.split('_')
    _cut = next((i for i, t in enumerate(_toks) if t in _ARCHS), len(_toks))
    _group = '_'.join(_toks[:_cut]) or _toks[0]
    modality = 't1iop_t2' if _group == 't1iop' else _group

    # Build the list of (display label, arch, checkpoint-stem, weights-root) to evaluate.
    if args.paper_sweep:
        configs = [(f'{arch}-{suffix}', arch, f'{modality}_{arch}_{suffix}', args.weights_dir)
                   for arch in PAPER_TEST_ARCHS for suffix in PAPER_TEST_SUFFIXES]
    else:
        configs = [(f'{modality}_{args.arch}', args.arch, args.suffix, args.results_dir)]

    results = []
    to_save = []
    for label, arch, stem, root in configs:
        probs_matrix, labels, uids, deltas, pfx, pfx_deltas = evaluate_stem(args, loader, device, arch, stem, root, num_classes)
        if probs_matrix is not None:
            results.append(summarise(label, probs_matrix, labels))
            to_save.append((stem, probs_matrix, labels, uids, deltas, pfx, pfx_deltas))

    if rank == 0 and results:
        report = '\n'.join(r['text'] for r in results)
        print('\n' + report)
        diag_dir = os.path.join(args.results_dir, 'model_diagnostics')
        os.makedirs(diag_dir, exist_ok=True)
        tag = 'paper_sweep' if args.paper_sweep else f'{modality}_{args.arch}_{args.suffix}'
        with open(os.path.join(diag_dir, f'test_metrics_{tag}.txt'), 'w') as f:
            f.write(report)
        # Persist the per-model per-patient predictions so etiology strata, month horizons,
        # per-fold ensemble / bootstrap CIs are all computable OFFLINE, forever, no retrain.
        pred_dir = os.path.join(diag_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        for stem, pm, lb, uid, dl, pfx, pfxd in to_save:
            save_kw = dict(probs=pm, labels=lb,
                           uids=np.array(uid, dtype=object),
                           deltas=np.array(dl, dtype=object))
            if pfx is not None:                       # per-timepoint (as-of truncation) preds for month horizons
                save_kw['prefix_probs'] = pfx         # [n_models, n_patients, max_asof]
                save_kw['prefix_deltas'] = pfxd       # [n_patients, max_asof]
            np.savez(os.path.join(pred_dir, f'{stem}.npz'), **save_kw)
        n_pfx = sum(1 for t in to_save if t[5] is not None)
        print(f'Saved per-patient predictions for {len(to_save)} config(s) under {pred_dir}'
              f'{f" (+ per-timepoint prefixes for {n_pfx})" if n_pfx else ""}')
        # Prediction-horizon table: re-evaluate the same checkpoints on each horizon's test set
        # (labels_Nmonths.csv = observations >= N months before dx, relabelled for that horizon).
        if getattr(args, 'month_horizons', False):
            mh_dir = os.path.join(diag_dir, 'month_predictions')
            os.makedirs(mh_dir, exist_ok=True)
            # Re-enable MONAI metadata tracking: it was disabled at line ~340 after the standard
            # loader was cached, but the fresh per-horizon loaders rebuild their transform pipeline
            # (EnsureChannelFirstd needs the metadata to infer the channel dim), so meta must be on.
            set_track_meta(True)
            # Tag 1 = all observations (labels.csv); every other tag T loads labels_Tmonths.csv.
            for N in [t.strip() for t in args.horizon_files.split(',') if t.strip()]:
                try:
                    mdl, _ = load_data(args, device, phase='test', month_horizon=N)
                except FileNotFoundError as exc:
                    print(f'  horizon {N}mo: SKIPPED (missing label file: {exc})')
                    continue
                mloader = mdl['test'][0]
                for label, arch, stem, root in configs:
                    pm, lb, uid = evaluate_stem(args, mloader, device, arch, stem, root, num_classes)[:3]
                    if pm is not None:
                        # The ids tie each score to its patient, which subgroup and paired analyses need.
                        np.savez(os.path.join(mh_dir, f'{stem}_{N}months.npz'), probs=pm, labels=lb,
                                 uids=np.array(uid, dtype=object))
                        print(f'  horizon {N}mo: {stem} -> {pm.shape[1]} obs, {int(np.asarray(lb).sum())} pos')
            print(f'Saved month-horizon predictions under {mh_dir}')
        plot_curves(results, 'AUROC', os.path.join(diag_dir, 'roc_pr_curves', f'{tag}_roc.png'))
        plot_curves(results, 'AUPRC', os.path.join(diag_dir, 'roc_pr_curves', f'{tag}_pr.png'))
        plot_reliability(results, os.path.join(diag_dir, 'reliability', f'{tag}_reliability.png'))
        plot_cumulative_gain(results, 'AUPRC', os.path.join(diag_dir, 'cumulative_gain', f'{tag}_cumgain_auprc.png'))
        plot_cumulative_gain(results, 'AUROC', os.path.join(diag_dir, 'cumulative_gain', f'{tag}_cumgain_auroc.png'))
        print(f'Saved metrics + ROC/PR / reliability / cumulative-gain plots under {diag_dir}')

    if args.distributed:
        cleanup()
    print('Script finished')


if __name__ == '__main__':
    args = parse_args()
    main(args)
