"""Private radiologist atlas from raw NIfTI and saved held-out prefix predictions."""
import argparse
import csv
import html
import json
import os
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODS = ['T1WI', 'T1A', 'T1V', 'T1D', 'DWI_b0', 'DWI_b150', 'DWI_b400', 'DWI_b800', 'T1W_IP', 'T1W_OOP', 'T2W_TES', 'T2W_TEL']
LABELS = ['Precontrast', 'Arterial', 'Venous', 'Late', 'DWI b=0', 'DWI b=150', 'DWI b=400', 'DWI b=800', 'T1 in-phase', 'T1 out-of-phase', 'T2 short TE', 'T2 long TE']
GROUPS = {'dwi': 'DWI', 't1dce': 'DCE (all phases)', 't1iop': 'T1 IOP + T2', 't1iop_only': 'T1 IOP', 't2_only': 'T2'}


def longitudinal(history, modalities, preds, pid, target):
    fig, axes = plt.subplots(len(modalities), len(history), figsize=(3*len(history), 3.3*len(modalities)), squeeze=False)
    for col, (uid, years, images) in enumerate(history):
        for row, mod in enumerate(modalities):
            ax = axes[row, col]
            if mod in images:
                ax.imshow(images[mod], cmap='gray', origin='lower', vmin=0, vmax=1)
            else:
                ax.text(.5,.5,'Unavailable',ha='center',transform=ax.transAxes)
            group = 'dwi' if mod.startswith('DWI') else 't2_only' if mod.startswith('T2') else 't1iop_only' if mod in ['T1W_IP','T1W_OOP'] else 't1dce'
            values = preds[group]
            matches = np.flatnonzero(np.isclose(values['years'], years))
            prob = f"p={values['prob'][int(matches[0])]:.3f}" if len(matches) else 'No prediction for this bin'
            ax.set_title(f'{uid}\n{years:g} yr before HCC' if row == 0 else LABELS[MODS.index(mod)], fontsize=9)
            ax.set_xlabel(prob, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(LABELS[MODS.index(mod)], fontsize=10)
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle(pid+' | Earlier to later surveillance | Pico ensemble probabilities', fontsize=12)
    fig.text(.5,.01,'Back at bottom | Liver-centroid slice per visit; no longitudinal registration | Same-year visits share a prediction bin',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.025,1,.96)); fig.savefig(target,dpi=135);plt.close(fig)


def predictions(root):
    result = {}
    cases = set()
    for group in GROUPS:
        path = root / f'{group}_pico_paper_ft.npz'
        with np.load(path, allow_pickle=True) as z:
            for i in np.flatnonzero(z['labels'].ravel() == 1):
                pid = str(z['uids'][i]).rsplit('_', 1)[0]
                cases.add(pid)
                d = z['prefix_deltas'][i]
                valid = np.isfinite(d)
                p = z['prefix_probs'][:, i, valid]
                if not np.isfinite(p).all():
                    raise ValueError(f'Nonfinite prediction: {pid}/{group}')
                if valid.any() and not np.allclose(p[:, -1], z['probs'][:, i], atol=1e-5):
                    raise ValueError(f'Final prefix differs: {pid}/{group}')
                result.setdefault(pid, {})[group] = {'years': d[valid].tolist(), 'prob': p.mean(0).tolist(), 'models': p.shape[0]}
    if len(cases) != 8:
        raise ValueError(f'Expected 8 held-out HCC cases, found {len(cases)}')
    return result


def plane_geometry(folder):
    """Axial RAS world grid; radiological display has patient right on image left."""
    ref_path = next((folder / f'{m}.nii.gz' for m in MODS if (folder / f'{m}.nii.gz').exists()), None)
    if ref_path is None:
        raise ValueError(f'No images in {folder}')
    ref = nib.as_closest_canonical(nib.load(ref_path))
    corners = nib.affines.apply_affine(ref.affine, np.array([[i, j, k] for i in (0, ref.shape[0]-1) for j in (0, ref.shape[1]-1) for k in (0, ref.shape[2]-1)]))
    low, high = corners.min(0), corners.max(0)
    z = (low[2] + high[2]) / 2
    method = 'image centre (no liver mask)'
    if (folder / 'SEG.nii.gz').exists():
        seg = nib.load(folder / 'SEG.nii.gz')
        coords = np.argwhere(np.asarray(seg.dataobj) == 1)
        if len(coords):
            centre = nib.affines.apply_affine(seg.affine, coords.mean(0))
            z = float(np.clip(centre[2], low[2], high[2]))
            method = 'liver-mask centroid'
    # Equal physical pixel spacing, preserving anatomical proportions.
    spacing = max(high[0]-low[0], high[1]-low[1]) / 320
    x = np.arange(high[0], low[0]-spacing/2, -spacing)
    y = np.arange(low[1], high[1]+spacing/2, spacing)
    xx, yy = np.meshgrid(x, y)
    world = np.stack([xx, yy, np.full_like(xx, z)], -1)
    return world, z, method


def sample(path, world):
    img = nib.load(path)
    if len(img.shape) != 3:
        raise ValueError(f'Expected 3D image: {path}: {img.shape}')
    vox = nib.affines.apply_affine(np.linalg.inv(img.affine), world)
    arr = map_coordinates(np.asarray(img.dataobj, dtype=np.float32), vox.reshape(-1, 3).T, order=1, mode='constant', cval=np.nan).reshape(world.shape[:2])
    finite = arr[np.isfinite(arr) & (arr != 0)]
    if len(finite):
        lo, hi = np.percentile(finite, [1, 99])
        arr = np.clip((arr-lo)/max(hi-lo, 1e-8), 0, 1)
    return arr


def panels(images, order, title, target, ncols):
    nrows = int(np.ceil(len(order)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2*ncols, 3.4*nrows), squeeze=False, facecolor='white')
    for ax, m in zip(axes.flat, order):
        if m in images:
            ax.imshow(images[m], cmap='gray', origin='lower', vmin=0, vmax=1)
            ax.text(.02, .5, 'R', color='white', transform=ax.transAxes, va='center')
            ax.text(.96, .5, 'L', color='white', transform=ax.transAxes, va='center')
        else:
            ax.text(.5, .5, 'Not acquired / unavailable', ha='center', transform=ax.transAxes, fontsize=9)
        ax.set_title(LABELS[MODS.index(m)], fontsize=11)
        ax.axis('off')
    for ax in list(axes.flat)[len(order):]:
        ax.axis('off')
    fig.suptitle(title, fontsize=12)
    fig.text(.5, .012, 'Back at bottom | Same physical axial plane within visit | Separate 1-99% intensity windows', ha='center', fontsize=8)
    fig.tight_layout(rect=(0,.03,1,.96))
    fig.savefig(target, dpi=130)
    plt.close(fig)


def trajectory(pid, preds, target):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for group, vals in preds.items():
        ax.plot(-np.asarray(vals['years']), vals['prob'], 'o-', label=GROUPS[group], ms=5)
    ax.set(xlabel='Years relative to HCC diagnosis (recorded whole-year bins)', ylabel='Model probability', ylim=(-.03,1.03), title=f'{pid} | Growing scan history | Pico, mean of 50 models')
    ax.axvline(0, color='#777777', linestyle=':'); ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc='best', ncol=2)
    fig.tight_layout(); fig.savefig(target, dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(os.environ.get('HCCNET_DATA_DIR', Path.home()/'data/hccnet-mri')))
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not (args.data/'labels/labels.csv').exists():
        parser.error('Missing raw cohort: set --data to the institutional hcc_study directory containing labels/ and nifti/. Obtain access from the cohort data steward.')
    preds = predictions(args.predictions)
    with (args.data/'labels/labels.csv').open() as f:
        rows = list(csv.DictReader(f))
    args.out.mkdir(parents=True, exist_ok=True)
    inventory, links = [], []
    for pid, pr in sorted(preds.items()):
        dest = args.out/pid; dest.mkdir(exist_ok=True)
        trajectory(pid, pr, dest/'trajectory.png')
        visits = sorted([r for r in rows if r['id'] == pid], key=lambda r: (r['date'], float(r['observation'])))
        history = []
        body = [f'<h1>{pid}</h1>', '<img src="trajectory.png">', '<p>Predictions use only scans available at each recorded year bin. Same-year visits are grouped. Lines connect observed predictions; they do not estimate intervening risk. All models are Pico for a matched size comparison. Each modality may have a different usable scan history.</p>']
        for row in visits:
            uid = f"{pid}_{int(float(row['observation'])):03d}"
            folder = args.data/'nifti'/uid
            if not folder.is_dir():
                inventory.append({'patient':pid, 'visit':uid, 'years':row['delta'], 'missing':'entire visit'})
                body.append(f'<h2>{uid}: images unavailable</h2>'); continue
            world, z, method = plane_geometry(folder)
            images = {m: sample(folder/f'{m}.nii.gz', world) for m in MODS if (folder/f'{m}.nii.gz').exists()}
            history.append((uid, float(row['delta']), images))
            unit = 'year' if float(row['delta']) == 1 else 'years'
            title = f"{uid} | {row['delta']} {unit} before diagnosis | {method}"
            panels(images, MODS, title, dest/f'{uid}_all.png', 4)
            panels(images, MODS[:4], title, dest/f'{uid}_dce.png', 4)
            panels(images, ['T1A','DWI_b0','DWI_b150','T1D'], title, dest/f'{uid}_comparison.png', 4)
            body.extend([f'<h2>{uid} | {row["delta"]} years before diagnosis</h2>', f'<img src="{uid}_all.png">', f'<details><summary>DCE phases</summary><img src="{uid}_dce.png"></details>', f'<details><summary>Arterial / low-b DWI / late comparison</summary><img src="{uid}_comparison.png"></details>'])
            for m in MODS:
                inventory.append({'patient':pid, 'visit':uid, 'years':row['delta'], 'modality':m, 'available':m in images, 'world_z_mm':round(z,3), 'slice_selection':method})
            print(uid, len(images), 'modalities', flush=True)
        for name, mods in [('dce',MODS[:4]), ('dwi',MODS[4:8]), ('structural',MODS[8:])]:
            longitudinal(history, mods, pr, pid, dest/f'longitudinal_{name}.png')
        body.insert(3, '<h2>Images through time</h2>'+''.join(f'<details><summary>{name.upper()} through time</summary><img src="longitudinal_{name}.png"></details>' for name in ['dce','dwi','structural']))
        body.insert(1, '<p>Slice selected at the liver centroid independently for each visit. NIfTI coordinates match the plane within a visit; respiratory motion is not corrected and visits are not registered. SEG is an organ mask, not a tumor annotation. Phase names follow the dataset file labels; acquisition timing is unavailable.</p>')
        (dest/'index.html').write_text(page('\n'.join(body)))
        links.append(f'<li><a href="{pid}/index.html">{pid}</a> ({len(visits)} labelled visits)</li>')
    (args.out/'index.html').write_text(page('<h1>HCC patient MRI atlas</h1><p>Eight held-out HCC cases. Private radiologist review.</p><ul>'+''.join(links)+'</ul>'))
    (args.out/'inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
    (args.out/'predictions.json').write_text(json.dumps(preds, indent=2)+'\n')


def page(body):
    return '<!doctype html><meta charset="utf-8"><title>HCC MRI atlas</title><style>body{font:16px system-ui;margin:30px auto;max-width:1400px;padding:0 20px}img{max-width:100%}h2{margin-top:50px}p{max-width:1000px;line-height:1.5}summary{cursor:pointer}</style>'+body


if __name__ == '__main__':
    main()
