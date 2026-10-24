'''Identify SEG labels: overlay each label (1-4) in a distinct color on an anatomical scan,
rendered on the SEG grid (SEG is finer than DWI), at three depths, for a few HCC-positive
observations. MONAI affine resample brings the scan onto the SEG grid. Run via sbatch.
'''
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, Orientationd, ResampleToMatchd)

ROOT = '/deepstore/datasets/bms/hcc_study/nifti'
OUT = os.path.join(os.environ['HOME'], 'projects/hccnet/results/model_diagnostics/seg_viz')
os.makedirs(OUT, exist_ok=True)
OBS = ['PT_318_007', 'PT_254_004', 'PT_034_001']
ANAT = 'T2W_TEL'                                 # T2 shows liver lesions well
COLORS = ['red', 'cyan', 'yellow', 'magenta']    # labels 1,2,3,4


def win(a):
    lo, hi = np.percentile(a, [1, 99]); return np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)


for o in OBS:
    d = {'scan': os.path.join(ROOT, o, f'{ANAT}.nii.gz'), 'seg': os.path.join(ROOT, o, 'SEG.nii.gz')}
    if not all(os.path.isfile(v) for v in d.values()):
        print(f'{o}: missing files'); continue
    d = Compose([
        LoadImaged(keys=['scan', 'seg'], image_only=True),
        EnsureChannelFirstd(keys=['scan', 'seg']),
        Orientationd(keys=['scan', 'seg'], axcodes='PLI'),
        ResampleToMatchd(keys='scan', key_dst='seg', mode=3),
    ])(d)
    scan = np.asarray(d['scan'][0]); seg = np.asarray(d['seg'][0])
    zc = int((seg > 0).sum(axis=(0, 1)).argmax())
    zs = [max(0, zc - 20), zc, min(seg.shape[2] - 1, zc + 20)]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
    for a, z in zip(ax, zs):
        a.imshow(win(scan[:, :, z]).T, cmap='gray', origin='lower'); a.axis('off'); a.set_title(f'z={z}')
        for li, col in zip([1, 2, 3, 4], COLORS):
            m = seg[:, :, z] == li
            if m.any():
                a.contour(m.T, colors=col, linewidths=1.3)
    handles = [plt.Line2D([0], [0], color=c, lw=2, label=f'label {i+1}') for i, c in enumerate(COLORS)]
    fig.legend(handles=handles, loc='lower center', ncol=4)
    plt.suptitle(f'{o}  |  SEG labels on {ANAT} (SEG grid)')
    plt.savefig(os.path.join(OUT, f'{o}_labels.png'), dpi=140, bbox_inches='tight')
    plt.close()
    print(f'{o}: wrote overlay (zc={zc}, scan {scan.shape})')
print(f'DONE -> {OUT}')
