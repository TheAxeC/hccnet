'''Diagnostic: what does SEG.nii.gz contain? Print unique labels + per-label voxel counts +
bounding-box extent for the HCC-positive test observations, so we know which label is the tumor.
Inference-free; pure I/O + numpy. Run via sbatch (never the login node).
'''
import os
import numpy as np
import nibabel as nib

ROOT = '/deepstore/datasets/bms/hcc_study/nifti'
OBS = ['PT_009_003', 'PT_034_001', 'PT_035_003', 'PT_100_003',
       'PT_254_004', 'PT_303_002', 'PT_315_001', 'PT_318_007']

for o in OBS:
    f = os.path.join(ROOT, o, 'SEG.nii.gz')
    if not os.path.isfile(f):
        print(f'{o}: NO SEG'); continue
    a = np.asarray(nib.load(f).dataobj)
    vals, cnts = np.unique(a, return_counts=True)
    parts = []
    for v, c in zip(vals, cnts):
        if v == 0:
            continue
        idx = np.argwhere(a == v)
        ext = (idx.max(0) - idx.min(0) + 1)
        parts.append(f'label {int(v)}: {int(c)} vox, bbox {tuple(int(x) for x in ext)}')
    print(f'{o}: shape {a.shape} | ' + ' ; '.join(parts) if parts else f'{o}: only background')
