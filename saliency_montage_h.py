'''Horizontal saliency montages, one per modality: patients across the columns, native MRI on the
TOP row and the Layer-CAM overlay on the BOTTOM row (two images per patient, stacked). Splits each
per-patient localize figure (MRI | Layer-CAM) into its two panels. The four cases per modality are
the highest-confidence HCC-positive test patients (ensemble probability from the finetune npz).
'''
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(__file__), 'results/model_diagnostics/localize')
PRED_DIR = os.environ.get('HCC_PRED_DIR', '/private/tmp/claude-502/scratch_pred')
OUT = os.environ.get('HCC_BREAKDOWN_OUT', '/private/tmp/claude-502/breakdowns')
os.makedirs(OUT, exist_ok=True)
YT, YB, XM = 0.20, 0.99, 0.50   # crop below the panel titles; split MRI (left) vs Layer-CAM (right)

# display label -> (stem = localize subdir + npz basename, output name)
MODS = [('DW-MRI', 'dwi_pico_paper_ft'),
        ('T1 DCE-MRI', 't1dce_pico_paper_ft'),
        ('T1 IOP + T2 (combined)', 't1iop_pico_paper_ft'),
        ('T1 IOP alone', 't1iop_only_pico_paper_ft'),
        ('T2 alone', 't2_only_pico_paper_ft')]


def pick_cases(stem, k=4):
    '''The k highest-confidence HCC-positive test patients, as (patient id, p-tag). The npz uid is
    the patient's FIRST observation while the localize PNG is named by the MOST-RECENT one, so we
    key on the patient prefix (PT_xxx) and let build() resolve it to the actual PNG.'''
    d = np.load(os.path.join(PRED_DIR, f'{stem}.npz'), allow_pickle=True)
    probs = np.asarray(d['probs'], float); labels = np.asarray(d['labels']).ravel()
    uids = np.array([str(u) for u in d['uids']])
    ens = probs.mean(0)
    pos = np.where(labels == 1)[0]
    top = pos[np.argsort(ens[pos])[::-1][:k]]
    return [(uids[i].rsplit('_', 1)[0], f'p={ens[i]:.2f}') for i in top]


def build(subdir, out_name, label, cases):
    d = os.path.join(BASE, subdir)
    resolved = []
    for pref, tag in cases:                                   # patient prefix -> the patient's localize PNG
        m = sorted(glob.glob(os.path.join(d, f'{pref}_*.png')))
        if m:
            resolved.append((os.path.basename(m[0])[:-4], tag))
    cases = resolved
    if not cases:
        print(f'skip {label}: no case PNGs in {subdir}'); return
    n = len(cases)
    fig, axes = plt.subplots(2, n, figsize=(4.0 * n, 8.3), squeeze=False)
    for j, (pid, tag) in enumerate(cases):
        img = plt.imread(os.path.join(d, f'{pid}.png'))
        h, w = img.shape[:2]
        panels = [img[int(h * YT):int(h * YB), int(w * 0.02):int(w * XM), :],   # MRI (left)
                  img[int(h * YT):int(h * YB), int(w * XM):int(w * 0.99), :]]    # Layer-CAM (right)
        for i in range(2):
            ax = axes[i, j]
            ax.imshow(panels[i]); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(f'{pid}   {tag}', fontsize=12)
            if j == 0:
                ax.set_ylabel('MRI' if i == 0 else 'Layer-CAM', fontsize=13, weight='bold')
    fig.suptitle(f'HCCNet attention on liver parenchyma - {label}  (green = liver outline; HCC+ cases)',
                 fontsize=15, y=0.98)
    fig.text(0.5, 0.03, "Top: native MRI with liver outline.  Bottom: Layer-CAM on the model's most-attended "
             'slice - heat concentrates within the liver.', ha='center', fontsize=10, color='#444')
    plt.subplots_adjust(wspace=0.03, hspace=0.05, top=0.92, bottom=0.07)
    fig.savefig(os.path.join(OUT, out_name), dpi=150, bbox_inches='tight'); plt.close(fig)
    print('wrote', out_name)


def main():
    for label, stem in MODS:
        subdir = stem
        out_name = f'saliency_montage_{stem.split("_pico")[0]}.png'
        if not os.path.isdir(os.path.join(BASE, subdir)):
            print(f'skip {label}: localize dir {subdir} not present yet'); continue
        if not os.path.isfile(os.path.join(PRED_DIR, f'{stem}.npz')):
            print(f'skip {label}: npz {stem} not present'); continue
        build(subdir, out_name, label, pick_cases(stem))


if __name__ == '__main__':
    main()
