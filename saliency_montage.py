'''Assemble a clean saliency montage for the meeting from the per-patient Layer-CAM liver-attention
figures (results/model_diagnostics/localize/<stem>/PT_*.png). Each source panel is MRI (liver
outlined) + Layer-CAM; here we stack a few high-confidence HCC+ cases with a single title and a
short per-row label. No recompute - just composition of figures already generated.
'''
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC = os.environ.get('HCC_LOCALIZE_DIR',
    os.path.join(os.path.dirname(__file__), 'results/model_diagnostics/localize/dwi_tiny_paper_ft'))
OUT = os.environ.get('HCC_BREAKDOWN_OUT', '/private/tmp/claude-502/breakdowns')
os.makedirs(OUT, exist_ok=True)
# high-confidence HCC+ cases with attention on the liver (p ~ 1.0)
CASES = [('PT_318_007', 'p=1.00'), ('PT_254_004', 'p=1.00'),
         ('PT_035_003', 'p=0.99'), ('PT_034_001', 'p=0.99')]
CROP_TOP = 0.11   # trim the per-figure suptitle strip


def main():
    n = len(CASES)
    fig, axes = plt.subplots(n, 1, figsize=(9, 4.3 * n))
    for ax, (pid, tag) in zip(np.atleast_1d(axes), CASES):
        img = plt.imread(os.path.join(SRC, f'{pid}.png'))
        img = img[int(img.shape[0] * CROP_TOP):, :, :]           # drop redundant suptitle
        ax.imshow(img); ax.axis('off')
        ax.set_ylabel(f'{pid}\n{tag}', rotation=0, ha='right', va='center', fontsize=11, labelpad=30)
        ax.set_frame_on(False)
    fig.suptitle('Layer-CAM attention on diffusion-weighted images, four patients who developed HCC\n(green outline = liver)', fontsize=13, y=0.997)
    fig.text(0.5, 0.005, 'Per-patient Layer-CAM on the model\'s most-attended slice; heat concentrates '
             'within the liver, not background or adjacent organs', ha='center', fontsize=9, color='#444')
    plt.subplots_adjust(hspace=0.05, top=0.97, bottom=0.02)
    fig.savefig(os.path.join(OUT, 'saliency_montage.png'), dpi=150, bbox_inches='tight'); plt.close(fig)
    print('wrote saliency_montage.png ->', OUT)


if __name__ == '__main__':
    main()
