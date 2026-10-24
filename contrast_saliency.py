"""Display joint DCE Layer-CAM on enhanced phases in a common physical plane."""
from pathlib import Path
import json
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from patient_figures import plane_geometry, sample


def render(folder, cam, affine, slice_index, probability, out):
    folder, out = Path(folder), Path(out)
    world, _, _ = plane_geometry(folder)
    # The model-native PLI grid's third axis is inferior; transform its chosen slice to RAS mm.
    world[..., 2] = nib.affines.apply_affine(affine, [0, 0, slice_index])[2]
    vox = nib.affines.apply_affine(np.linalg.inv(affine), world)
    heat = map_coordinates(cam, vox.reshape(-1,3).T, order=1, mode='constant', cval=0).reshape(world.shape[:2])
    heat = np.ma.masked_less(heat, .15)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for col, (mod, label) in enumerate([('T1A','Arterial'), ('T1V','Venous'), ('T1D','Late')]):
        arr = sample(folder/f'{mod}.nii.gz', world)
        for row in range(2):
            ax = axes[row,col]
            ax.imshow(arr, cmap='gray', origin='lower', vmin=0, vmax=1)
            if row:
                ax.imshow(heat, cmap='inferno', alpha=.55, origin='lower', vmin=0, vmax=1)
            ax.axis('off'); ax.set_title(label + (' + joint DCE Layer-CAM' if row else ''))
            ax.text(.02,.5,'R',color='white',transform=ax.transAxes)
            ax.text(.96,.5,'L',color='white',transform=ax.transAxes)
    fig.suptitle(f'{folder.name} | DCE Pico representative fold-0 model | p={probability:.3f}')
    fig.text(.5,.02,'Back at bottom | Same joint-input attention map on each phase; this does not measure phase-specific contribution.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.96))
    fig.savefig(out/f'{folder.name}_contrast_cam.png',dpi=140); plt.close(fig)
    (out/f'{folder.name}_contrast_cam.json').write_text(json.dumps({'visit':folder.name,'probability_single_model':probability,'world_z_mm':float(world[0,0,2]),'map':'joint four-channel DCE Layer-CAM; fold 0; latest model visit'},indent=2)+'\n')
