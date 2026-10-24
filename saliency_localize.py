'''Localization saliency for HCCNet: WHERE in the liver does the model look?

Layer-CAM (fine, class-discriminative) at a shallow ConvNeXt stage, evaluated against the
confirmed liver label (SEG has NO tumor mask - it is a liver/spleen/kidney organ map). The liver
mask is carried through the SAME geometry the image goes through, so the CAM and liver live on the
identical 72^3 grid the model actually sees - no inverse-interpolation smearing. Per HCC+ patient:
  * peak-in-liver  : is the CAM argmax inside the liver? (attends to liver vs spillover organs)
  * energy-in-liver: fraction of total CAM mass inside the liver.
Actual TUMOR placement is confirmed qualitatively by the project radiologist on these figures.
Inference + one backward pass; no training.
'''
import os
import numpy as np
import numpy.ma as ma
import torch
import torch.nn.functional as F
import torch.distributed as dist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from monai.data import set_track_meta
from monai.utils import set_determinism
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ConcatItemsd,
    CenterSpatialCropd, ResampleToMatchd, DeleteItemsd, SpatialPadd, SpatialCropd, Invertd,
    CropForeground, KeepLargestConnectedComponent)
from data.transforms import YeoJohnsond
from utils.preprocessing import load_data
from utils.config import parse_args
from utils.utils import prep_batch
from test import build_model

N_POSITIVES = 25
TARGET_STAGE = 1          # ~12^3 on the 72^3 crop: fine enough to point at a >=2 cm nodule
LIVER_LABEL = 1           # SEG label 1 = liver (largest organ; confirmed by seg_viz). SEG has NO tumor mask.


def head(modalities):
    '''Load modalities + seg, orient, YeoJohnson (image only), resample-to-mod0, concat image; keep seg.'''
    keys = list(modalities) + ['seg']
    return Compose([
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes='PLI'),
        YeoJohnsond(keys=modalities, lmbda=0.5),
        ResampleToMatchd(keys=modalities, key_dst=modalities[0], mode=3),
        ResampleToMatchd(keys='seg', key_dst=modalities[0], mode='nearest'),
        ConcatItemsd(keys=modalities, name='image'),
        DeleteItemsd(keys=list(modalities)),
    ])


def pct_box(shape, center, size, min_size):
    s, e = [], []
    for i in range(len(shape)):
        c = int(center[i] * shape[i]); sz = int(size[i] * shape[i])
        st = int(c - sz / 2); en = int(c + sz / 2)
        if en - st < min_size[i]:
            en = min_size[i] + st
        s.append(max(st, 0)); e.append(min(en, shape[i]))
    return s, e


def layer_cam(model, x, pad_mask, pos):
    '''Layer-CAM: elementwise relu(grad) * activation, summed over channels then relu.'''
    acts = {}

    def hook(_m, _i, o):
        o.retain_grad(); acts['A'] = o
    handle = model.backbone.stages[TARGET_STAGE].register_forward_hook(hook)
    logit = model(x, pad_mask=pad_mask, pos=pos)
    score = logit.squeeze()
    model.zero_grad(set_to_none=True)
    score.backward()
    handle.remove()
    A = acts['A']                                        # [B*S, C, h, w, d]
    cam = torch.relu((torch.relu(A.grad) * A).sum(dim=1))  # [B*S, h, w, d]
    return cam.detach(), float(torch.sigmoid(score.detach()).reshape(-1)[0])


def window(a):
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)


def main(args):
    set_determinism(seed=args.seed)
    if args.distributed:
        dist.init_process_group(backend='nccl')
    device = (dist.get_rank() % torch.cuda.device_count()) if torch.cuda.is_available() else 'cpu'
    set_track_meta(True)

    dataloader, _ = load_data(args, device, phase='test', max_delta=args.max_delta)
    loader = dataloader['test'][0]

    stem = args.suffix
    model = build_model(args, args.arch, 1, device)
    model.load_state_dict(torch.load(os.path.join(args.results_dir, 'model_weights',
                                                   f'weights_fold0_{stem}.pth'), map_location='cpu'))
    model.eval()

    modalities = args.mod_list
    head_t = head(modalities)
    klc = KeepLargestConnectedComponent(connectivity=1)
    cf = CropForeground(select_fn=lambda x: x > 0, margin=0, allow_smaller=False)
    nifti_dir = os.path.join(args.data_dir, 'nifti')
    out_dir = os.path.join(args.results_dir, 'model_diagnostics', 'localize', stem)
    os.makedirs(out_dir, exist_ok=True)
    print(f'##### LOCALIZE {stem}: Layer-CAM stage {TARGET_STAGE}, liver label {LIVER_LABEL} #####')

    hits, energies, n = [], [], 0
    for i, batch in enumerate(loader):
        inputs, label, delta, pad = prep_batch(batch, batch_size=1, device=device)
        if int(np.asarray(label.detach().cpu()).reshape(-1)[0]) != 1:
            continue
        uids = batch.get('uid', None)
        try:
            cam, prob = layer_cam(model, inputs, pad, delta)
            pm = np.asarray(pad.detach().cpu()).reshape(-1)
            real = np.where(pm < 0.5)[0]
            ts = int(real.max()) if real.size else 0
            obs_uid = str(uids[ts][0] if isinstance(uids[ts], (list, tuple)) else uids[ts])
            obs_dir = os.path.join(nifti_dir, obs_uid)
            files = {m: os.path.join(obs_dir, f'{m}.nii.gz') for m in modalities}
            files['seg'] = os.path.join(obs_dir, 'SEG.nii.gz')
            if not all(os.path.isfile(f) for f in files.values()):
                print(f'  {obs_uid}: SKIP (missing files)'); continue

            d = head_t(dict(files))
            ops = []
            def apply(t):
                nonlocal d
                d = t(d); ops.append(t)
            b = pct_box(d['image'].shape[1:], (0.5, 0.5, 0.5), (0.85, 0.8, 0.99), (82, 82, 82))
            apply(SpatialCropd(keys=['image', 'seg'], roi_start=b[0], roi_end=b[1]))
            img0 = d['image'][0:1]
            mask = klc((img0 > img0.mean()).to(torch.uint8))
            fs, fe = cf.compute_bounding_box(mask)
            apply(SpatialCropd(keys=['image', 'seg'], roi_start=list(fs), roi_end=list(fe)))
            apply(Spacingd(keys=['image', 'seg'], pixdim=(1.5, 1.5, 1.5), mode=(3, 'nearest')))
            b2 = pct_box(d['image'].shape[1:], (0.5, 0.3, 0.3), (0.6, 0.5, 0.4), (82, 82, 82))
            apply(SpatialCropd(keys=['image', 'seg'], roi_start=b2[0], roi_end=b2[1]))
            apply(SpatialPadd(keys=['image', 'seg'], spatial_size=(72, 72, 72)))
            apply(CenterSpatialCropd(keys=['image', 'seg'], roi_size=(96, 96, 96)))
            apply(CenterSpatialCropd(keys=['image', 'seg'], roi_size=(72, 72, 72)))
            crop = d['image']
            liver72 = (np.asarray(d['seg'][0].detach().cpu()) == LIVER_LABEL)

            cam72 = F.interpolate(cam[ts][None, None].float(), size=crop.shape[-3:],
                                  mode='trilinear', align_corners=False)[0, 0]
            cam72 = (cam72 / (cam72.max() + 1e-8))
            c = np.asarray(cam72.detach().cpu())

            if liver72.sum() == 0:
                print(f'  {obs_uid}: p={prob:.3f} liver OUTSIDE crop (skip metric)')
            else:
                peak = np.unravel_index(int(c.argmax()), c.shape)
                hit = bool(liver72[peak])
                efrac = float(c[liver72].sum() / (c.sum() + 1e-8))
                hits.append(hit); energies.append(efrac)
                print(f'  {obs_uid}: p={prob:.3f} peak-in-liver={hit} energy-in-liver={efrac:.3f} '
                      f'(liver {int(liver72.sum())} vox)')

            # native-space display on the tumor-centroid slice
            cam_meta = crop.clone(); cam_meta.copy_(cam72[None].repeat(crop.shape[0], 1, 1, 1).to(crop.device))
            d2 = dict(d); d2['image'] = cam_meta
            native_cam = np.asarray(Invertd(keys='image', transform=Compose(ops),
                                            orig_keys='image')(d2)['image'].detach().cpu())
            native_cam = native_cam.max(axis=0) if native_cam.ndim == 4 else native_cam

            dd = Compose([
                LoadImaged(keys=['scan', 'seg'], image_only=True),
                EnsureChannelFirstd(keys=['scan', 'seg']),
                Orientationd(keys=['scan', 'seg'], axcodes='PLI'),
                ResampleToMatchd(keys='seg', key_dst='scan', mode='nearest'),
            ])({'scan': files[modalities[0]], 'seg': files['seg']})
            scan = np.asarray(dd['scan'][0]); seg_n = np.asarray(dd['seg'][0]) == LIVER_LABEL
            zc = int(native_cam.sum(axis=(0, 1)).argmax())   # slice the model most attends to (for radiologist to judge vs lesion)

            cam_n = native_cam / (native_cam.max() + 1e-8)
            if os.environ.get('HCCNET_CONTRAST_CAM') == '1':
                from contrast_saliency import render
                render(obs_dir, cam_n, np.asarray(dd['scan'].affine), zc, prob, out_dir)
            hm = ma.masked_where(cam_n < 0.15, cam_n)
            fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
            for a in ax:
                a.imshow(window(scan[:, :, zc]).T, cmap='gray', origin='lower'); a.axis('off')
            ax[0].set_title('MRI (liver outlined)')
            if seg_n[:, :, zc].any():
                ax[0].contour(seg_n[:, :, zc].T, colors='lime', linewidths=1.2)
            ax[1].imshow(hm[:, :, zc].T, cmap='jet', alpha=0.55, origin='lower', vmin=0, vmax=1)
            if seg_n[:, :, zc].any():
                ax[1].contour(seg_n[:, :, zc].T, colors='lime', linewidths=1.2)
            ax[1].set_title(f'Layer-CAM  (p={prob:.2f})')
            plt.suptitle(f'{stem}  |  {obs_uid} (HCC+)  |  attention on liver parenchyma')
            safe = ''.join(ch if ch.isalnum() else '_' for ch in obs_uid)
            plt.savefig(os.path.join(out_dir, f'{safe}.png'), dpi=150, bbox_inches='tight')
            plt.close()
            n += 1
        except Exception as exc:
            import traceback
            print(f'  patient #{i}: SKIPPED ({type(exc).__name__}: {exc})'); traceback.print_exc()
        if n >= N_POSITIVES:
            break
    if hits:
        print(f'##### {stem} LOCALIZATION: peak-in-liver {sum(hits)}/{len(hits)} '
              f'({100*sum(hits)/len(hits):.0f}%) | mean energy-in-liver {np.mean(energies):.3f} #####')
    print(f'##### LOCALIZE DONE: {n} figures under {out_dir} #####')
    if os.environ.get('HCCNET_CONTRAST_CAM') == '1' and n != 8:
        raise RuntimeError(f'Expected all 8 HCC cases, rendered {n}')
    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main(parse_args())
