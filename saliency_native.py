'''Native-space Grad-CAM overlay for HCCNet (radiologist-grade).

Computes Grad-CAM on the model's 72^3 crop, then maps it BACK onto the ORIGINAL scan and
overlays it in full anatomical context with the shipped SEG.nii.gz contour drawn for
reference / placement validation. Inference + one backward pass; no training.

Why this is built the way it is: the training pipeline's PercentileSpatialCropd is a plain
(non-invertible) array-slice, so MONAI's inverse silently drops it. Here the geometry is
reproduced with MONAI's INVERTIBLE SpatialCropd (same box the percentile logic yields) applied
step by step, so the recorded ops invert cleanly all the way back to native. The display scan
and SEG are loaded through the same Orientation so everything shares one coordinate space.
'''
import os
import argparse
import numpy as np
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
    LoadImage, EnsureChannelFirst, Orientation, CropForeground, KeepLargestConnectedComponent)
import numpy.ma as ma
from data.transforms import YeoJohnsond
from utils.preprocessing import load_data
from utils.config import parse_args
from utils.utils import prep_batch
from test import build_model

N_POSITIVES = 25
TARGET_STAGE = 2


def head(modalities):
    '''Native load -> orient -> YeoJohnson -> resample-to-mod0 -> concat; 'image' created here.'''
    return Compose([
        LoadImaged(keys=modalities, image_only=False),
        EnsureChannelFirstd(keys=modalities),
        Orientationd(keys=modalities, axcodes='PLI'),
        YeoJohnsond(keys=modalities, lmbda=0.5),
        ResampleToMatchd(keys=modalities, key_dst=modalities[0], mode=3),
        ConcatItemsd(keys=modalities, name='image'),
        DeleteItemsd(keys=list(modalities)),
    ])


def pct_box(shape, center, size, min_size):
    '''The exact box PercentileSpatialCropd slices (fractional center/size + min-size floor).'''
    s, e = [], []
    for i in range(len(shape)):
        c = int(center[i] * shape[i]); sz = int(size[i] * shape[i])
        st = int(c - sz / 2); en = int(c + sz / 2)
        if en - st < min_size[i]:
            en = min_size[i] + st
        s.append(max(st, 0)); e.append(min(en, shape[i]))
    return s, e


def grad_cam(model, x, pad_mask, pos):
    acts = {}

    def hook(_m, _i, o):
        o.retain_grad(); acts['A'] = o
    handle = model.backbone.stages[TARGET_STAGE].register_forward_hook(hook)
    logit = model(x, pad_mask=pad_mask, pos=pos)
    score = logit.squeeze()
    model.zero_grad(set_to_none=True)
    score.backward()
    handle.remove()
    A = acts['A']
    w = A.grad.mean(dim=(2, 3, 4), keepdim=True)
    cam = torch.relu((w * A).sum(dim=1))
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
    disp = Compose([EnsureChannelFirst(), Orientation(axcodes='PLI')])   # scan/SEG display space
    nifti_dir = os.path.join(args.data_dir, 'nifti')
    out_dir = os.path.join(args.results_dir, 'model_diagnostics', 'native', stem)
    os.makedirs(out_dir, exist_ok=True)
    print(f'##### NATIVE OVERLAY {stem}: stage {TARGET_STAGE} #####')

    n = 0
    for i, batch in enumerate(loader):
        inputs, label, delta, pad = prep_batch(batch, batch_size=1, device=device)
        if int(np.asarray(label.detach().cpu()).reshape(-1)[0]) != 1:
            continue
        uids = batch.get('uid', None)
        try:
            cam, prob = grad_cam(model, inputs, pad, delta)
            pm = np.asarray(pad.detach().cpu()).reshape(-1)
            real = np.where(pm < 0.5)[0]
            ts = int(real.max()) if real.size else 0
            obs_uid = str(uids[ts][0] if isinstance(uids[ts], (list, tuple)) else uids[ts])
            obs_dir = os.path.join(nifti_dir, obs_uid)
            files = {m: os.path.join(obs_dir, f'{m}.nii.gz') for m in modalities}
            if not all(os.path.isfile(f) for f in files.values()):
                print(f'  {obs_uid}: SKIP (missing native files)'); continue

            d = head_t(dict(files))                                # post-concat 'image' (oriented, mod0 grid)
            ops = []
            def apply(t):                                          # apply an invertible op + record it
                nonlocal d
                d = t(d); ops.append(t)
            b = pct_box(d['image'].shape[1:], (0.5, 0.5, 0.5), (0.85, 0.8, 0.99), (82, 82, 82))
            apply(SpatialCropd(keys='image', roi_start=b[0], roi_end=b[1]))
            img0 = d['image'][0:1]
            mask = klc((img0 > img0.mean()).to(torch.uint8))
            fs, fe = cf.compute_bounding_box(mask)
            apply(SpatialCropd(keys='image', roi_start=list(fs), roi_end=list(fe)))
            apply(Spacingd(keys='image', pixdim=(1.5, 1.5, 1.5), mode=3))
            b2 = pct_box(d['image'].shape[1:], (0.5, 0.3, 0.3), (0.6, 0.5, 0.4), (82, 82, 82))
            apply(SpatialCropd(keys='image', roi_start=b2[0], roi_end=b2[1]))
            apply(SpatialPadd(keys='image', spatial_size=(72, 72, 72)))
            apply(CenterSpatialCropd(keys='image', roi_size=(96, 96, 96)))
            apply(CenterSpatialCropd(keys='image', roi_size=(72, 72, 72)))
            crop = d['image']

            cam72 = F.interpolate(cam[ts][None, None].float(), size=crop.shape[-3:],
                                  mode='trilinear', align_corners=False)[0]
            cam72 = (cam72 / (cam72.max() + 1e-8)).repeat(crop.shape[0], 1, 1, 1)
            cam_meta = crop.clone(); cam_meta.copy_(cam72.to(crop.device))
            d['image'] = cam_meta
            inv = Invertd(keys='image', transform=Compose(ops), orig_keys='image')(d)
            nc = np.asarray(inv['image'].detach().cpu())
            native_cam = nc.max(axis=0) if nc.ndim == 4 else nc     # [H,W,D] on the oriented native grid

            # Display scan + SEG on ONE shared grid (mod0, PLI) that native_cam also lives on.
            seg_f = os.path.join(obs_dir, 'SEG.nii.gz')
            dd = {'scan': files[modalities[0]], 'seg': seg_f}
            keys = ['scan'] + (['seg'] if os.path.isfile(seg_f) else [])
            dd = Compose([
                LoadImaged(keys=keys, image_only=True),
                EnsureChannelFirstd(keys=keys),
                Orientationd(keys=keys, axcodes='PLI'),
                *( [ResampleToMatchd(keys='seg', key_dst='scan', mode='nearest')] if 'seg' in keys else [] ),
            ])(dd)
            scan = np.asarray(dd['scan'][0])
            seg = np.asarray(dd['seg'][0]) if 'seg' in keys else None

            cam_n = native_cam / (native_cam.max() + 1e-8)
            zc = int((cam_n > 0.15).sum(axis=(0, 1)).argmax())     # slice with most salient support
            hm = ma.masked_where(cam_n < 0.15, cam_n)              # only heat the CAM support; anatomy stays visible
            fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
            for a in ax:
                a.imshow(window(scan[:, :, zc]).T, cmap='gray', origin='lower'); a.axis('off')
            ax[0].set_title('native MRI')
            ax[1].imshow(hm[:, :, zc].T, cmap='jet', alpha=0.55, origin='lower', vmin=0, vmax=1)
            if seg is not None and seg.shape == scan.shape and seg[:, :, zc].any():
                ax[1].contour(seg[:, :, zc].T > 0, colors='lime', linewidths=1.2)
            ax[1].set_title(f'Grad-CAM + SEG  (p={prob:.2f})')
            plt.suptitle(f'{stem}  |  {obs_uid} (HCC+)  |  native space')
            safe = ''.join(c if c.isalnum() else '_' for c in obs_uid)
            plt.savefig(os.path.join(out_dir, f'{safe}.png'), dpi=150, bbox_inches='tight')
            plt.close()
            n += 1
            print(f'  {obs_uid}: p={prob:.3f}, native_cam {native_cam.shape}, scan {scan.shape}')
        except Exception as exc:
            import traceback
            print(f'  patient #{i}: SKIPPED ({type(exc).__name__}: {exc})')
            traceback.print_exc()
        if n >= N_POSITIVES:
            break
    print(f'##### NATIVE OVERLAY DONE: {n} under {out_dir} #####')
    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main(parse_args())
