'''Grad-CAM saliency for HCCNet: where does the model look?

Loads a representative fold-0 checkpoint (weights_fold0_<suffix>.pth), runs the held-out test
set, and for the first N HCC-positive patients computes Grad-CAM on a ConvNeXt stage of the CNN
backbone (class-discriminative localisation) for the MOST-RECENT scan in each patient's sequence.
Saves the heatmap overlaid on a REAL single-channel slice of that scan (not a max-projection),
plus the raw CAM volume. Inference + one backward pass; no training.

TARGET_STAGE picks the ConvNeXt stage to hook: on the 72^3 liver-crop input the stages sit at
~24/12/6/3 voxels (0..3). Stage 2 (~6^3) is deep/semantic but not the coarsest; drop to 1 for
finer spatial detail, raise to 3 for the most class-specific-but-blobby map.
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
from utils.preprocessing import load_data
from utils.config import parse_args
from utils.utils import prep_batch
from test import build_model

N_POSITIVES = 25   # cover all HCC-positive test patients, so the same UID can be matched across modalities
TARGET_STAGE = 2


def grad_cam(model, x, pad_mask, pos):
    '''Return (cam [S,h,w,d], prob) - Grad-CAM per scan at the target ConvNeXt stage.'''
    acts = {}

    def hook(_m, _i, o):
        o.retain_grad()
        acts['A'] = o
    handle = model.backbone.stages[TARGET_STAGE].register_forward_hook(hook)
    logit = model(x, pad_mask=pad_mask, pos=pos)
    score = logit.squeeze()
    model.zero_grad(set_to_none=True)
    score.backward()
    handle.remove()
    A = acts['A']                                   # [B*S, C, h, w, d]
    weights = A.grad.mean(dim=(2, 3, 4), keepdim=True)
    cam = torch.relu((weights * A).sum(dim=1))      # [B*S, h, w, d]
    return cam.detach(), float(torch.sigmoid(score.detach()).reshape(-1)[0])


def main(args: argparse.Namespace) -> None:
    set_determinism(seed=args.seed)
    if args.distributed:
        dist.init_process_group(backend='nccl')
    device = (dist.get_rank() % torch.cuda.device_count()) if torch.cuda.is_available() else 'cpu'

    dataloader, _ = load_data(args, device, phase='test', max_delta=args.max_delta)
    loader = dataloader['test'][0]
    set_track_meta(False)

    stem = args.suffix
    ckpt = os.path.join(args.results_dir, 'model_weights', f'weights_fold0_{stem}.pth')
    if not os.path.isfile(ckpt):
        print(f'ABORT: checkpoint not found: {ckpt}')
        return
    model = build_model(args, args.arch, 1, device)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()

    out_dir = os.path.join(args.results_dir, 'model_diagnostics', 'gradcam', stem)
    os.makedirs(out_dir, exist_ok=True)
    print(f'##### GRAD-CAM {stem}: ConvNeXt stage {TARGET_STAGE}, up to {N_POSITIVES} HCC+ patients #####')

    n = 0
    for i, batch in enumerate(loader):
        inputs, label, delta, pad = prep_batch(batch, batch_size=1, device=device)
        if int(np.asarray(label.detach().cpu()).reshape(-1)[0]) != 1:   # HCC-positive only
            continue
        u = batch.get('uid', ['?'])
        uid = str(u[0] if isinstance(u, (list, tuple)) else u)
        safe = ''.join(c if c.isalnum() else '_' for c in uid)          # filename-safe patient id
        try:
            cam, prob = grad_cam(model, inputs, pad, delta)             # cam [S,h,w,d]
            pm = np.asarray(pad.detach().cpu()).reshape(-1)             # 1 = padding
            real = np.where(pm < 0.5)[0]
            ts = int(real.max()) if real.size else 0                   # most-recent real scan
            cam_up = F.interpolate(cam[ts][None, None].float(), size=inputs.shape[-3:],
                                   mode='trilinear', align_corners=False)[0, 0].cpu().numpy()
            cam_up = cam_up / (cam_up.max() + 1e-8)
            ref = inputs[0, ts, 0].detach().cpu().numpy()              # REAL channel-0 volume of that scan
            zc = int(cam_up.sum(axis=(0, 1)).argmax())                 # most-salient slice
            sl_cam = cam_up[:, :, zc]
            sl = ref[:, :, zc]                                          # clinical windowing (1-99th pct) for a natural MRI look
            lo, hi = np.percentile(sl, [1, 99])
            sl_img = np.clip((sl - lo) / (hi - lo + 1e-8), 0, 1)
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax[0].imshow(sl_img, cmap='gray'); ax[0].set_title('MRI (real slice)'); ax[0].axis('off')
            ax[1].imshow(sl_img, cmap='gray')
            ax[1].imshow(sl_cam, cmap='jet', alpha=0.5)
            ax[1].set_title(f'Grad-CAM  (p={prob:.2f})'); ax[1].axis('off')
            plt.suptitle(f'{stem}  |  patient {uid} (HCC+), scan {ts}')
            plt.savefig(os.path.join(out_dir, f'patient_{safe}.png'), dpi=150, bbox_inches='tight')
            plt.close()
            np.save(os.path.join(out_dir, f'patient_{safe}_gradcam.npy'), cam_up)
            n += 1
            print(f'  patient {uid}: p={prob:.3f}, scan {ts}, cam {tuple(cam[ts].shape)}')
        except Exception as exc:
            print(f'  patient #{i}: SKIPPED ({type(exc).__name__}: {exc})')
        if n >= N_POSITIVES:
            break
    print(f'##### GRAD-CAM DONE: {n} patients under {out_dir} #####')
    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main(parse_args())
