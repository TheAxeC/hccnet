"""Visit-prefix Layer-CAM on native MRI for all held-out HCC patients."""
import csv
from datetime import date
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from scipy.ndimage import map_coordinates
from monai.data import set_track_meta
from monai.utils import set_determinism
from monai.transforms import Compose, SpatialCropd, Spacingd, SpatialPadd, CenterSpatialCropd, Invertd, CropForeground, KeepLargestConnectedComponent
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.config import parse_args
from utils.transforms import transforms
from test import build_model
from saliency_localize import head, pct_box, layer_cam
from patient_figures import plane_geometry, sample, MODS, LABELS, page


def native_map(folder, modalities, cam):
    files={m:str(folder/f'{m}.nii.gz') for m in modalities}
    files['seg']=str(folder/'SEG.nii.gz')
    d=head(modalities)(files)
    ops=[]
    def apply(t):
        nonlocal d
        d=t(d);ops.append(t)
    b=pct_box(d['image'].shape[1:],(.5,.5,.5),(.85,.8,.99),(82,82,82))
    apply(SpatialCropd(keys=['image','seg'],roi_start=b[0],roi_end=b[1]))
    img=d['image'][0:1]
    mask=KeepLargestConnectedComponent(connectivity=1)((img>img.mean()).to(torch.uint8))
    fs,fe=CropForeground(select_fn=lambda x:x>0,margin=0,allow_smaller=False).compute_bounding_box(mask)
    apply(SpatialCropd(keys=['image','seg'],roi_start=list(fs),roi_end=list(fe)))
    apply(Spacingd(keys=['image','seg'],pixdim=(1.5,1.5,1.5),mode=(3,'nearest')))
    b=pct_box(d['image'].shape[1:],(.5,.3,.3),(.6,.5,.4),(82,82,82))
    apply(SpatialCropd(keys=['image','seg'],roi_start=b[0],roi_end=b[1]))
    apply(SpatialPadd(keys=['image','seg'],spatial_size=(72,72,72)))
    apply(CenterSpatialCropd(keys=['image','seg'],roi_size=(96,96,96)))
    apply(CenterSpatialCropd(keys=['image','seg'],roi_size=(72,72,72)))
    crop=d['image']
    c=F.interpolate(cam[None,None].float(),size=crop.shape[-3:],mode='trilinear',align_corners=False)[0,0].cpu()
    c=c/(c.max()+1e-8)
    meta=crop.clone();meta.copy_(c[None].repeat(crop.shape[0],1,1,1))
    inverse=dict(d);inverse['image']=meta
    restored=Invertd(keys='image',transform=Compose(ops),orig_keys='image')(inverse)['image']
    return np.asarray(restored[0]),np.asarray(restored.affine)


def heat_on_plane(cam,affine,world):
    vox=nib.affines.apply_affine(np.linalg.inv(affine),world)
    return map_coordinates(cam,vox.reshape(-1,3).T,order=1,mode='constant',cval=0).reshape(world.shape[:2])


def draw(ax,arr,heat=None):
    ax.imshow(arr,cmap='gray',origin='lower',vmin=0,vmax=1)
    if heat is not None:
        ax.imshow(np.ma.masked_less(heat,.1),cmap='inferno',origin='lower',alpha=.6,vmin=0,vmax=1)
    ax.set_xticks([]);ax.set_yticks([])
    ax.text(.015,.5,'R',color='white',transform=ax.transAxes)
    ax.text(.97,.5,'L',color='white',transform=ax.transAxes)
    for s in ax.spines.values():s.set_visible(False)


def visit_figure(uid,group,years,p,images,heat,out):
    mods=list(images)
    fig,axes=plt.subplots(2,len(mods),figsize=(3.6*len(mods),7.8),squeeze=False)
    for j,m in enumerate(mods):
        draw(axes[0,j],images[m]);draw(axes[1,j],images[m],heat)
        axes[0,j].set_title(LABELS[MODS.index(m)])
    axes[0,0].set_ylabel('MRI');axes[1,0].set_ylabel('MRI + Layer-CAM')
    timing = str(years) if isinstance(years, str) else f'{years:g} yr before HCC'
    fig.suptitle(f'{uid} | {timing} | {group} Pico fold 0 | prefix p={p:.3f}')
    fig.text(.5,.025,'Back at bottom | Joint-input attention on each displayed channel | Single-model saliency, not a phase-contribution estimate',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.045,1,.95));fig.savefig(out/f'{uid}_{group}_saliency.png',dpi=140);plt.close(fig)


def timeline(pid,group,history,out):
    mods=list(history[0]['images'])
    fig,axes=plt.subplots(len(mods),len(history),figsize=(3.3*len(history),3.7*len(mods)),squeeze=False)
    for j,v in enumerate(history):
        for i,m in enumerate(mods):
            draw(axes[i,j],v['images'][m],v['heat'])
            axes[i,j].set_xlabel(f"prefix p={v['probability']:.3f}",fontsize=9)
            if i==0:
                timing = v["scan_date"] if os.environ.get("HCCNET_DISPLAY_SCAN_DATES") else f"{v['years']:g} yr before HCC"
                axes[i,j].set_title(f"{v['visit']}\n{timing}",fontsize=10)
            if j==0:axes[i,j].set_ylabel(LABELS[MODS.index(m)])
    fig.suptitle(f'{pid} | {group} MRI + Layer-CAM through time | Pico fold 0',fontsize=13)
    fig.text(.5,.015,'Each column uses only visits available up to that visit | Back at bottom | Per-visit CAM scaling; compare location, not colour intensity',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(out/f'{pid}_{group}_longitudinal_saliency.png',dpi=140);plt.close(fig)


def main(args):
    set_determinism(seed=1234);set_track_meta(True)
    device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    root=Path(args.data_dir);out=Path(args.results_dir)/'patient_saliency';out.mkdir(parents=True,exist_ok=True)
    predroot=Path(os.environ['HCCNET_PREDICTIONS'])
    group=args.suffix.split('_pico')[0]
    with np.load(predroot/f'{args.suffix}.npz',allow_pickle=True) as z:
        cases=sorted({str(z['uids'][i]).rsplit('_',1)[0] for i in np.flatnonzero(z['labels'].ravel()==1)})
        reference={str(z['uids'][i]).rsplit('_',1)[0]:float(z['probs'][0,i]) for i in np.flatnonzero(z['labels'].ravel()==1)}
        requested = os.environ.get('HCCNET_REVIEW_PATIENTS')
        if requested:
            cases = sorted(requested.split(','))
            by_id = {str(u).rsplit('_',1)[0]:i for i,u in enumerate(z['uids'])}
            if not set(cases).issubset(by_id):
                raise ValueError('Review patients must belong to the saved held-out set')
            reference = {p:float(z['probs'][0,by_id[p]]) for p in cases}
    if not requested:
        assert len(cases)==8,cases
    diagnosis_dates = json.loads(Path(os.environ['HCCNET_DIAGNOSIS_DATES']).read_text()) if requested and os.environ.get('HCCNET_DIAGNOSIS_DATES') else {}
    with (root/'labels/labels.csv').open() as f: rows=list(csv.DictReader(f))
    scan_dates = {f"{r['id']}_{int(float(r['observation'])):03d}":r['date'] for r in rows}
    model=build_model(args,args.arch,1,device)
    model.load_state_dict(torch.load(Path(args.results_dir)/'model_weights'/f'weights_fold0_{args.suffix}.pth',map_location='cpu'));model.eval()
    transform=transforms('test',args.mod_list,'cpu')
    report=[]
    for pid in cases:
        visits=[];inputs=[]
        for row in sorted([r for r in rows if r['id']==pid],key=lambda r:(r['date'],float(r['observation']))):
            uid=f"{pid}_{int(float(row['observation'])):03d}";folder=root/'nifti'/uid
            if not all((folder/f'{m}.nii.gz').exists() for m in (args.required_modalities or args.mod_list)):
                report.append({'patient':pid,'visit':uid,'group':group,'unavailable':True});continue
            if float(row['delta'])<=0:
                raise ValueError(f'{uid}: nonpositive delta is treated as padding by the original model')
            image=transform({m:str(folder/f'{m}.nii.gz') for m in args.mod_list})['image']
            inputs.append(torch.as_tensor(image).float());visits.append((uid,float(row['delta'])))
        if not visits or len(visits)>8:raise ValueError(f'{pid}: unsupported visit count {len(visits)}')
        patient_out=out/pid;patient_out.mkdir(exist_ok=True)
        history=[]
        for ts,(uid,original_delta) in enumerate(visits):
            years = (date.fromisoformat(diagnosis_dates[pid]) - date.fromisoformat(scan_dates[uid])).days / 365.25 if pid in diagnosis_dates else original_delta
            # Exclude future acquisitions explicitly, including later visits in the same year bin.
            x=torch.stack(inputs[:ts+1])[None].to(device)
            pos=torch.tensor([[v[1] for v in visits[:ts+1]]],device=device)
            pad=torch.zeros_like(pos)
            cam,p=layer_cam(model,x,pad,pos)
            folder=root/'nifti'/uid
            native,affine=native_map(folder,args.mod_list,cam[ts])
            world,world_z,selection=plane_geometry(folder)
            heat=heat_on_plane(native,affine,world)
            # Enhanced DCE is the primary saliency display; precontrast remains in the raw atlas.
            display=[m for m in args.mod_list if not ((group=='t1dce' or group in ['phase_arterial','phase_venous','phase_late','phase_all']) and m=='T1WI')]
            entry={'patient':pid,'visit':uid,'group':group,'scan_date':scan_dates[uid],'original_model_delta':original_delta,'updated_outcome':pid in diagnosis_dates,'years':None if os.environ.get('HCCNET_DISPLAY_SCAN_DATES') else years,'probability':p,'n_prefix_visits':ts+1,'display_plane_mm':world_z,'selection':selection,'plane_cam_max':float(heat.max()),'model':'Pico fold 0','normalization':'per-visit 3D CAM maximum','full_history_saved_fold0_probability':reference[pid] if ts==len(visits)-1 else None}
            images={m:sample(folder/f'{m}.nii.gz',world) for m in display}
            visit_figure(uid,group,scan_dates[uid] if os.environ.get('HCCNET_DISPLAY_SCAN_DATES') else years,p,images,heat,patient_out)
            history.append(dict(entry,images=images,heat=heat))
            report.append(entry)
            print(uid,group,'prefix',ts+1,'p',p,'CAM max on plane',heat.max(),flush=True)
            del x,cam
        timeline(pid,group,history,patient_out)
        body=f'<h1>{pid}: {group} MRI + saliency</h1><p>Every overlay uses the representative Pico fold-0 model, with only visits available up to the displayed visit. Its probabilities differ from the 50-model ensemble graph. The same joint-input attention map is displayed on each channel; this does not isolate a channel contribution. The maps have separate per-visit scales. SEG is a liver/organ mask, not a tumor annotation.</p><img src="{pid}_{group}_longitudinal_saliency.png">'
        for v in history:body+=f'<h2>{v["visit"]}</h2><img src="{v["visit"]}_{group}_saliency.png">'
        (patient_out/f'{group}.html').write_text(page(body))
    (out/f'{group}_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    assert len({r['patient'] for r in report if not r.get('unavailable')})==len(cases)


if __name__=='__main__':main(parse_args())
