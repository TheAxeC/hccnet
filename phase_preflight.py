"""Check phase eligibility, raw-image transforms, and model shapes on a compute node."""
import json
import os
from pathlib import Path
import torch
from monai.utils import set_determinism
from data.utils import DatasetPreprocessor
from data.splits import GroupStratifiedSplit
from utils.config import parse_args
from utils.transforms import transforms
from test import build_model


def main(args):
    set_determinism(seed=1234)
    required=['T1WI','T1A','T1V','T1D']
    root=Path(args.data_dir)
    records,df=DatasetPreprocessor(args.data_dir).load_data(required,['label','delta'],'labels.csv',False)
    _,dwi_df=DatasetPreprocessor(args.data_dir).load_data(['DWI_b0','DWI_b150','DWI_b400','DWI_b800'],['label','delta'],'labels.csv',False)
    _,test=GroupStratifiedSplit(split_ratio=.75).split_dataset(dwi_df)
    held=df[df.patient_id.isin(test.patient_id)]
    cases=held.groupby('patient_id').label.max()
    assert int(cases.sum())==8, cases
    sample=next(r for r in records if r['label']==1)
    report={'eligible_visits':len(df),'held_out_patients':len(cases),'hcc':int(cases.sum()),'arms':{}}
    for name,mods in [('base',['T1WI']),('arterial',['T1WI','T1A']),('venous',['T1WI','T1V']),('late',['T1WI','T1D']),('all',required)]:
        other,odf=DatasetPreprocessor(args.data_dir).load_data(mods,['label','delta'],'labels.csv',False)
        eligible={r['uid'] for r in other if all((root/'nifti'/r['uid']/(m+'.nii.gz')).is_file() for m in required)}
        assert set(odf[odf.uid.isin(eligible)].uid)==set(df.uid), name
        values={m:sample[m] for m in mods}
        image=transforms('test',mods,'cpu')(values)['image']
        args.mod_list=mods
        model=build_model(args,'pico',1,'cuda').eval()
        x=torch.as_tensor(image).float().cuda()[None,None]
        with torch.no_grad():
            y=model(x,pad_mask=torch.zeros(1,1,device='cuda'),pos=torch.ones(1,1,device='cuda'))
        assert torch.isfinite(y).all(), name
        report['arms'][name]={'channels':len(mods),'image_shape':list(image.shape),'finite_forward':True}
        del model,x;torch.cuda.empty_cache()
    Path(args.results_dir).mkdir(parents=True,exist_ok=True)
    (Path(args.results_dir)/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main(parse_args())
