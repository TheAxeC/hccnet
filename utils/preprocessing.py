import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import numpy as np
import argparse
import os
import json
from typing import List, Tuple
from models.convnext3d import (
    convnext3d_atto,
    convnext3d_femto,
    convnext3d_compact,
    convnext3d_pico,
    convnext3d_nano,
    convnext3d_tiny
)
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    partition_dataset_classes,
    list_data_collate
)
from monai.utils.misc import ensure_tuple_rep
from sklearn.model_selection import StratifiedGroupKFold
from data.splits import GroupStratifiedSplit
from data.datasets import CacheSeqDataset
from data.utils import (
    DatasetPreprocessor, 
    convert_to_dict, 
    convert_to_seqdict, 
    SequenceBatchCollater
)
from utils.transforms import transforms, dino_transforms
from utils.utils import cosine_scheduler, get_params_groups
from losses.focalloss import FocalLoss
from losses.binaryceloss import BinaryCELoss
from losses.dinoloss import DINOLoss

def load_backbone(
        args: argparse.Namespace,
        arch: str,
        dino_pretraining: bool = False
    ) -> nn.Module | Tuple[nn.Module]:

    '''
    Args:
        args (argparse.Namespace): Command line arguments.
        dino_pretraining (bool): Boolean flag to indicate pretraining of the CNN backbone
    '''

    in_chans = 1 if dino_pretraining else len(args.mod_list)

    if arch == 'atto':
        student = convnext3d_atto(
            in_chans=in_chans, stem_mode=getattr(args, "stem_mode", "agnostic"), kernel_size=args.kernel_size, drop_path_rate=args.stochastic_depth, use_v2=args.use_v2, eps=args.epsilon)
        teacher = convnext3d_atto(
            in_chans=in_chans, stem_mode=getattr(args, "stem_mode", "agnostic"), kernel_size=args.kernel_size, use_v2=args.use_v2, eps=args.epsilon)
    elif arch == 'femto':
        student = convnext3d_femto(
            in_chans=in_chans, stem_mode=getattr(args, "stem_mode", "agnostic"), kernel_size=args.kernel_size, drop_path_rate=args.stochastic_depth, use_v2=args.use_v2, eps=args.epsilon)
        teacher = convnext3d_femto(
            in_chans=in_chans, stem_mode=getattr(args, "stem_mode", "agnostic"), kernel_size=args.kernel_size, use_v2=args.use_v2, eps=args.epsilon)
    elif arch == 'compact':
        student = convnext3d_compact(
            in_chans=in_chans, stem_mode=getattr(args, "stem_mode", "agnostic"), kernel_size=args.kernel_size, drop_path_rate=args.stochastic_depth, use_v2=args.use_v2, eps=args.epsilon)
        teacher = convnext3d_compact(
            in_chans=in_chans, stem_mode=getattr(args, "stem_mode", "agnostic"), kernel_size=args.kernel_size, use_v2=args.use_v2, eps=args.epsilon)
    elif arch == 'pico':
        student = convnext3d_pico(
            in_chans=in_chans, stem_mode=getattr(args, "stem_mode", "agnostic"), kernel_size=args.kernel_size, drop_path_rate=args.stochastic_depth, use_v2=args.use_v2, eps=args.epsilon)
        teacher = convnext3d_pico(
            in_chans=in_chans, stem_mode=getattr(args, "stem_mode", "agnostic"), kernel_size=args.kernel_size, use_v2=args.use_v2, eps=args.epsilon)
    elif arch == 'nano':
        student = convnext3d_nano(
            in_chans=in_chans, stem_mode=getattr(args, "stem_mode", "agnostic"), kernel_size=args.kernel_size, drop_path_rate=args.stochastic_depth, use_v2=args.use_v2, eps=args.epsilon)
        teacher = convnext3d_nano(
            in_chans=in_chans, stem_mode=getattr(args, "stem_mode", "agnostic"), kernel_size=args.kernel_size, use_v2=args.use_v2, eps=args.epsilon)
    elif arch == 'tiny':
        student = convnext3d_tiny(
            in_chans=in_chans, stem_mode=getattr(args, "stem_mode", "agnostic"), kernel_size=args.kernel_size, drop_path_rate=args.stochastic_depth, use_v2=args.use_v2, eps=args.epsilon)
        teacher = convnext3d_tiny(
            in_chans=in_chans, stem_mode=getattr(args, "stem_mode", "agnostic"), kernel_size=args.kernel_size, use_v2=args.use_v2, eps=args.epsilon)
    
    if dino_pretraining:
        return student, teacher
    else:
        return student

def load_data(
        args: argparse.Namespace,
        device: torch.device,
        phase: str = 'train',
        partial: bool = False,
        max_delta: int = 3,
        month_horizon: int | str = 0
    ) -> tuple:

    '''
    Args:
        args (argparse.Namespace): Command line arguments.
        device (torch.device): Pytorch device.
        phase (str): Current training phase. Can be 'train', 'pretrain', or 'test'.
        partial (bool): Boolean flag to indicate whether loaded images need to include all specified modalities. Defaults to false.
    '''
    folds = range(args.k_folds) if args.k_folds > 0 else range(1)
    if phase == 'test':
        phases = ['test']
        # labels_file = f'labels_{max_delta}months.csv'
    elif phase == 'pretrain':
        phases = ['train']
    else:
        phases = ['train','val']
        
    preprocessor = DatasetPreprocessor(
        data_dir=args.data_dir, 
        partial=True if partial else False)
    # Prediction-horizon evaluation (Sec. results by prediction horizon): at test, horizons > 1
    # month load labels_{N}months.csv (same test patients, but observations subset to those >= N
    # months before diagnosis and relabelled). The 1-month horizon uses the full labels.csv (all
    # observations, per the paper's horizon table note); only labels_{3,6,12,24,36}months.csv exist.
    # month_horizon in {0, 1} therefore keeps the default labels.csv, so the standard test is unaffected.
    if phase == 'test' and str(month_horizon) not in ('0', '1'):
        labels_file = f'labels_{month_horizon}months.csv'
    else:
        labels_file = 'lirads.csv' if phase == 'pretrain' else 'labels.csv'
    data_dict, label_df = preprocessor.load_data(
        modalities=args.mod_list,
        keys=['label','delta'],
        file_name=labels_file,
        verbose=False)
    required = getattr(args, 'required_modalities', None)
    if required:
        # Cohort eligibility is independent of the channels supplied to this arm.
        eligible = {item['uid'] for item in data_dict if all(
            os.path.isfile(os.path.join(args.data_dir, 'nifti', item['uid'], mod + '.nii.gz'))
            for mod in required)}
        data_dict = [item for item in data_dict if item['uid'] in eligible]
        label_df = label_df[label_df['uid'].isin(eligible)].copy()
        if label_df.empty:
            raise ValueError('No observations meet --required-modalities')
    # We always load diffusion weighted MRIs to ensure that the data in the test is consistent across training and pretraining.
    default_dict, default_df = DatasetPreprocessor(data_dir=args.data_dir).load_data(
        modalities=['DWI_b0','DWI_b150','DWI_b400','DWI_b800'], 
        keys=['label','delta'], 
        file_name='labels.csv')
    default_dev, default_test = GroupStratifiedSplit(split_ratio=0.75).split_dataset(default_df)
    test = label_df[label_df['patient_id'].isin(default_test['patient_id'])]
    dev = label_df[-label_df['patient_id'].isin(test['patient_id'])]
    if phase == 'train':
        if args.k_folds > 1:
            cv_folds = StratifiedGroupKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed).split(dev, y=dev['label'], groups=dev['patient_id'])
            indices = [(dev.iloc[train_idx], dev.iloc[val_idx]) for train_idx, val_idx in list(cv_folds)]
            split_dict = [convert_to_dict([indices[k][0], indices[k][1]], data_dict=data_dict, split_names=phases) for k in folds]
        elif args.k_folds == 1:
            train, val = GroupStratifiedSplit(split_ratio=0.8).split_dataset(dev)
            split_dict = [convert_to_dict([train, val], data_dict=data_dict, split_names=phases, verbose=True) for k in folds]
        else:
            split_dict = [convert_to_dict([dev, test], data_dict=data_dict, split_names=phases, verbose=True) for k in folds]
    elif phase == 'pretrain':
        folds = range(1)
        split_dict = [convert_to_dict([dev], data_dict=data_dict, split_names=phases, verbose=True) for k in folds]
    elif phase == 'test':
        folds = range(1)
        # --score-all-patients widens the test phase to the whole cohort. The development
        # patients trained the model, so what comes back describes the cohort and does not
        # validate on unseen data.
        scored = label_df if getattr(args, 'score_all_patients', False) else test
        split_dict = [convert_to_dict([scored], data_dict=data_dict, split_names=phases, verbose=True) for k in folds]
    seq_split_dict = [convert_to_seqdict(split_dict[k], args.mod_list, phases) for k in folds]
    if required and (not dist.is_initialized() or dist.get_rank() == 0):
        audit_dir = os.path.join(args.results_dir, 'cohort_audit')
        os.makedirs(audit_dir, exist_ok=True)
        audit = [{name: sorted(item['uid'] for item in fold[name]) for name in phases}
                 for fold in split_dict]
        with open(os.path.join(audit_dir, phase + '.json'), 'w') as handle:
            json.dump(audit, handle, indent=2)

    class_dict = {x: [[patient['label'] for patient in split_dict[k][x]] for k in folds] for x in phases}
    seq_class_dict = {x: [[max(patient['label']) for patient in seq_split_dict[k][x][0]] for k in folds] for x in phases}
    data_partition_dict = {x: [partition_dataset_classes(
        data=split_dict[k][x] if partial else seq_split_dict[k][x][0],
        classes=class_dict[x][k] if partial else seq_class_dict[x][k],
        num_partitions=dist.get_world_size(),
        shuffle=(True if x != 'test' else False),
        even_divisible=False
        )[dist.get_rank()] for k in folds] for x in phases}
    if partial:
        # In-memory cache for the DINO (single-image) path. NOTE: this rebuilds the
        # ~26-min deterministic-preprocessing cache every run (it is not persisted to disk
        # like the sequence path's CacheSeqDataset). A PersistentDataset disk cache would
        # avoid that, but the transform uses LoadImaged(image_only=True) so the channel +
        # affine metadata that EnsureChannelFirstd/Orientationd/Spacingd need is lost across
        # disk serialization - a real but non-trivial follow-up optimization.
        # Cache the DINO views on CPU RAM (250G), NOT the GPU. CacheDataset caches the whole
        # dataset (cache_rate=1); with device=<cuda> that piles every cached item into GPU
        # memory, and the higher-channel fusion configs (8-12 ch) overflow the A40's 46G at
        # the CACHE-BUILD step (~43G, before training) -> CUDA OOM. The training step already
        # moves each batch to the GPU (encoder_step: view.to(self.gpu_id)), so a CPU cache is
        # numerically identical (MONAI randomises via numpy RNG, device-independent) and only
        # adds a per-batch host->device copy. Matches the sequence path, which already uses cpu.
        datasets = {x: [CacheDataset(
            data=data_partition_dict[x][k],
            transform=dino_transforms(
                modalities=args.mod_list,
                device='cpu',
                global_crop_size=ensure_tuple_rep(args.global_crop_size, 3),
                local_crop_size=ensure_tuple_rep(args.local_crop_size, 3)),
            num_workers=8,
            copy_cache=False
            ) for k in folds] for x in phases}
    else:
        datasets = {x: [CacheSeqDataset(
            data=data_partition_dict[x][k],
            image_keys=args.mod_list,
            transform=transforms(
                dataset=x,
                modalities=args.mod_list,
                device='cpu',
                crop_size=ensure_tuple_rep(args.global_crop_size, 3)),
            num_workers=8,
            copy_cache=False,
            cache_dir=getattr(args, 'cache_dir', None),
            cache_device='cpu'
            ) for k in folds] for x in phases}
    dataloader = {x: [ThreadDataLoader(
        datasets[x][k], 
        batch_size=(args.batch_size if x == 'train' else 1), 
        shuffle=(True if x == 'train' else False),   
        drop_last=(True if x == 'train' else False),   
        num_workers=0,
        collate_fn=(SequenceBatchCollater(
            keys=['image','label','delta'], 
            seq_length=args.seq_length) if (x == 'train') & (not partial) else list_data_collate)
        ) for k in folds] for x in phases}
    _, counts = np.unique(seq_class_dict['test' if phase == 'test' else 'train'][0], return_counts=True)
    pos_weight = counts[1] / counts.sum()

    return dataloader, pos_weight

def load_objs(
        args: argparse.Namespace,
        model: nn.Module,
        learning_rate: float = 1e-4,
        pos_weight: float | List[float] | None = None
    ) -> tuple:

    '''
    Args:
        args (argparse.Namespace): Command line arguments.
        model (nn.Module): Pytorch module object.
        learning_rate (float): Base learning rate.
        pos_weight (float | List[float] | None): Class weight of the positive class, class weights of both classes, or None. Defaults to None.
    '''

    if args.loss_fn == 'bce':
        train_fn = BinaryCELoss(weights=pos_weight, label_smoothing=args.label_smoothing)
        loss_fn = [train_fn, BinaryCELoss(weights=pos_weight)]
    elif args.loss_fn == 'focal':
        train_fn = FocalLoss(gamma=args.gamma, alpha=pos_weight)
        loss_fn = [train_fn, train_fn]
    elif args.loss_fn == 'mse':
        loss_fn = nn.MSELoss()
    elif args.loss_fn == 'dino':
        loss_fn = DINOLoss(
            out_dim=args.out_dim,
            num_crops=4,
            num_steps=args.num_steps,
            teacher_temp=args.teacher_temp,
            teacher_warmup_temp=args.teacher_warmup_temp,
            teacher_warmup_steps=int(args.warmup_steps * 2))
    params = get_params_groups(model)
    optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=args.weight_decay)
    lr_schedule = cosine_scheduler(
        base_value=learning_rate,
        final_value=args.min_learning_rate,
        steps=args.num_steps,
        warmup_steps=args.warmup_steps)
    wd_schedule = cosine_scheduler(
        base_value=args.weight_decay,
        final_value=args.max_weight_decay,
        steps=args.num_steps)
    m_schedule = cosine_scheduler(
        base_value=args.teacher_momentum,
        final_value=1,
        steps=args.num_steps)
    schedules = [lr_schedule, wd_schedule, m_schedule]
    return loss_fn, optimizer, schedules
