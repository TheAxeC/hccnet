'''
Original configuration used in the HCCNet paper (Nolte et al., 2025).

These values were previously hard-coded across pretrain.py / train.py / test.py.
They are collected here so that (a) the main code is parameterised rather than
littered with magic constants, and (b) the exact paper setup can still be
reproduced. The functions below define the artifact naming convention used to
hand a checkpoint from one training stage to the next.
'''

import os

# --- Pretraining checkpoint schedule (paper) -------------------------------
# pretrain.py historically saved a checkpoint only when the update step reached
# one of these milestones. Kept for backwards-compatible reproduction; the
# hand-off checkpoint (see pretrain_weight_path) is now always saved at the end
# of pretraining regardless of step count.
PRETRAIN_CHECKPOINT_STEPS = [2000, 4000, 8000, 16000, 32000]

# Training lengths the paper used for each stage (update steps).
DINO_PRETRAIN_STEPS = 32000     # DINO backbone (self-supervised) pretraining
DECODER_PRETRAIN_STEPS = 8000   # Transformer/decoder (shuffle) pretraining
FINETUNE_STEPS = 400            # Downstream supervised fine-tuning

# --- Paper modality groups -------------------------------------------------
# The three input-channel sets evaluated in the paper. The key is the short tag
# used in run suffixes; the value is the --mod-list. transforms.py auto-selects the
# matching intensity normalisation from the modalities present, so switching group
# only requires changing --mod-list (and the suffix/run-tag).
PAPER_MODALITIES = {
    'dwi':        ['DWI_b0', 'DWI_b150', 'DWI_b400', 'DWI_b800'],   # DW-MRI (4 b-values)
    't1dce':      ['T1WI', 'T1A', 'T1V', 'T1D'],                    # T1 dynamic contrast-enhanced
    't1iop':      ['T1W_IP', 'T1W_OOP', 'T2W_TES', 'T2W_TEL'],      # T1 in/out-of-phase + T2 (combined)
    # Constituent splits of the combined t1iop group (reported per arch). At
    # fine-tuning these REUSE the 't1iop' DINO+decoder backbone (pass
    # --run-tag t1iop_<arch>...); the channel-agnostic stem makes that exact.
    't1iop_only': ['T1W_IP', 'T1W_OOP'],                           # T1 IOP-MRI (2 ch)
    't2_only':    ['T2W_TES', 'T2W_TEL'],                          # T2-MRI (2 ch)
}
# Fusion presets (HCCNet-Pico only): concatenated modality groups, each pre-trained
# with its OWN DINO (within-contrast augmentation retained; reported in the fusion table).
PAPER_FUSION = {
    'fuse_dwi_t1dce': ['DWI_b0', 'DWI_b150', 'DWI_b400', 'DWI_b800', 'T1WI', 'T1A', 'T1V', 'T1D'],                # 8 ch
    'fuse_dwi_t1iop': ['DWI_b0', 'DWI_b150', 'DWI_b400', 'DWI_b800', 'T1W_IP', 'T1W_OOP', 'T2W_TES', 'T2W_TEL'],  # 8 ch
    'fuse_full':      ['DWI_b0', 'DWI_b150', 'DWI_b400', 'DWI_b800', 'T1WI', 'T1A', 'T1V', 'T1D',
                       'T1W_IP', 'T1W_OOP', 'T2W_TES', 'T2W_TEL'],                                                # 12 ch
}
PAPER_ARCHS = ['femto', 'compact', 'pico', 'nano', 'tiny']


# --- Paper evaluation sweep (test.py --paper-sweep) ------------------------
PAPER_TEST_ARCHS = ['femto', 'compact', 'pico', 'nano', 'tiny']
PAPER_TEST_SUFFIXES = ['pre_400steps', '400steps']  # pretrained-init vs random-init
HCC_TEST_PREVALENCE = 0.138     # positive base rate; PR-curve chance baseline


# --- Full paper hyper-parameters per stage (arXiv:2501.10733) --------------
# Verified against the paper's implementation-details section. Only values that
# DIFFER from the argparse defaults are listed; values that already match the
# defaults (teacher_temp=0.04, teacher_warmup_temp=0.04, teacher_momentum=0.9995,
# global/local crop 72/48) are intentionally omitted. Applied by
# apply_paper_config() when --paper is passed.
#
# Note on effective batch size: the paper's per-GPU batch differs from the
# effective batch (it accumulates/uses multiple GPUs). The code derives the
# learning rate from --effective-batch-size, so only that needs to be exact;
# --batch-size stays a per-GPU memory knob (accumulation adapts automatically).
PAPER_STAGES = {
    # DINO CNN backbone pretraining: 32000 steps, effective batch 128, 1600 warm-up.
    'dino':     {'num_steps': 32000, 'effective_batch_size': 128, 'seq_length': 8},
    # Transformer/decoder pretraining: 8000 steps, effective batch 32, dropout 0.2.
    'decoder':  {'num_steps': 8000,  'effective_batch_size': 32,  'seq_length': 8,
                 'dropout': 0.2},
    # Fine-tuning: 400 steps, 5-fold CV x 10 seeds, constant weight decay 1e-5.
    'finetune': {'num_steps': 400,   'effective_batch_size': 32,  'seq_length': 8,
                 'dropout': 0.2, 'k_folds': 5, 'num_seeds': 10,
                 'weight_decay': 1e-5, 'max_weight_decay': 1e-5},
}


def _paper_warmup_steps(stage: str, num_steps: int, arch: str) -> int:
    '''Warm-up steps used in the paper for a given stage/architecture.

    DINO used a fixed 1600 warm-up steps. Decoder and fine-tuning used 5% of the
    schedule for the smaller variants (femto/pico) and 10% for the bigger ones
    (nano/tiny).
    '''
    if stage == 'dino':
        return 1600
    pct = 0.10 if arch in ('nano', 'tiny') else 0.05
    return max(1, round(num_steps * pct))


def apply_paper_config(args, stage: str):
    '''Overwrite the hyper-parameters covered by the paper preset for `stage`.

    stage in {'dino', 'decoder', 'finetune'}. Sets num_steps, warm-up, effective
    batch size, sequence length, dropout, folds/seeds and weight decay to the paper
    values; leaves arch / suffix / paths / distributed / amp untouched. Verified
    against arXiv:2501.10733.
    '''
    if stage not in PAPER_STAGES:
        raise ValueError(f"unknown paper stage {stage!r}; expected one of {list(PAPER_STAGES)}")
    for key, value in PAPER_STAGES[stage].items():
        setattr(args, key, value)
    args.warmup_steps = _paper_warmup_steps(stage, args.num_steps, args.arch)
    return args


# --- Artifact naming -------------------------------------------------------
def resolve_run_tag(args, modality: str) -> str:
    '''Identifier that links a pretraining run to the stages that load it.

    All three stages (DINO -> decoder -> fine-tune) of one pipeline must share the
    same run_tag so they read/write a consistent set of hand-off checkpoints.

    Defaults to '<modality>_<arch>', which reproduces the historical naming. Pass a
    distinct --run-tag to keep two configs that share the same modality+arch (e.g.
    different pretraining hyper-parameters) from overwriting each other.
    '''
    tag = getattr(args, 'run_tag', None)
    return tag if tag else f'{modality}_{args.arch}'


def pretrain_weight_name(stage: str, run_tag: str) -> str:
    '''Filename of the hand-off checkpoint produced by a pretraining stage.

    Args:
        stage: 'dino' (CNN backbone) or 'decoder' (Transformer encoder).
        run_tag: identifier shared across the stages of one pipeline (see resolve_run_tag).
    '''
    if stage not in ('dino', 'decoder'):
        raise ValueError(f"stage must be 'dino' or 'decoder', got {stage!r}")
    return f'weights_{stage}pretrain_{run_tag}.pth'


def pretrain_weight_path(results_dir: str, stage: str, run_tag: str) -> str:
    '''Full path of a pretraining hand-off checkpoint under results_dir.'''
    return os.path.join(results_dir, 'model_weights', pretrain_weight_name(stage, run_tag))
