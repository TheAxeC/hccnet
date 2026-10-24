# Getting Started with HCCNet (UTwente HPC)

A practical guide to running this repo on the UTwente SLURM cluster. HCCNet predicts future hepatocellular carcinoma (HCC) from a patient's *sequence* of past liver MRIs, using a 3D ConvNeXt CNN backbone + a Transformer encoder over time.

> This repo has been adapted from the original paper code. Key changes: sensible default paths (no more hard-coded personal paths), an on-disk preprocessing cache, de-hard-coded pretraining hand-off filenames, a torch-1.13 fix in the DINO head, and a rewritten `test.py`. The paper's exact hyper-parameters live in `utils/paper_config.py`. See section 7 for the full change list.

---

## 1. What's where

| Thing | Location |
|-------|----------|
| Code | `~/projects/hccnet` |
| Dataset | `/deepstore/datasets/bms/hcc_study/` (read-only, group `bms-code-hcc_study`) |
| Results | `~/projects/hccnet/results` (default, auto-created) |
| Preprocessing cache | `~/projects/hccnet/cache` (default, auto-created) |

Dataset layout:
```
hcc_study/
  nifti/         # 925 observation dirs: PT_<patient>_<obs>/<MODALITY>.nii.gz
  images/        # single-modality copies (NOT used by training)
  labels/
    labels.csv   # id, observation, date, delta, label, lirads  <- fine-tuning labels
    lirads.csv   # used for pretraining
    labels_Nmonths.csv
```
- A patient (`PT_002`) has several timepoints (`PT_002_001`, `PT_002_002`, ...).
- Each observation dir holds one `.nii.gz` per MRI modality (DWI_b0/150/400/800, T1WI, T1A/D/V, T1W_IP/OOP, T2W_TES/TEL, SEG).
- `label` = future HCC (0/1); `delta` = years before diagnosis (0 = padding/none).
- There is **no etiology / clinical metadata** in this dataset - only the columns above.

---

## 2. One-time environment setup

The cluster has no conda on the PATH by default and the head node has no GPU.

```bash
module load anaconda3/2024.02
conda create -y -p $HOME/envs/hccnet python=3.10
source activate $HOME/envs/hccnet

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 \
    --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r ~/projects/hccnet/requirements.txt
```

> The original `requirements.txt` (Python 3.7 / monai 1.1.0) was **stale**. The updated file pins monai 1.3.0 and adds packages that were missing entirely (`timm`, `scikit-image`, `nibabel`, `setuptools`). Already done on this account.

---

## 3. Two things to know before running

**A. This code REQUIRES torchrun + `--distributed`.** `train.py` / `pretrain.py` / `test.py` call `dist.get_rank()` / `dist.get_world_size()` unconditionally, even on one GPU. Launch like:
```bash
torchrun --standalone --nproc_per_node=1 train.py --distributed ...
```

**B. Defaults now "just work" - you rarely pass path flags.** `utils/config.py` defaults:

| Flag | Default | Override env var |
|------|---------|------------------|
| `--data-dir` | `/deepstore/datasets/bms/hcc_study` | `HCCNET_DATA_DIR` |
| `--results-dir` | `<repo>/results` | `HCCNET_RESULTS_DIR` |
| `--weights-dir` | `<repo>/results/model_weights` | `HCCNET_WEIGHTS_DIR` |
| `--cache-dir` | `<repo>/cache` (persistent caching ON) | `HCCNET_CACHE_DIR` |
| `--amp` | **on** (use `--no-amp` to disable) | - |
| `--mod-list` | the 4 DWI b-values | - |

So a real run only needs the experiment knobs: `--arch`, `--k-folds`, `--num-seeds`, `--num-steps`, `--suffix`, batch sizes, `--distributed`. Pass `--cache-dir ''` to turn the disk cache off (falls back to in-RAM only).

### SLURM notes
- Account `bms-code` is **not** allowed on `debug`. Use **`main-gpu`** (allows all). Its GPUs: turing, **ampere (A40, 46 GB)**, **lovelace (L40, 48 GB, fastest here)**. Check: `sinfo -p main-gpu -N -o "%N %G %t"`.
- `--effective-batch-size` MUST be one of {8,16,32,64,128,256,512} - it indexes a hard-coded LR table in `utils/utils.py:scale_learning_rate`.

---

## 4. Quick smoke test

A ~90-patient symlinked subset lives at `~/projects/hccnet/hcc_subset` (`nifti/` = symlinks, `labels/` = the real CSVs) for fast end-to-end checks. Example (`~/projects/hccnet/run_smoke.sbatch`):
```bash
torchrun --standalone --nproc_per_node=1 train.py \
  --distributed --arch femto \
  --data-dir $HOME/projects/hccnet/hcc_subset \
  --k-folds 1 --num-seeds 1 \
  --num-steps 4 --warmup-steps 1 --log-every 2 \
  --batch-size 2 --effective-batch-size 8 \
  --suffix DWI_smoke
```
Expected: caches data, prints `Step 0/4 ... Script finished`, writes weights/history/ learning-curve PNGs under `results/`. Metrics at 4 steps are meaningless - it only proves the pipeline runs.

---

## 5. The full pipeline (staged)

Three training stages + evaluation. Ready-to-submit scripts are in `~/projects/hccnet`: `stage1_dino.sbatch`, `stage2_decoder.sbatch`, `stage3_finetune.sbatch`, `stage4_test.sbatch`. Run them in order (each checks its predecessor's output exists).

```
   Stage 1 - DINO backbone pretrain        pretrain.py --loss-fn dino
     self-supervised, single images,        -> results/model_weights/
     multi-crop teacher/student                weights_dinopretrain_<modality>_<arch>.pth
             v
   Stage 2 - decoder/Transformer pretrain  pretrain.py --loss-fn bce
     shuffle-detection task, CNN frozen,     loads the DINO hand-off; ->
     loads the DINO hand-off                   weights_decoderpretrain_<modality>_<arch>.pth
             v
   Stage 3 - supervised fine-tuning        train.py --pretrained
     k-folds x num-seeds runs,               loads the decoder hand-off; ->
     loads the decoder hand-off                weights_fold{0..N-1}_<suffix>.pth  (N = folds*seeds)
             v
   Stage 4 - evaluation                     test.py
     ensembles all fold checkpoints on       -> results/model_diagnostics/
     the held-out test set                     test_metrics_*.txt + roc_pr_curves/*.png
```

**Hand-off naming (important).** Stages are chained by descriptive filenames, NOT by the old hard-coded `fold32000`/`fold8000` sentinels. The hand-off file is keyed by `(stage, modality, arch)` where `modality` = the first token of `--suffix` and `arch` = `--arch`. Because of this, **pretraining length is now a free parameter** (any `--num-steps`); the final checkpoint is always saved. (The paper's milestone checkpoints at 2000/4000/8000/16000/32000 steps are still written too - see `utils/paper_config.py:PRETRAIN_CHECKPOINT_STEPS`.)

### Running several pretraining configs (parallel or sequential)
- **Each pipeline has a `--run-tag`** that keys its pretraining hand-off chain (DINO -> decoder -> fine-tune must all share it). It defaults to `<modality>_<arch>` (e.g. `dwi_femto`) - the historical naming.
- **To run multiple configs that share modality+arch** (e.g. different pretraining hyper-parameters), pass a distinct `--run-tag` on all three stages. Their hand-off checkpoints are `weights_*pretrain_<run-tag>.pth`, so they never collide - even in parallel. Example: add `--run-tag dwi_femto_lr2x` to stages 1-3 of the second config.
- Different `--arch` **or** different `--run-tag` -> distinct hand-off files -> safe.
- **One pretraining -> many fine-tunes is safe** (fine-tune only reads the hand-off).
- The **disk cache is shared-safe** (content-keyed, atomic writes) and reused across runs on the same data/transforms - caching is paid once, not per job.
- Also give each run a distinct `--suffix`: the suffix names a run's own outputs (`weights_fold*/hist_/preds_`), while `--run-tag` links the pretraining chain.

### Reproduce the exact paper run (`--paper`)
Pass `--paper` and every hyper-parameter is set to the published values (arXiv:2501.10733) - no need to know them by hand. Each stage pulls its block from `utils/paper_config.py:PAPER_STAGES`:

| Stage | command | `--paper` sets |
|-------|---------|----------------|
| DINO | `pretrain.py --paper --loss-fn dino` | 32000 steps, eff-batch 128, 1600 warm-up, seq-len 8 |
| Decoder | `pretrain.py --paper --loss-fn bce` | 8000 steps, eff-batch 32, 5%/10% warm-up, dropout 0.2 |
| Fine-tune | `train.py --paper` | 400 steps, 5 folds x 10 seeds, eff-batch 32, dropout 0.2, weight decay 1e-5 |

`--paper` is **modality-agnostic** - it only sets training hyper-parameters, which are identical across the paper's three input groups. The modality is chosen by `--mod-list` (and `transforms.py` auto-selects the matching intensity normalisation):

| Group | `--mod-list` |
|-------|--------------|
| `dwi` | `DWI_b0 DWI_b150 DWI_b400 DWI_b800` |
| `t1dce` | `T1WI T1A T1V T1D` |
| `t1iop` | `T1W_IP T1W_OOP T2W_TES T2W_TEL` |

(See `utils/paper_config.py:PAPER_MODALITIES` / `PAPER_ARCHS`.)

**Ready-made scripts** take `[GROUP] [ARCH]` arguments (default `dwi femto`), so any of the paper's 3 groups x 4 architectures runs turnkey:
```bash
sbatch hpc/stage1_dino_paper.slurm      t1dce pico   # then 2, 3, 4 with the SAME args
sbatch hpc/stage2_decoder_paper.slurm   t1dce pico
sbatch hpc/stage3_finetune_paper.slurm  t1dce pico
sbatch hpc/stage4_test_paper.slurm      t1dce pico
```
Each config gets its own `--run-tag <group>_<arch>_paper`, so different modalities/archs (and the earlier light run) never overwrite each other's hand-offs or checkpoints. `--paper` overrides only the hyper-parameters; `--batch-size` (a per-GPU memory knob - accumulation adapts) is left to you.

> **Time / GPUs.** DINO at effective batch 128 is **~27 h on one GPU** (accum 32). Run it on **4 GPUs** (`--gres=gpu:4`, `--nproc_per_node=4`) for **~7 h** - see the header comment in `hpc/stage1_dino_paper.slurm` (verify a short multi-GPU run first; multi-GPU hasn't been exercised here yet). Decoder ~3 h, fine-tune ~4 h on one GPU.

### Evaluation (`test.py`)
- Default: evaluates the run named by `--arch` + `--suffix`. It globs every `weights_fold*_<suffix>.pth`, runs each on the test set, and reports **per-model AUROC/AUPRC (mean $\pm$ std)** plus the **probability-ensemble** of all of them, and writes ROC + PR curves and a `test_metrics_*.txt`.
- `--paper-sweep`: reproduce the paper's table across all architectures and the pretrained/random-init suffixes (from `utils/paper_config.py`), loading from `--weights-dir`.

### How one forward pass works (`models/mednet.py:MedNet`)
1. Each MRI observation = a multi-channel 3D volume (the 4 DWI b-values = 4 channels).
2. The **3D ConvNeXt backbone** (femto/pico/nano/tiny) embeds each volume into a vector.
3. A patient = a **sequence** of those vectors; `delta` (years-before-diagnosis) is added as a time-based positional encoding; padding timepoints (delta=0) are masked.
4. The **Transformer encoder** processes the sequence; pooling + linear head outputs the HCC probability. Loss = weighted BCE (or focal). Metrics = AUPRC + AUROC.

### Data flow (`utils/preprocessing.py` + `data/`)
- `DatasetPreprocessor` keeps observations containing **all** requested `--mod-list` modalities, joined to the label CSV.
- Patient-grouped, label-stratified splits (`data/splits.py`) -> train/val/test with **no patient leakage**.
- Observations are grouped into time-ordered **sequences** (`CacheSeqDataset`); `SequenceBatchCollater` pads/truncates each to `--seq-length`.
- Heavy MONAI transforms run once; the deterministic part is cached (in RAM, and on disk via `--cache-dir`). That's the "Loading dataset" bar on first run.

### Key knobs (`utils/config.py`)
| Arg | Meaning |
|-----|---------|
| `--arch` | backbone size: femto / pico / nano / tiny |
| `--mod-list` | which MRI modalities = input channels (default: 4 DWI b-values) |
| `--seq-length` | max timepoints per patient (default 7) |
| `--k-folds` / `--num-seeds` | CV folds x seeds -> N = folds*seeds checkpoints |
| `--num-steps` / `--warmup-steps` / `--log-every` | length / schedule / eval interval |
| `--batch-size` / `--effective-batch-size` | per-GPU batch / target batch (grad accum) |
| `--pretrained` / `--partial` | load decoder hand-off / DINO-backbone-only hand-off |
| `--run-tag` | links a pipeline's pretrain hand-off chain; default `<modality>_<arch>` |
| `--loss-fn` | `bce` (fine-tune / decoder pretrain), `dino` (backbone pretrain), `focal` |
| `--paper-sweep` | test.py: reproduce the paper's multi-arch table |

---

## 6. Scaling / practical notes
- First full-dataset run is slow to start (caches all sequences to `~/projects/hccnet/cache`); every later job reads the cache in seconds.
- Prefer **lovelace (L40)** for speed: `--gres=gpu:lovelace:1` (else generic `gpu:1` schedules fastest).
- For multi-GPU data-parallel use `--nproc_per_node=N` and scale `--effective-batch-size`.
- Pretraining length: the paper used DINO 32000 / decoder 8000 / fine-tune 400 update steps (`utils/paper_config.py`). Shorter pretraining = lighter, faster, weaker features (results between random-init and the paper's headline numbers).

---

## 7. What was changed vs. the original repo
- **`utils/config.py`** - removed hard-coded `/Users/noltinho/...` paths; defaults now point at the dataset/repo and are overridable via `HCCNET_*` env vars; `--amp` on by default; `--cache-dir` defaults to `<repo>/cache`.
- **`data/datasets.py`** - `CacheSeqDataset` gained an optional on-disk persistent cache (`--cache-dir`) so deterministic preprocessing is computed once across all jobs.
- **`models/dinohead.py`** - `weight_norm` import fixed for torch 1.13 (the original used the torch >=2.1 parametrize API).
- **`utils/paper_config.py`** (new) - the paper's fixed values (checkpoint milestones, step counts, the test-sweep arch/suffix lists, prevalence) + hand-off naming helpers.
- **`pretrain.py` / `train.py`** - pretraining stages now save/load a descriptive hand-off checkpoint instead of hard-coded `fold32000`/`fold8000` filenames, so the pretraining length is no longer pinned to a milestone value. A new `--run-tag` (default `<modality>_<arch>`, i.e. backward-compatible) keys the hand-off chain so multiple configs sharing modality+arch can run without overwriting each other.
- **`test.py`** - rewritten to evaluate the current run (ensemble of its checkpoints) by default, with `--paper-sweep` for paper reproduction.
- **`--paper` preset** - a single flag (in `pretrain.py` / `train.py`) that sets every hyper-parameter to the published values per stage (verified against arXiv:2501.10733; see `utils/paper_config.py:PAPER_STAGES` + the `hpc/stage*_paper.slurm` scripts), so the exact paper-length pipeline runs without manual parameter-hunting.
- Removed unused `models/resnet3d.py` and `models/swinvit3d.py` (never imported).
