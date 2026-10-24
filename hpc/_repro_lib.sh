# Shared tables/helpers for the paper-reproduction array jobs. Source, do not run.
# Sourced after `cd $HOME/projects/hccnet` (so paths are repo-relative).

ARCHS=(femto compact pico nano tiny)                        # 5 architectures (paper order F C P N T)
MAIN_GROUPS=(dwi t1dce t1iop t1iop_only t2_only)            # 5 groups, each gets its OWN DINO + decoder
FT_MODS=(dwi t1dce t1iop t1iop_only t2_only)                # 5 modalities fine-tuned + tested (per arch)
FUSES=(fuse_dwi_t1dce fuse_dwi_t1iop fuse_full)             # 3 fusion configs (pico only, own DINO)

# Channel list (--mod-list) for a modality / fusion key.
modlist_for() {
  case "$1" in
    dwi)            echo "DWI_b0 DWI_b150 DWI_b400 DWI_b800" ;;
    t1dce)          echo "T1WI T1A T1V T1D" ;;
    t1iop)          echo "T1W_IP T1W_OOP T2W_TES T2W_TEL" ;;
    t1iop_only)     echo "T1W_IP T1W_OOP" ;;
    t2_only)        echo "T2W_TES T2W_TEL" ;;
    fuse_dwi_t1dce) echo "DWI_b0 DWI_b150 DWI_b400 DWI_b800 T1WI T1A T1V T1D" ;;
    fuse_dwi_t1iop) echo "DWI_b0 DWI_b150 DWI_b400 DWI_b800 T1W_IP T1W_OOP T2W_TES T2W_TEL" ;;
    fuse_full)      echo "DWI_b0 DWI_b150 DWI_b400 DWI_b800 T1WI T1A T1V T1D T1W_IP T1W_OOP T2W_TES T2W_TEL" ;;
    *) echo "UNKNOWN" ;;
  esac
}

# Which pre-trained backbone a fine-tune/test should LOAD. Every modality - including
# the two constituents (T1 in/out-of-phase alone, T2 alone) - now uses its OWN backbone.
# The constituents are genuine standalone modalities: their DINO + decoder are pre-trained
# ONLY on their own 2 channels, so "T1 IOP alone" / "T2 alone" carry no representation
# leakage from the other contrast. (The former reuse of the combined 4-channel t1iop
# backbone both contaminated the standalone claim and broke the strict load - a 4-channel
# decoder stem cannot load into a 2-channel model under the original multi-channel stem.)
pretrain_group_for() {
  echo "$1"
}

WEIGHTS_DIR="$HOME/projects/hccnet/results/model_weights"   # where the code actually saves (RESULTS_DIR/model_weights)

# Unique per-job torchrun rendezvous port. `torchrun --standalone` (torch 1.13.1) uses a
# FIXED port, so two array tasks co-located on one A40 node collide on the c10d store and
# one dies mid-run with "Broken pipe / RendezvousConnectionError". Deriving the port from
# the (unique) SLURM job id gives each task its own store. Fusion's sequential torchruns
# reuse this port harmlessly (they never overlap in time).
export RDZV_PORT="$(( 20000 + ${SLURM_JOB_ID:-$$} % 40000 ))"

# Node-local NVMe staging (UTwente admin request 2026-07-15 + our HPC howto): put the torch
# env + the training cache on the compute node's fast /local NVMe instead of NFS home, and
# clean up on exit. This removes the small-file NFS reads (env) and the heavy mmap'd cache
# I/O - the latter also being the likely cause of the recurring SIGBUS (NFS-backed mmap under
# concurrent load). Checkpoints stay on NFS home: they are the cross-job hand-offs read by
# later stages on OTHER nodes, and /local is per-node. Build the env tarball ONCE on the
# login node:  ~/envs/hccnet/bin/pip install conda-pack \
#              && ~/envs/hccnet/bin/conda-pack -p ~/envs/hccnet -o ~/hccnet-env.tar.gz
# The reproducing architecture config (Nolte's original, validated by the dwi/femto slice:
# 0.920/0.738 vs the 0.928/0.693 arch-mean). The sqrt PE + channel-agnostic stem we had
# added HURT; revert both. The within-contrast DINO augmentation is always-on (not a flag)
# and stays. Every reproduction torchrun appends $REPRO_ARCH so the whole sweep is consistent.
REPRO_ARCH="--stem-mode original --pe-mode linear"

ENV_TARBALL="$HOME/hccnet-env.tar.gz"
stage_local() {
  local root="/local/${SLURM_JOB_ID:-$$}"
  mkdir -p "$root/cache"
  export HCCNET_CACHE_DIR="$root/cache"   # config.py reads this -> CacheSeqDataset on local NVMe
  trap "rm -rf '$root'" EXIT              # clean this job's scratch (env + cache) on exit
  if [ ! -f "$ENV_TARBALL" ]; then
    echo "FATAL: env tarball $ENV_TARBALL missing. Build ONCE on the login node:"
    echo "  ~/envs/hccnet/bin/pip install conda-pack && ~/envs/hccnet/bin/conda-pack -p ~/envs/hccnet -o $ENV_TARBALL"
    exit 1
  fi
  echo "[stage_local] extracting env -> $root/env"; date
  mkdir -p "$root/env"
  tar -xzf "$ENV_TARBALL" -C "$root/env"
  "$root/env/bin/conda-unpack"            # rewrite shebangs/prefix to the local path
  export PATH="$root/env/bin:$PATH"
  # The env has a pip-over-conda setuptools clobber (conda-pack couldn't pack the vendored
  # distutils cleanly -> 'No module named distutils.compat.py39'). Force Python 3.10's
  # complete stdlib distutils instead of setuptools' broken vendored copy. Build-tool only,
  # no effect on training numerics.
  export SETUPTOOLS_USE_DISTUTILS=stdlib
  echo "[stage_local] python: $(command -v python3.10)"; date
}
