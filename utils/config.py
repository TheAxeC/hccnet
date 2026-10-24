import argparse
import os

# Default paths. Each can be overridden on the command line or via the matching
# environment variable, so the common case needs no path flags at all. Results
# and cache default to folders inside the repo; the dataset defaults to its
# location on the UTwente deepstore share.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get('HCCNET_DATA_DIR', '/deepstore/datasets/bms/hcc_study')
RESULTS_DIR = os.environ.get('HCCNET_RESULTS_DIR', os.path.join(_REPO_ROOT, 'results'))
WEIGHTS_DIR = os.environ.get('HCCNET_WEIGHTS_DIR', os.path.join(RESULTS_DIR, 'model_weights'))
CACHE_DIR = os.environ.get('HCCNET_CACHE_DIR', os.path.join(_REPO_ROOT, 'cache'))
MOD_LIST = ['DWI_b0', 'DWI_b150', 'DWI_b400', 'DWI_b800']


def parse_args() -> argparse.Namespace:

    '''
    Parse command line arguments.

    Returns:
        argparse.Namespace: Command line arguments.
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Model architecture
    parser.add_argument("--pe-mode", type=str, default='sqrt', choices=['sqrt', 'linear'],
        help="Temporal positional-encoding salience: 'sqrt' (paper Sec 3.2, months) or 'linear' (original code, years). Default sqrt = current behaviour.")
    parser.add_argument("--stem-mode", type=str, default='agnostic', choices=['agnostic', 'original'],
        help="CNN stem: 'agnostic' (paper Sec 3.4, shared 1-channel conv + pool) or 'original' (Nolte's multi-channel Conv3d, 1->N repeat at load). Default agnostic = current behaviour.")
    parser.add_argument("--arch", type=str, default='femto',
                        help="ConvNeXT model architecture. Choices are femto, pico, nano, tiny. Defaults to femto.")
    parser.add_argument("--use-v2", action='store_true',
                        help="Whether to use ConvNext v1 or v2.")
    parser.add_argument("--kernel-size", default=3, type=int,
                        help="Kernel size of convolutional downsampling layers. Defaults to 3.")
    parser.add_argument("--out-dim", default=4096, type=int,
                        help="Output dimensionality of projection head. Defaults to 4096")
    parser.add_argument("--seq-length", default=7, type=int,
                        help="Length of transformer encoder input sequence. Defaults to 7.")
    parser.add_argument("--num-classes", default=2, type=int,
                        help="Number of output classes. Defaults to 2.")
    
    # Training specifics
    parser.add_argument("--loss-fn", type=str, default='bce',
                        help="Loss function to use. Can dino, or BCE.")
    parser.add_argument("--distributed", action='store_true',
                        help="Whether to enable distributed training.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable automatic mixed precision training (on by default; use --no-amp to disable).")
    parser.add_argument("--pretrained", action='store_true',
                        help="Flag to use pretrained weights.")
    parser.add_argument("--partial", action='store_true',
                        help="Ablation (CNN-only): load only the DINO-pretrained CNN backbone; Transformer encoder random.")
    parser.add_argument("--encoder-only", action='store_true',
                        help="Ablation (encoder-only): load only the SOP-pretrained Transformer encoder + pooler; CNN backbone random.")
    parser.add_argument("--k-folds", default=5, type=int, 
                        help="Number of folds to use in cross validation. Defaults to 5.")
    parser.add_argument("--max-delta", default=3, type=int, 
                        help="Maximum time before diagnosis. Defaults to 3.")
    parser.add_argument("--num-seeds", default=3, type=int,
                        help="Number of random seeds to run training over. Defaults to 3.")
    parser.add_argument("--save-prefixes", action=argparse.BooleanOptionalAction, default=False,
                        help="At test time, also save each patient's prediction at every as-of truncation "
                             "(scans >= thr before diagnosis) so month-horizon curves are reconstructable offline.")
    parser.add_argument("--month-horizons", action=argparse.BooleanOptionalAction, default=False,
                        help="After the standard test, evaluate at the 1/3/6/12/24/36-month prediction "
                             "horizons (loads labels_Nmonths.csv per horizon) and save per-horizon predictions.")
    parser.add_argument("--horizon-files", default="1,3,6,12,24,36", type=str,
                        help="Comma-separated horizon tags for --month-horizons. Each tag T loads "
                             "labels_Tmonths.csv; the tag 1 means all observations (labels.csv).")
    parser.add_argument("--score-all-patients", action=argparse.BooleanOptionalAction, default=False,
                        help="At test time, score the whole cohort instead of the held-out test split. "
                             "The development patients trained the model, so the result is in-sample for "
                             "them and is a whole-cohort description, not a validation.")

    # Hyperparameters
    parser.add_argument("--num-steps", default=100, type=int, 
                        help="Number of steps to train for. Defaults to 100.")
    parser.add_argument("--warmup-steps", default=10, type=int, 
                        help="Number of warmup steps. Defaults to 10.")
    parser.add_argument("--log-every", default=10, type=int,
                        help="Interval to log model results. Defaults to 10.")
    parser.add_argument("--batch-size", default=4, type=int, 
                        help="Batch size to use for training. Defaults to 4.")
    parser.add_argument("--effective-batch-size", default=32, type=int,
                        help="Total batch size: batch size x number of devices x number of accumulations steps.")
    parser.add_argument("--min-learning-rate", default=1e-6, type=float, 
                        help="Final learning rate. Defaults to 1e-6.")
    parser.add_argument("--label-smoothing", default=0.0, type=float, 
                        help="Label smoothing to use for training. Defaults to 0.")
    parser.add_argument("--gamma", type=int, default=2,
                        help="Gamma parameter to use for focal loss. Defaults to 2.")
    parser.add_argument("--weight-decay", default=0.05, type=float, 
                        help="Initial weight decay value. Defaults to 0.05.")
    parser.add_argument("--max-weight-decay", default=0.5, type=float, 
                        help="Final weight decay value. Defaults to 0.5.")
    parser.add_argument("--stochastic-depth", default=0.0, type=float, 
                        help="Stochastic depth rate to use for training. Defaults to 0.")
    parser.add_argument("--dropout", default=0.0, type=float, 
                        help="Dropout rate to use for training. Defaults to 0.")
    parser.add_argument("--epsilon", default=1e-5, type=float, 
                        help="Epsilon value to use for norm layers. Defaults to 1e-5.")
    parser.add_argument("--global-crop-size", default=72, type=int, 
                        help="Global crop size to use. Defaults to 72.")
    parser.add_argument("--local-crop-size", default=48, type=int, 
                        help="Local crop size to use. Defaults to 48.")
          
    # DINO pre-training specifics
    parser.add_argument("--norm-last-layer", action='store_true',
                        help="Whether to weight normalize the last layer of the DINO head.")
    parser.add_argument("--teacher-momentum", default=0.9995, type=float, 
                        help="Inital momentum value to update teacher network with EMA. Defaults to 0.9995.")
    parser.add_argument("--teacher-temp", default=0.04, type=float, 
                        help="Final (i.e., after warmup) teacher temperature value. Defaults to 0.04")
    parser.add_argument("--teacher-warmup-temp", default=0.04, type=float, 
                        help="Initial teacher temperature value. Defaults to 0.04")

    # File paths and auxiliaries
    parser.add_argument("--mod-list", default=MOD_LIST, nargs='+', 
                        help="List of modalities to use for training. Can be DWI_b0, DWI_b150, DWI_b400, DWI_b800, T1WI, T1A, T1D, T1V, T1W_IP, T1W_OOP, T2W_TES, T2W_TEL")
    parser.add_argument("--required-modalities", nargs='+', default=None,
                        help="Require these acquired modalities for every observation, even when the model uses fewer channels. Keeps phase ablations on identical visits.")
    parser.add_argument("--seed", default=1234, type=int, 
                        help="Seed to use for reproducibility")
    parser.add_argument("--suffix", default='MedNet', type=str, 
                        help="File suffix for identification")
    parser.add_argument("--data-dir", default=DATA_DIR, type=str, 
                        help="Path to data directory")
    parser.add_argument("--results-dir", default=RESULTS_DIR, type=str, 
                        help="Path to results directory")
    parser.add_argument("--weights-dir", default=WEIGHTS_DIR, type=str,
                        help="Path to weights directory")
    parser.add_argument("--cache-dir", default=CACHE_DIR, type=str,
                        help="Directory for the on-disk preprocessing cache, so data caching only happens once across jobs. Defaults to <repo>/cache; pass '' to disable and use in-RAM caching only.")
    parser.add_argument("--paper", action='store_true',
                        help="Run this stage at the exact paper hyper-parameters (arXiv:2501.10733): sets num-steps, warm-up, effective-batch-size, seq-length, dropout, folds/seeds and weight decay for the stage (dino/decoder via pretrain.py, fine-tune via train.py). Overrides those args; see utils/paper_config.py:PAPER_STAGES.")
    parser.add_argument("--run-tag", default=None, type=str,
                        help="Identifier linking the pretraining hand-off checkpoints of one pipeline (DINO -> decoder -> fine-tune must share it). Defaults to '<modality>_<arch>'. Set a distinct value to run multiple configs that share modality+arch (e.g. different pretraining hyper-parameters) without overwriting each other.")
    parser.add_argument("--paper-sweep", action='store_true',
                        help="test.py only: reproduce the paper's evaluation over all architectures and pretrained/random suffixes (see utils/paper_config.py) instead of evaluating the single --arch/--suffix run.")
    return parser.parse_args()
