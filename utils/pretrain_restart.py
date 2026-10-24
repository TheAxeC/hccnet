"""Single-process pretraining restart state, copied atomically from scratch."""
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch


def models(trainer):
    names = ['student', 'teacher'] if trainer.backbone_only else ['model']
    return {name: getattr(getattr(trainer, name), 'module', getattr(trainer, name)) for name in names}


def save(trainer, completed, metadata, local_path, persistent_path):
    state = dict(version=1, completed=completed, metadata=metadata,
                 models={k: m.state_dict() for k, m in models(trainer).items()},
                 optimizer=trainer.optim.state_dict(), scaler=trainer.scaler.state_dict(),
                 loss=trainer.loss_fn.state_dict(), history=trainer.results_dict,
                 random=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    local_path, persistent_path = Path(local_path), Path(persistent_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    persistent_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = local_path.with_suffix('.tmp')
    torch.save(state, temporary)
    os.replace(temporary, local_path)
    temporary = persistent_path.with_name(persistent_path.name + f'.{os.getpid()}.tmp')
    try:
        shutil.copyfile(local_path, temporary)
        os.replace(temporary, persistent_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f'Restart state preserved after {completed} optimizer updates', flush=True)


def load(trainer, metadata, local_path):
    state = torch.load(local_path, map_location='cpu')
    if state['version'] != 1 or state['metadata'] != metadata:
        raise ValueError('Restart configuration differs from saved training configuration')
    completed = int(state['completed'])
    if not 0 <= completed <= trainer.num_steps:
        raise ValueError('Restart update count outside the training schedule')
    for name, model in models(trainer).items():
        model.load_state_dict(state['models'][name], strict=True)
    trainer.optim.load_state_dict(state['optimizer'])
    trainer.scaler.load_state_dict(state['scaler'])
    trainer.loss_fn.load_state_dict(state['loss'])
    trainer.results_dict = state['history']
    random.setstate(state['random'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    print(f'Resuming after {completed} optimizer updates; data-loader workers restart', flush=True)
    return completed
