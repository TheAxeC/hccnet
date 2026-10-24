"""Exercise interrupted training and restart on CPU inside a compute allocation."""
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.cuda.amp import GradScaler
# Load the actual training-loop class without importing MRI preprocessing dependencies.
import ast
import time
import types
import numpy as np
from utils.paper_config import PRETRAIN_CHECKPOINT_STEPS
source = Path(__file__).resolve().parents[1] / 'pretrain.py'
node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'Pretrainer')
module = types.ModuleType('pretrain')
module.__dict__.update(torch=torch, nn=nn, os=os, time=time, np=np,
                       dist=torch.distributed, PRETRAIN_CHECKPOINT_STEPS=PRETRAIN_CHECKPOINT_STEPS)
sys.modules['pretrain'] = module
code = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
exec(compile(ast.fix_missing_locations(code), str(source), 'exec'), module.__dict__)
Pretrainer = module.Pretrainer
from utils import pretrain_restart as R
module.pretrain_restart = R


class Wrapped(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = nn.Linear(2, 1)


class Trial(Pretrainer):
    def __init__(self, root):
        self.student, self.teacher = Wrapped(), Wrapped()
        self.teacher.load_state_dict(self.student.state_dict())
        self.backbone_only = True
        self.optim = torch.optim.AdamW(self.student.parameters(), lr=.002)
        self.scaler = GradScaler(enabled=False)
        self.loss_fn = nn.Module()
        self.loss_fn.register_buffer('center', torch.zeros(1))
        self.results_dict = {'train': {'loss': []}}
        self.num_steps, self.gpu_id = 510, 0
        self.output_dir, self.suffix = str(root), 'restart_test'
        self.dataloaders = {'train': [None] * 7}

    def encoder_step(self, batch, step, accum_steps):
        x = torch.randn(2, 2)
        loss = self.student.module(x).square().mean() / accum_steps
        loss.backward()
        self.loss_fn.center.add_(.001)
        return loss.item()

    def accumulation_step(self, step, warmup_steps, clip_grad=True):
        self.optim.param_groups[0]['lr'] = .002 * (1 - step / 600)
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)

    def update_teacher(self, step):
        with torch.no_grad():
            for t, s in zip(self.teacher.parameters(), self.student.parameters()):
                t.mul_(.9).add_(s, alpha=.1)


def run():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        persistent = root / 'persistent' / 'restart.pth'
        metadata = {'schedule': 510, 'arm': 'test'}
        original_to = torch.Tensor.to
        def cpu_to(tensor, *args, **kwargs):
            return tensor if args == (0,) else original_to(tensor, *args, **kwargs)
        with patch.object(torch.Tensor, 'to', cpu_to), patch('pretrain.dist.all_reduce'), patch('pretrain.dist.barrier'), patch('pretrain.dist.get_world_size', return_value=1):
            torch.manual_seed(12)
            reference = Trial(root)
            with patch.dict(os.environ, {'HCCNET_PRETRAIN_RESTART': ''}):
                reference.pretrain(1, 2, 0, handoff_path=str(root/'reference.pth'))
            torch.manual_seed(12)
            interrupted = Trial(root)
            real_save = R.save

            def stop(*args):
                real_save(*args)
                raise InterruptedError('Simulated allocation loss after durable checkpoint')

            with patch.dict(os.environ, {'HCCNET_PRETRAIN_RESTART': str(persistent)}):
                with patch.object(R, 'save', side_effect=stop):
                    try:
                        interrupted.pretrain(1, 2, 0, handoff_path=str(root/'partial.pth'), restart_metadata=metadata)
                    except InterruptedError:
                        pass
                    else:
                        raise AssertionError('Interruption did not occur')
                assert persistent.exists()
                # The scratch checkpoint is expendable; restart must use the durable copy.
                (root/'restart.pth').unlink()
                resumed = Trial(root)
                resumed.pretrain(1, 2, 0, handoff_path=str(root/'resumed.pth'), restart_metadata=metadata)
                for key in reference.student.state_dict():
                    torch.testing.assert_close(reference.student.state_dict()[key], resumed.student.state_dict()[key], rtol=0, atol=0)
                    torch.testing.assert_close(reference.teacher.state_dict()[key], resumed.teacher.state_dict()[key], rtol=0, atol=0)
                torch.testing.assert_close(reference.loss_fn.center, resumed.loss_fn.center, rtol=0, atol=0)
                # A restart after final checkpoint but before handoff must finish without updates.
                final = Trial(root)
                final.pretrain(1, 2, 0, handoff_path=str(root/'final.pth'), restart_metadata=metadata)
                assert (root/'final.pth').exists()
                try:
                    R.load(final, {'schedule': 511}, root/'restart.pth')
                except ValueError:
                    pass
                else:
                    raise AssertionError('Configuration mismatch accepted')
        print('PASS: interruption, scratch loss, optimizer continuation, teacher/loss state, final handoff, configuration rejection')
        print('Scope: CPU synthetic data; not a GPU/MONAI worker replay test')


if __name__ == '__main__':
    run()
