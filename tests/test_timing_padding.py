"""Local checks for structural sequence padding and zero elapsed time."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.utils import SequenceBatchCollater
from utils.utils import prep_batch


def test_real_zero_delta_is_not_padding():
    image = torch.ones(1, 2, 2, 2)
    collate = SequenceBatchCollater(keys=['image', 'label', 'delta'], seq_length=2)
    batch = collate([[{'image': image, 'label': torch.tensor(1), 'delta': torch.tensor(0.0)}]])
    _, labels, delta, padding = prep_batch(batch, batch_size=1, device='cpu')
    assert delta.tolist() == [[0.0, 0.0]]
    assert padding.tolist() == [[0.0, 1.0]]
    assert labels.tolist() == [1]


def test_existing_nonzero_delta_remains_real():
    image = torch.ones(1, 2, 2, 2)
    collate = SequenceBatchCollater(keys=['image', 'label', 'delta'], seq_length=2)
    batch = collate([[{'image': image, 'label': torch.tensor(0), 'delta': torch.tensor(3.0)}]])
    _, _, delta, padding = prep_batch(batch, batch_size=1, device='cpu')
    assert delta.tolist() == [[3.0, 0.0]]
    assert padding.tolist() == [[0.0, 1.0]]


if __name__ == '__main__':
    test_real_zero_delta_is_not_padding()
    test_existing_nonzero_delta_remains_real()
    print('PASS: structural padding distinguishes a real zero delta from a padded step')
