"""REINFORCE and GRPO (tools/train/train_pg.py): a few iterations train and save a checkpoint policy_eval loads."""
import pathlib
import sys

import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'tools'))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Warp env on CUDA')
@pytest.mark.parametrize('algo', ['reinforce', 'grpo'])
def test_a_few_iterations_train(tmp_path, algo):
    from advsim import net as net_
    from train import train_pg
    out = tmp_path / f'{algo}_t.npz'
    train_pg.main(['--algo', algo, '--batch', '64', '--group', '8', '--iters', '3', '--hidden', '32',
                   '--eval-every', '0', '--log-every', '1', '--tb', '', '--save', str(out)])
    net = net_.load(out, 'cpu')
    obs = torch.zeros(2, net.body[0].in_features, dtype=torch.int16)
    legal = torch.zeros(2, 12, dtype=torch.bool)
    legal[:, [1, 4]] = True
    assert set(net(obs, legal)[0].sample().tolist()) <= {1, 4}
