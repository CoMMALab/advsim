"""DQN: a few steps train with a finite loss, and a checkpoint loads back to the same Q-values."""
import pathlib
import sys

import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'tools'))
from advsim.engine import obs_layout  # noqa: E402
from advsim.rl import dqn  # noqa: E402

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='Warp env on CUDA')


def test_save_load_round_trip(tmp_path):
    net = dqn.Net(hidden=32, layers=2)
    obs = torch.randint(0, 64, (5, obs_layout.OBS_DIM), dtype=torch.int16)
    legal = torch.zeros(5, 12, dtype=torch.bool)
    legal[:, [0, 3, 7]] = True
    dqn.save(net, tmp_path / 'dqn_t.npz')
    back = dqn.load(tmp_path / 'dqn_t.npz', 'cpu')
    assert torch.equal(net.q(obs, legal), back.q(obs, legal))
    dist, value = back(obs, legal)
    assert value is None and set(dist.sample().tolist()) <= {0, 3, 7}


@needs_cuda
@pytest.mark.parametrize('double', [False, True])
def test_a_few_steps_train(tmp_path, double):
    from train import train_dqn
    out = tmp_path / 'dqn_t.npz'
    train_dqn.main(['--batch', '64', '--steps', '30', '--start', '256', '--minibatch', '128', '--buffer', '4096',
                    '--hidden', '32', '--eval-every', '0', '--log-every', '10', '--save', str(out)]
                   + (['--double'] if double else []))
    assert out.exists()
