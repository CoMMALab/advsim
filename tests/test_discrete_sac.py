"""Discrete SAC: a checkpoint loads back to the same actor and critics, and a few steps train."""
import pathlib
import sys

import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'tools'))
from advsim.engine import obs_layout  # noqa: E402
from advsim.rl import discrete_sac  # noqa: E402


def test_save_load_round_trip(tmp_path):
    net = discrete_sac.Net(hidden=32, layers=2)
    obs = torch.randint(0, 64, (5, obs_layout.OBS_DIM), dtype=torch.int16)
    legal = torch.zeros(5, 12, dtype=torch.bool)
    legal[:, [0, 3, 7]] = True
    discrete_sac.save(net, tmp_path / 'discrete_sac_t.npz')
    back = discrete_sac.load(tmp_path / 'discrete_sac_t.npz', 'cpu')
    assert torch.equal(net.logits(obs, legal), back.logits(obs, legal))
    assert all(torch.equal(a, b) for a, b in zip(net.qs(obs), back.qs(obs)))
    dist, value = back(obs, legal)
    assert value is None and set(dist.sample().tolist()) <= {0, 3, 7}


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Warp env on CUDA')
def test_a_few_steps_train(tmp_path):
    from train import train_discrete_sac
    out = tmp_path / 'discrete_sac_t.npz'
    train_discrete_sac.main(['--batch', '64', '--steps', '30', '--start', '256', '--minibatch', '128', '--buffer', '4096',
                             '--hidden', '32', '--eval-every', '0', '--log-every', '10', '--tb', '', '--save', str(out)])
    assert out.exists()
