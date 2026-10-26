"""policy_eval resolves baselines and its stubs raise until implemented."""
import pathlib
import sys

import pytest

pytest.importorskip('torch')  # policy_eval imports advsim.players
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'tools'))
import policy_eval  # noqa: E402


def test_baselines_resolve_to_names():
    assert policy_eval.load('maxdamage', 'cpu') == 'maxdamage'


@pytest.mark.parametrize('name', ['dqn', 'double_dqn', 'drqn', 'discrete_sac'])
def test_stubs_raise(name):
    with pytest.raises(NotImplementedError):
        policy_eval.load(name, 'cpu')


def test_an_unregistered_name_is_a_ppo_tag():
    with pytest.raises(FileNotFoundError, match='ppo_nope.npz'):
        policy_eval.load('nope', 'cpu')
