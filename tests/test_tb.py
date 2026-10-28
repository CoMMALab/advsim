"""tools/tb.py: a curve file and an Elo summary land in TensorBoard event files under the right tags."""
import json
import pathlib
import sys

import pytest

pytest.importorskip('tensorboard')
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'tools'))
import tb  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402


def tags(run: pathlib.Path) -> dict:
    acc = EventAccumulator(str(run), size_guidance={'tensors': 0, 'scalars': 0})
    acc.Reload()
    return acc.Tags()


def test_curves_and_elo(tmp_path):
    (tmp_path / 'dqn_x_curve.jsonl').write_text(''.join(json.dumps({'step': s, 'minutes': s / 10, 'maxdamage': 0.5})
                                                       + '\n' for s in (2000, 4000)))
    (tmp_path / 'ppo_y_curve.jsonl').write_text(json.dumps({'iter': 100, 'minutes': 1.0, 'maxdamage': 0.6}) + '\n')
    (tmp_path / 'summary.json').write_text(json.dumps({
        'players': ['random', 'dqn:x'], 'elo': [1000.0, 1500.0], 'lo': [1000.0, 1450.0], 'hi': [1000.0, 1550.0],
        'games': 512, 'matrix': {'random': {'random': None, 'dqn:x': 0.1}, 'dqn:x': {'random': 0.9, 'dqn:x': None}}}))
    logs = tmp_path / 'runs'
    tb.main(['--logdir', str(logs), '--curves', str(tmp_path / 'dqn_x_curve.jsonl'), str(tmp_path / 'ppo_y_curve.jsonl'),
             '--elo', str(tmp_path / 'summary.json')])
    assert {'eval/maxdamage', 'eval/minutes'} <= set(tags(logs / 'dqn_x')['scalars'])
    acc = EventAccumulator(str(logs / 'ppo_y')); acc.Reload()
    assert [e.step for e in acc.Scalars('eval/maxdamage')] == [6400]  # iterations of 64 decisions
    elo = tags(logs / 'elo')
    assert {'elo/random', 'elo/dqn:x'} <= set(elo['scalars'])
    assert {'elo/ratings/text_summary', 'elo/head_to_head/text_summary'} <= set(elo['tensors'])
