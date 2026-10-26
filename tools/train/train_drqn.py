"""Train advsim.rl.drqn. Score the result with tools/policy_eval.py.

    uv run --group rl python tools/train/train_drqn.py
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]  # advsim, and league.py beside the tools


def main(argv=None) -> None:
    # TODO: argparse (batch, steps, seed, device, checkpoint tag) like tools/train/train_league.py.
    # TODO: rollouts from advsim.env.Gen3Env; opponents through advsim.players (Baselines, net_act).
    # TODO: the update.
    # TODO: periodic scores with league.Evaluator; save to artifacts/drqn_<tag>.npz via advsim.rl.drqn.save.
    raise NotImplementedError('tools/train/train_drqn.py')


if __name__ == '__main__':
    main()
