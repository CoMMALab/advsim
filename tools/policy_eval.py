"""Score policies against opponents on fixed mirrored couples, no training.

    uv run --group rl python tools/policy_eval.py v2 --temps 1,0.5,0.25,0 --against maxdamage,v2
    uv run --group rl python tools/policy_eval.py ppo:dmg46,maxdamage,dqn --against random,maxdamage

A policy is a baseline name, a name added with `register` (`dqn[:tag]`,
`ppo[:tag]`, ...) or a bare checkpoint tag (artifacts/ppo_<tag>.npz).
Temperature 0 takes the top action; temperatures apply to networks only, and
opponents play at temperature 1. A network takes `(obs [n, OBS_DIM] int16,
legal [n, 12])` and returns `(torch.distributions.Categorical, value)`, as
`net.Policy` does. Unimplemented algorithms are skipped. See docs/RL_EVAL.md.
"""
from __future__ import annotations

import argparse
import importlib
import pathlib
import sys
from typing import Callable

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from advsim import fileio, net as net_, players  # noqa: E402

# name -> loader(device, tag): a baseline name (str) or a network.
POLICIES: dict[str, Callable] = {b: (lambda b: lambda device, tag: b)(b) for b in players.BASELINES}
POLICIES['ppo'] = lambda device, tag: net_.load(players.checkpoint(tag), device)


def register(name: str, loader: Callable) -> None:
    POLICIES[name] = loader


def _rl(module: str, prefix: str) -> Callable:
    """Loads artifacts/<prefix>_<tag>.npz with advsim.rl.<module>.load (a stub until implemented)."""
    def loader(device, tag):
        return importlib.import_module(f'advsim.rl.{module}').load(
            fileio.ARTIFACTS / f'{prefix}_{tag or "baseline"}.npz', device)
    return loader


register('dqn', _rl('dqn', 'dqn'))
register('double_dqn', _rl('dqn', 'double_dqn'))
register('dueling_dqn', _rl('dqn', 'dueling_dqn'))
register('dueling_double_dqn', _rl('dqn', 'dueling_double_dqn'))
register('reinforce', lambda device, tag: net_.load(fileio.ARTIFACTS / f'reinforce_{tag or "baseline"}.npz', device))
register('grpo', lambda device, tag: net_.load(fileio.ARTIFACTS / f'grpo_{tag or "baseline"}.npz', device))
register('drqn', _rl('drqn', 'drqn'))
register('discrete_sac', _rl('discrete_sac', 'discrete_sac'))


def load(spec: str, device: str):
    """'dqn:x' -> POLICIES['dqn'](device, 'x'); an unregistered name is a PPO checkpoint tag."""
    name, _, tag = spec.partition(':')
    if name not in POLICIES:
        return POLICIES['ppo'](device, spec)
    return POLICIES[name](device, tag or None)


def main(argv=None) -> None:
    from league import Evaluator  # needs the built artifacts; keep the registry importable without them

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('policies', help='comma-separated policies to score')
    ap.add_argument('--temps', default='1', help='comma-separated temperatures for networks (0: the top action)')
    ap.add_argument('--against', default='maxdamage', help='comma-separated opponents')
    ap.add_argument('--games', type=int, default=4096, help='per policy, opponent and temperature; even')
    ap.add_argument('--seed', type=int, default=11, help='the deals')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args(argv)
    torch.set_float32_matmul_precision('high')
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda'))
    ev = Evaluator(args.games, args.device, args.seed, amp)
    opponents = {o: load(o, args.device) for o in args.against.split(',')}
    for spec in args.policies.split(','):
        try:
            net = load(spec, args.device)
        except NotImplementedError as e:
            print(f'{spec}: skipped ({e})', flush=True)
            continue
        temps = [1.0] if isinstance(net, str) else [float(x) for x in args.temps.split(',')]
        for temp in temps:
            me = net if temp == 1 else net_.Tempered(net, max(temp, 1e-3))
            scores = '  '.join(f'{o} {ev.score(me, opp):.3f}' for o, opp in opponents.items())
            print(f'{spec} at temperature {temp:g}: {scores}', flush=True)


if __name__ == '__main__':
    main()
