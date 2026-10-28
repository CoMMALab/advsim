"""TensorBoard logs for comparing RL runs and the Elo bracket.

    uv run --group rl python tools/tb.py                       # import every artifacts/*_curve.jsonl and artifacts/elo
    uv run --group rl tensorboard --logdir runs                # then open http://localhost:6006

Training runs (train_league.py, train_dqn.py --tb) log to runs/<name>, with
the step in decisions per battle slot (a PPO iteration is --steps decisions),
so PPO and DQN share an axis; TensorBoard's "Relative" axis is wall time.
The bracket goes to runs/elo: a sorted table and the head-to-head matrix in
the Text tab, and elo/<player> in Scalars at step = games in the fit, so each
`elo.py --add` adds a point.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from advsim import fileio  # noqa: E402

RUNS = ROOT / 'runs'


def writer(logdir, name: str):
    from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(str(pathlib.Path(logdir) / name))


def run_name(save, algo: str) -> str:
    """'artifacts/dqn_mlp.npz' -> 'dqn_mlp'; no --save -> '<algo>_<time>'."""
    import time
    return pathlib.Path(save).stem if save else f'{algo}_{time.strftime("%m%d-%H%M%S")}'


def log_elo(logdir, summary: dict) -> None:
    """The bracket's ratings as a table and scalars in runs/elo."""
    order = np.argsort(summary['elo'])[::-1]
    names = [summary['players'][i] for i in order]
    table = '| player | Elo | 95% interval |\n|---|---|---|\n' + ''.join(
        f"| {summary['players'][i]} | {summary['elo'][i]:.0f} | [{summary['lo'][i]:.0f}, {summary['hi'][i]:.0f}] |\n"
        for i in order)
    matrix = ('row player\'s score against column player\n\n| | ' + ' | '.join(names) + ' |\n|'
              + '---|' * (len(names) + 1) + '\n' + ''.join(
                  f'| **{a}** | ' + ' | '.join('-' if summary['matrix'][a][b] is None else f"{summary['matrix'][a][b]:.3f}"
                                              for b in names) + ' |\n' for a in names))
    w = writer(logdir, 'elo')
    step = summary['games']
    w.add_text('elo/ratings', table, step)
    w.add_text('elo/head_to_head', matrix, step)
    for i in order:
        w.add_scalar(f"elo/{summary['players'][i]}", summary['elo'][i], step)
    w.close()


def import_curve(logdir, path: pathlib.Path) -> None:
    """A trainer's <stem>_curve.jsonl as eval/<opponent> in runs/<stem>. PPO rows are per iteration
    of 64 decisions (train_league.py's default --steps), DQN rows per decision."""
    rows = list(fileio.iter_jsonl(path))
    w = writer(logdir, path.stem.removesuffix('_curve'))
    for r in rows:
        step = r['step'] if 'step' in r else r['iter'] * 64
        for k, v in r.items():
            if k not in ('step', 'iter'):
                w.add_scalar(f'eval/{k}', v, step)
    w.close()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--logdir', default=str(RUNS))
    ap.add_argument('--curves', nargs='*', default=None, help='curve files (default artifacts/*_curve.jsonl)')
    ap.add_argument('--elo', default=str(fileio.ARTIFACTS / 'elo' / 'summary.json'), help="elo.py's summary.json")
    args = ap.parse_args(argv)
    curves = [pathlib.Path(c) for c in args.curves] if args.curves is not None else sorted(fileio.ARTIFACTS.glob('*_curve.jsonl'))
    for c in curves:
        import_curve(args.logdir, c)
        print(f'{c} -> {args.logdir}/{c.stem.removesuffix("_curve")}')
    if pathlib.Path(args.elo).exists():
        log_elo(args.logdir, fileio.read_json(pathlib.Path(args.elo)))
        print(f'{args.elo} -> {args.logdir}/elo')


if __name__ == '__main__':
    main()
