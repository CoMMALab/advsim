"""Train advsim.rl.dqn in the PPO league (self-play, snapshots, baselines). Score the result with tools/policy_eval.py.

    uv run --group rl python tools/train/train_dqn.py --minutes 30 --save artifacts/dqn_mlp.npz
    uv run --group rl python tools/train/train_dqn.py --double --minutes 30 --save artifacts/double_dqn_mlp.npz

Every decision, the learner acts epsilon-greedily wherever it plays (both sides
in self-play), and each of its transitions (obs, action, reward, next obs on the
same side) goes into a replay buffer on the GPU. Truncated games are dropped,
since the next observation already belongs to the next game. Then `--updates`
minibatches of 1-step Huber TD against a target network synced every
`--target-every` updates. With --double the online network picks the next
action and the target network scores it.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import pathlib
import sys
import time

import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]  # advsim, and league.py beside the tools

from advsim import fileio  # noqa: E402
from advsim.engine import obs_layout  # noqa: E402
from advsim.env import Gen3Env  # noqa: E402
from advsim.net import ACTIONS, legal_bits  # noqa: E402
from advsim.players import Baselines  # noqa: E402
from advsim.rl import dqn  # noqa: E402
from league import BASE, NAMES, Evaluator, League, Runner, learner_mask  # noqa: E402


class Replay:
    """A circular buffer of transitions on the device."""

    def __init__(self, size: int, device) -> None:
        self.size, self.n, self.at = size, 0, 0
        z = lambda *s, dtype=torch.float32: torch.zeros(size, *s, dtype=dtype, device=device)
        self.obs, self.next = z(obs_layout.OBS_DIM, dtype=torch.int16), z(obs_layout.OBS_DIM, dtype=torch.int16)
        self.next_legal = z(ACTIONS, dtype=torch.bool)
        self.act, self.rew, self.done = z(dtype=torch.long), z(), z()

    def add(self, obs, act, rew, done, nxt, next_legal) -> None:
        i = (self.at + torch.arange(len(obs), device=obs.device)) % self.size
        for buf, x in ((self.obs, obs), (self.act, act), (self.rew, rew), (self.done, done),
                       (self.next, nxt), (self.next_legal, next_legal)):
            buf[i] = x
        self.at = (self.at + len(obs)) % self.size
        self.n = min(self.n + len(obs), self.size)

    def sample(self, k: int, gen):
        i = torch.randint(0, self.n, (k,), device=self.obs.device, generator=gen)
        return self.obs[i], self.act[i], self.rew[i], self.done[i], self.next[i], self.next_legal[i]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--double', action='store_true', help='Double DQN targets')
    ap.add_argument('--batch', type=int, default=4096, help='battles played at once')
    ap.add_argument('--steps', type=int, default=200_000, help='decisions (unbounded with --minutes)')
    ap.add_argument('--minutes', type=float, default=0, help='stop after this much training time (0: --steps)')
    ap.add_argument('--buffer', type=int, default=1 << 21, help='replay transitions kept')
    ap.add_argument('--start', type=int, default=100_000, help='transitions before the first update')
    ap.add_argument('--minibatch', type=int, default=8192)
    ap.add_argument('--updates', type=int, default=2, help='minibatches per decision')
    ap.add_argument('--target-every', type=int, default=1000, help='updates between target syncs')
    ap.add_argument('--lr', type=float, default=1e-4, help='Adam')
    ap.add_argument('--gamma', type=float, default=0.995, help='discount')
    ap.add_argument('--eps', default='1.0,0.05', help='epsilon start,end')
    ap.add_argument('--eps-steps', type=int, default=5000, help='decisions over which epsilon decays linearly')
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--max-steps', type=int, default=400, help='decisions before a battle is truncated')
    ap.add_argument('--mix', default='0.5,0.25,0.25', help='self, snapshots, baselines')
    ap.add_argument('--snapshot-every', type=int, default=1600, help='decisions between league snapshots')
    ap.add_argument('--pool', type=int, default=10, help='snapshots kept in the league')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--save', default=None, help='the final network; evaluations also write <stem>_curve.jsonl')
    ap.add_argument('--eval-every', type=int, default=2000, help='decisions between evaluations against maxdamage (0: none)')
    ap.add_argument('--eval-games', type=int, default=2048)
    ap.add_argument('--eval-seed', type=int, default=7, help='the same evaluation teams as train_league.py')
    ap.add_argument('--log-every', type=int, default=200)
    args = ap.parse_args(argv)
    if not args.minutes and not args.steps:
        ap.error('give --steps or --minutes')

    torch.manual_seed(args.seed)
    torch.distributions.Distribution.set_default_validate_args(False)
    dev, B = args.device, args.batch
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    env = Gen3Env(B, device=dev, max_steps=args.max_steps)
    env.reset(seed=args.seed)
    baselines = Baselines(env)
    net = dqn.Net(args.hidden, args.layers).to(dev)
    target = copy.deepcopy(net).eval()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    replay = Replay(args.buffer, dev)
    runner = Runner(net, False)
    league = League(B, [float(x) for x in args.mix.split(',')], args.pool, dev, gen)
    evaluator = Evaluator(args.eval_games, dev, args.eval_seed, contextlib.nullcontext) if args.eval_every else None
    eps0, eps1 = (float(x) for x in args.eps.split(','))
    curve, updates, loss = [], 0, torch.zeros((), device=dev)
    prev = None
    started = t0 = time.time()

    step = 0
    while True:
        out_of_time = args.minutes and time.time() - started > args.minutes * 60
        if out_of_time or (not args.minutes and step >= args.steps):
            break
        if step and step % args.snapshot_every == 0:
            league.snapshot(net)
        obs, mask = env.observe()
        obs, legal = obs.clone(), legal_bits(mask)
        if prev is not None:  # last decision's transitions end here
            p_obs, p_act, p_rew, p_done, keep = prev
            replay.add(p_obs[keep], p_act[keep], p_rew[keep], p_done[keep], obs[keep], legal[keep])

        with torch.no_grad():
            greedy = net.q(obs.view(B * 2, -1), legal.view(B * 2, -1)).argmax(-1).view(B, 2)
        eps = eps0 + (eps1 - eps0) * min(step / args.eps_steps, 1.0)
        explore = torch.multinomial(legal.view(B * 2, -1).float(), 1, generator=gen).view(B, 2)
        act = torch.where(torch.rand(B, 2, device=dev, generator=gen) < eps, explore, greedy)
        acts = act.clone()
        foe = 1 - league.side
        for k, s in league.groups():
            f = foe[s]
            if k < args.pool:
                with torch.no_grad():
                    acts[s, f] = runner(league.pool[k], obs[s, f], legal[s, f])[0].sample()
            else:
                acts[s, f] = baselines.act(s, f, league.opp[s] - BASE, args.seed, step)
        mine = learner_mask(league.opp, league.side)
        env.step(acts)
        r = env.reward.clone()
        done, trunc = env.done.bool().clone(), env.truncated.bool().clone()
        rew = torch.stack([r, -r], dim=-1)
        league.record(done, rew[torch.arange(B, device=dev), league.side])
        both = lambda x: x.unsqueeze(-1).expand(B, 2)
        prev = (obs, act, rew, both(done).float(), mine & ~both(trunc))
        league.draw(done | trunc)
        assert not env.err.any(), 'an illegal action reached the engine'

        if replay.n >= args.start:
            for _ in range(args.updates):
                o, a, rw, d, nx, nl = replay.sample(args.minibatch, gen)
                with torch.no_grad():
                    tq = target.q(nx, nl)
                    pick = net.q(nx, nl).argmax(-1, keepdim=True) if args.double else tq.argmax(-1, keepdim=True)
                    y = rw + args.gamma * (1 - d) * tq.gather(1, pick).squeeze(1)
                q = net.q(o, torch.ones_like(nl)).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(q, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                opt.step()
                updates += 1
                if updates % args.target_every == 0:
                    target.load_state_dict(net.state_dict())
        step += 1

        if step % args.log_every == 0:
            assert torch.isfinite(loss), 'a non-finite loss'
            wr = ' '.join(f'{name[:6]} {float(w):.2f}' for name, w in zip(NAMES, league.base_wr))
            print(f'step {step:6d}  eps {eps:.3f}  buffer {replay.n:8d}  updates {updates:6d}  loss {loss.item():.4f}  '
                  f'win vs {wr}  {args.log_every * B / (time.time() - t0):,.0f} decisions/s', flush=True)
            t0 = time.time()
        if evaluator and step % args.eval_every == 0:
            net.eval()
            row = {'step': step, 'minutes': round((time.time() - started) / 60, 2),
                   'maxdamage': evaluator.score(net, 'maxdamage')}
            net.train()
            curve.append(row)
            print('eval ' + '  '.join(f'{k} {v}' for k, v in row.items()), flush=True)
            if args.save:
                stem = pathlib.Path(args.save)
                fileio.write_text(stem.with_name(stem.stem + '_curve.jsonl'), ''.join(json.dumps(x) + '\n' for x in curve))
                dqn.save(net, args.save)
    if args.save:
        dqn.save(net, args.save)
        print(f'saved {args.save}')


if __name__ == '__main__':
    main()
