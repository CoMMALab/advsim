"""Critic-free policy gradient: REINFORCE and GRPO, in the PPO league. Score the result with tools/policy_eval.py.

    uv run --group rl python tools/train/train_pg.py --algo reinforce --minutes 30 --save artifacts/reinforce_mlp.npz
    uv run --group rl python tools/train/train_pg.py --algo grpo --minutes 30 --save artifacts/grpo_mlp.npz

Each iteration deals fresh teams to every slot, draws opponents from the league
(self-play, snapshots, baselines) and plays the games out. Every learner
decision with a real choice is credited with its game's final result (win 1,
tie 0, loss -1; no discount, no value network).

REINFORCE: advantage = result - mean result over the batch; one full-batch
gradient step on -advantage * log pi, so every sample is on-policy.

GRPO: slots come in groups of --group that start as forks of one deal (same
teams and opponent, a fresh RNG key each), so the baseline is the group's own
mean and the deal's luck cancels. Advantage = (result - group mean) / (group
std + 1e-4); --epochs of PPO's clipped ratio, minus --kl times the k3 KL
estimate against a reference policy refreshed every --ref-every iterations.

An iteration stops once --finish of its games are over; games still going
are dropped, which slightly under-weights long games.
"""
from __future__ import annotations

import argparse
import copy
import contextlib
import json
import pathlib
import sys
import time

import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]  # advsim, and league.py beside the tools

from advsim import fileio, net as net_  # noqa: E402
from advsim.env import Gen3Env  # noqa: E402
from advsim.net import Policy, legal_bits  # noqa: E402
from advsim.players import Baselines  # noqa: E402
from league import BASE, NAMES, SELF, Evaluator, League, Runner, learner_mask  # noqa: E402
import tb  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--algo', choices=['reinforce', 'grpo'], required=True)
    ap.add_argument('--batch', type=int, default=4096, help='battles played at once')
    ap.add_argument('--group', type=int, default=8, help='grpo: forks per deal')
    ap.add_argument('--iters', type=int, default=1000, help='iterations (unbounded with --minutes)')
    ap.add_argument('--minutes', type=float, default=0, help='stop after this much training time (0: --iters)')
    ap.add_argument('--finish', type=float, default=0.98, help='end an iteration once this fraction of games is over')
    ap.add_argument('--max-steps', type=int, default=400, help='decisions before a battle is truncated')
    ap.add_argument('--lr', type=float, default=3e-4, help='Adam')
    ap.add_argument('--ent', type=float, default=0.01, help='entropy bonus')
    ap.add_argument('--epochs', type=int, default=4, help='grpo: passes over each batch (reinforce: 1)')
    ap.add_argument('--minibatch', type=int, default=32768, help='grpo: rows per gradient step (reinforce: the whole batch)')
    ap.add_argument('--clip', type=float, default=0.2, help='grpo: ratio clip')
    ap.add_argument('--kl', type=float, default=0.01, help='grpo: KL penalty against the reference policy')
    ap.add_argument('--ref-every', type=int, default=50, help='grpo: iterations between reference refreshes')
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--mix', default='0.5,0.25,0.25', help='self, snapshots, baselines')
    ap.add_argument('--snapshot-every', type=int, default=25, help='iterations between league snapshots')
    ap.add_argument('--pool', type=int, default=10, help='snapshots kept in the league')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--save', default=None, help='the final network; evaluations also write <stem>_curve.jsonl')
    ap.add_argument('--eval-every', type=int, default=25, help='iterations between evaluations against maxdamage (0: none)')
    ap.add_argument('--eval-games', type=int, default=2048)
    ap.add_argument('--eval-seed', type=int, default=7, help='the same evaluation teams as the other trainers')
    ap.add_argument('--log-every', type=int, default=5)
    ap.add_argument('--tb', default=str(tb.RUNS), help="TensorBoard logdir; the run is the --save stem ('': off)")
    args = ap.parse_args(argv)
    grpo = args.algo == 'grpo'
    if grpo and args.batch % args.group:
        ap.error('--batch must be a multiple of --group')
    epochs = args.epochs if grpo else 1
    if args.minutes:
        args.iters = 1 << 62

    torch.manual_seed(args.seed)
    torch.distributions.Distribution.set_default_validate_args(False)
    dev, B = args.device, args.batch
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    env = Gen3Env(B, device=dev, max_steps=args.max_steps)
    baselines = Baselines(env)
    net = Policy(args.hidden, args.layers).to(dev)  # the value head is never trained
    ref = copy.deepcopy(net).eval()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    runner = Runner(net, False)
    league = League(B, [float(x) for x in args.mix.split(',')], args.pool, dev, gen)
    evaluator = Evaluator(args.eval_games, dev, args.eval_seed, contextlib.nullcontext) if args.eval_every else None
    w = tb.writer(args.tb, tb.run_name(args.save, args.algo)) if args.tb else None
    slots = torch.arange(B, device=dev)
    leader = slots - slots % args.group if grpo else slots
    curve, decisions, started = [], 0, time.time()

    for it in range(args.iters):
        if args.minutes and time.time() - started > args.minutes * 60:
            break
        t0 = time.time()
        if it and it % args.snapshot_every == 0:
            league.snapshot(net)
        if grpo and it % args.ref_every == 0:
            ref.load_state_dict(net.state_dict())
        env.reset(seed=args.seed * 1_000_003 + it)
        league.draw(torch.ones(B, dtype=torch.bool, device=dev))
        if grpo:  # every slot of a group plays its leader's deal against its leader's opponent
            rest = slots[slots != leader]
            env.fork(leader[rest].int(), rest.int(), salt=it)
            league.opp, league.side = league.opp[leader], league.side[leader]
        alive = torch.ones(B, dtype=torch.bool, device=dev)
        ended = torch.zeros(B, dtype=torch.bool, device=dev)  # finished, not truncated: these games are scored
        result = torch.zeros(B, device=dev)  # p1's final reward
        buf = {k: [] for k in ('obs', 'legal', 'act', 'logp', 'who')}
        with torch.no_grad():
            for step in range(args.max_steps):
                obs, mask = env.observe()
                obs, legal = obs.clone(), legal_bits(mask)
                dist, _ = net(obs.view(B * 2, -1), legal.view(B * 2, -1))
                act = dist.sample().view(B, 2)
                acts = act.clone()
                foe = 1 - league.side
                for k, s in league.groups():
                    f = foe[s]
                    if k < args.pool:
                        acts[s, f] = runner(league.pool[k], obs[s, f], legal[s, f])[0].sample()
                    else:
                        acts[s, f] = baselines.act(s, f, league.opp[s] - BASE, args.seed, it * args.max_steps + step)
                keep = (learner_mask(league.opp, league.side) & (legal.sum(-1) > 1) & alive.unsqueeze(-1)).view(-1)
                buf['obs'].append(obs.view(B * 2, -1)[keep])
                buf['legal'].append(legal.view(B * 2, -1)[keep])
                buf['act'].append(act.view(-1)[keep])
                buf['logp'].append(dist.log_prob(act.view(-1))[keep])
                buf['who'].append(torch.arange(B * 2, device=dev)[keep])  # slot * 2 + side
                env.step(acts)
                done, trunc = env.done.bool(), env.truncated.bool()
                result = torch.where(alive & done, env.reward, result)
                ended |= alive & done
                league.record(alive & done, torch.stack([env.reward, -env.reward], -1)[slots, league.side])
                alive &= ~(done | trunc)
                decisions += 1
                # ponytail: fixed cutoff drops the longest games; carry them into the next iteration if that bias shows
                if step % 16 == 15 and float((~alive).float().mean()) >= args.finish:
                    break
            assert not env.err.any(), 'an illegal action reached the engine'
        # Each slot's outcome from each side; games still alive or truncated count for nothing.
        r = torch.stack([result, -result], -1).view(-1)  # [B * 2] by slot * 2 + side
        valid = ended.repeat_interleave(2)
        if grpo:
            g = (torch.arange(B * 2, device=dev) // 2) // args.group * 2 + torch.arange(B * 2, device=dev) % 2
            n_g = torch.zeros(B // args.group * 2, device=dev).index_add_(0, g, valid.float())
            s_g = torch.zeros_like(n_g).index_add_(0, g, r * valid)
            mean = s_g / n_g.clamp(min=1)
            var = torch.zeros_like(n_g).index_add_(0, g, (r - mean[g]) ** 2 * valid) / n_g.clamp(min=1)
            adv_side = (r - mean[g]) / (var.sqrt()[g] + 1e-4)
        else:
            adv_side = r - (r * valid).sum() / valid.sum().clamp(min=1)
        flat = {k: torch.cat(v) for k, v in buf.items()}
        sel = valid[flat['who']]
        flat = {k: v[sel] for k, v in flat.items()}
        adv = adv_side[flat['who']]
        n = len(adv)
        mb = args.minibatch if grpo else n
        for _ in range(epochs):
            perm = torch.randperm(n, device=dev)
            for s0 in range(0, n, mb):
                i = perm[s0:s0 + mb]
                dist, _ = net(flat['obs'][i], flat['legal'][i])
                logp = dist.log_prob(flat['act'][i])
                if grpo:
                    ratio = (logp - flat['logp'][i]).exp()
                    pg = -torch.min(ratio * adv[i], ratio.clamp(1 - args.clip, 1 + args.clip) * adv[i]).mean()
                    with torch.no_grad():
                        ref_logp = ref(flat['obs'][i], flat['legal'][i])[0].log_prob(flat['act'][i])
                    kl = ((ref_logp - logp).exp() - (ref_logp - logp) - 1).mean()  # k3, sampled from the old policy
                else:
                    pg, kl = -(adv[i] * logp).mean(), torch.zeros((), device=dev)
                ent = dist.entropy().mean()
                loss = pg + args.kl * kl - args.ent * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
        assert torch.isfinite(loss), 'a non-finite loss'

        if it % args.log_every == 0:
            stats = {'train/loss': loss.item(), 'train/entropy': ent.item(), 'train/kl': kl.item(),
                     'train/transitions': n, 'train/finished': float(valid.float().mean()),
                     'train/decisions_per_s': (step + 1) * B / (time.time() - t0),
                     **{f'win/{name}': float(x) for name, x in zip(NAMES, league.base_wr)}}
            if w:
                for k, v in stats.items():
                    w.add_scalar(k, v, decisions)
            wr = ' '.join(f'{name[:6]} {float(x):.2f}' for name, x in zip(NAMES, league.base_wr))
            print(f'iter {it:4d}  decisions {decisions:6d}  transitions {n:7d}  loss {loss.item():+.3f}  '
                  f'entropy {ent.item():.3f}  kl {kl.item():.4f}  win vs {wr}  '
                  f'{stats["train/decisions_per_s"]:,.0f} decisions/s', flush=True)
        if evaluator and (it + 1) % args.eval_every == 0:
            net.eval()
            row = {'step': decisions, 'iter': it + 1, 'minutes': round((time.time() - started) / 60, 2),
                   'maxdamage': evaluator.score(net, 'maxdamage')}
            net.train()
            curve.append(row)
            print('eval ' + '  '.join(f'{k} {v}' for k, v in row.items()), flush=True)
            if w:
                w.add_scalar('eval/maxdamage', row['maxdamage'], decisions)
                w.add_scalar('eval/minutes', row['minutes'], decisions)
            if args.save:
                stem = pathlib.Path(args.save)
                fileio.write_text(stem.with_name(stem.stem + '_curve.jsonl'), ''.join(json.dumps(x) + '\n' for x in curve))
                net_.save(net, args.save)
    if w:
        w.close()
    if args.save:
        net_.save(net, args.save)
        print(f'saved {args.save}')


if __name__ == '__main__':
    main()
