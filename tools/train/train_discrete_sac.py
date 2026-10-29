"""Train advsim.rl.discrete_sac in the PPO league (self-play, snapshots, baselines). Score it with tools/policy_eval.py.

    uv run --group rl python tools/train/train_discrete_sac.py --minutes 30 --save artifacts/discrete_sac_mlp.npz

Discrete SAC (Christodoulou 2019): the learner samples from its actor wherever
it plays, and its transitions go into the same GPU replay as train_dqn.py.
Each update: twin critics regress on r + gamma * (1 - done) * sum_a' pi(a'|s')
[min Q_target(s', a') - alpha log pi(a'|s')]; the actor minimizes sum_a pi(a|s)
[alpha log pi(a|s) - min Q(s, a)]; alpha is tuned so the policy's entropy
tracks --entropy-ratio * log(legal actions) per state, since the number of
legal actions varies. Target critics follow by Polyak averaging (--tau).

alpha starts small (--alpha-init 0.005): the only reward is +-1 at the end of
a game about 50 decisions long, and at SAC's usual alpha = 1 the entropy bonus
summed over the horizon dwarfs it, so the policy learns to seek states with
many legal actions instead of winning.
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
from advsim.env import Gen3Env  # noqa: E402
from advsim.net import legal_bits  # noqa: E402
from advsim.players import Baselines  # noqa: E402
from advsim.rl import discrete_sac  # noqa: E402
from league import BASE, NAMES, Evaluator, League, Runner, learner_mask  # noqa: E402
from train.train_dqn import Replay  # noqa: E402
import tb  # noqa: E402


def soft_v(net, obs, legal, critics, alpha):
    """sum_a pi(a|s) [min Q(s, a) - alpha log pi(a|s)], and pi, log pi."""
    logp = torch.log_softmax(net.logits(obs, legal), -1)
    p = logp.exp()
    q = torch.min(*critics)
    return (p * (q - alpha * logp)).sum(-1), p, logp


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--batch', type=int, default=4096, help='battles played at once')
    ap.add_argument('--steps', type=int, default=200_000, help='decisions (unbounded with --minutes)')
    ap.add_argument('--minutes', type=float, default=0, help='stop after this much training time (0: --steps)')
    ap.add_argument('--buffer', type=int, default=1 << 21, help='replay transitions kept')
    ap.add_argument('--start', type=int, default=100_000, help='transitions before the first update')
    ap.add_argument('--minibatch', type=int, default=8192)
    ap.add_argument('--updates', type=int, default=2, help='minibatches per decision')
    ap.add_argument('--lr', type=float, default=3e-4, help='Adam, for the actor, the critics and alpha')
    ap.add_argument('--gamma', type=float, default=0.995, help='discount')
    ap.add_argument('--tau', type=float, default=0.005, help='Polyak rate of the target critics')
    ap.add_argument('--alpha-init', type=float, default=0.005, help='the starting entropy temperature')
    ap.add_argument('--entropy-ratio', type=float, default=0.5, help='target entropy as a fraction of log(legal actions)')
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
    ap.add_argument('--eval-seed', type=int, default=7, help='the same evaluation teams as the other trainers')
    ap.add_argument('--log-every', type=int, default=200)
    ap.add_argument('--tb', default=str(tb.RUNS), help="TensorBoard logdir; the run is the --save stem ('': off)")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    torch.distributions.Distribution.set_default_validate_args(False)
    dev, B = args.device, args.batch
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    env = Gen3Env(B, device=dev, max_steps=args.max_steps)
    env.reset(seed=args.seed)
    baselines = Baselines(env)
    net = discrete_sac.Net(args.hidden, args.layers).to(dev)
    target = copy.deepcopy(net).eval()
    actor_opt = torch.optim.Adam(net.actor.parameters(), lr=args.lr)
    critic_opt = torch.optim.Adam([*net.q1.parameters(), *net.q2.parameters()], lr=args.lr)
    log_alpha = torch.tensor(args.alpha_init, device=dev).log().requires_grad_()
    alpha_opt = torch.optim.Adam([log_alpha], lr=args.lr)
    replay = Replay(args.buffer, dev)
    runner = Runner(net, False)
    league = League(B, [float(x) for x in args.mix.split(',')], args.pool, dev, gen)
    evaluator = Evaluator(args.eval_games, dev, args.eval_seed, contextlib.nullcontext) if args.eval_every else None
    w = tb.writer(args.tb, tb.run_name(args.save, 'discrete_sac')) if args.tb else None
    zero = torch.zeros((), device=dev)
    curve, updates, prev = [], 0, None
    critic_loss = actor_loss = entropy = zero
    started = t0 = time.time()

    step = 0
    while True:
        if (args.minutes and time.time() - started > args.minutes * 60) or (not args.minutes and step >= args.steps):
            break
        if step and step % args.snapshot_every == 0:
            league.snapshot(net)
        obs, mask = env.observe()
        obs, legal = obs.clone(), legal_bits(mask)
        if prev is not None:  # last decision's transitions end here
            p_obs, p_legal, p_act, p_rew, p_done, keep = prev
            replay.add(p_obs[keep], p_legal[keep], p_act[keep], p_rew[keep], p_done[keep], obs[keep], legal[keep])

        with torch.no_grad():
            act = net(obs.view(B * 2, -1), legal.view(B * 2, -1))[0].sample().view(B, 2)
            acts = act.clone()
            foe = 1 - league.side
            for k, s in league.groups():
                f = foe[s]
                if k < args.pool:
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
        prev = (obs, legal, act, rew, both(done).float(), mine & ~both(trunc))
        league.draw(done | trunc)
        assert not env.err.any(), 'an illegal action reached the engine'

        if replay.n >= args.start:
            for _ in range(args.updates):
                o, lg, a, rw, d, nx, nl = replay.sample(args.minibatch, gen)
                alpha = log_alpha.exp().detach()
                with torch.no_grad():
                    v_next, _, _ = soft_v(net, nx, nl, target.qs(nx), alpha)
                    y = rw + args.gamma * (1 - d) * v_next
                q1, q2 = net.qs(o)
                critic_loss = sum(nn.functional.mse_loss(q.gather(1, a.unsqueeze(1)).squeeze(1), y) for q in (q1, q2))
                critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_([*net.q1.parameters(), *net.q2.parameters()], 10.0)
                critic_opt.step()

                logp = torch.log_softmax(net.logits(o, lg), -1)
                p = logp.exp()
                actor_loss = (p * (alpha * logp - torch.min(q1, q2).detach())).sum(-1).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(net.actor.parameters(), 10.0)
                actor_opt.step()

                entropy = -(p * logp).sum(-1).detach()
                target_h = args.entropy_ratio * lg.sum(-1).float().log()
                alpha_loss = (log_alpha * (entropy - target_h)).mean()
                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()
                entropy = entropy.mean()

                with torch.no_grad():
                    for t_, s_ in zip([*target.q1.parameters(), *target.q2.parameters()],
                                      [*net.q1.parameters(), *net.q2.parameters()]):
                        t_.lerp_(s_, args.tau)
                updates += 1
        step += 1

        if step % args.log_every == 0:
            assert torch.isfinite(critic_loss) and torch.isfinite(actor_loss), 'a non-finite loss'
            sps = args.log_every * B / (time.time() - t0)
            stats = {'train/critic_loss': critic_loss.item(), 'train/actor_loss': actor_loss.item(),
                     'train/entropy': float(entropy), 'train/alpha': log_alpha.exp().item(), 'train/updates': updates,
                     'train/decisions_per_s': sps, **{f'win/{n}': float(x) for n, x in zip(NAMES, league.base_wr)}}
            if w:
                for k, v in stats.items():
                    w.add_scalar(k, v, step)
            wr = ' '.join(f'{name[:6]} {float(x):.2f}' for name, x in zip(NAMES, league.base_wr))
            print(f'step {step:6d}  buffer {replay.n:8d}  updates {updates:6d}  critic {critic_loss.item():.4f}  '
                  f'actor {actor_loss.item():+.4f}  entropy {float(entropy):.3f}  alpha {log_alpha.exp().item():.4f}  '
                  f'win vs {wr}  {sps:,.0f} decisions/s', flush=True)
            t0 = time.time()
        if evaluator and step % args.eval_every == 0:
            net.eval()
            row = {'step': step, 'minutes': round((time.time() - started) / 60, 2),
                   'maxdamage': evaluator.score(net, 'maxdamage')}
            net.train()
            curve.append(row)
            print('eval ' + '  '.join(f'{k} {v}' for k, v in row.items()), flush=True)
            if w:
                w.add_scalar('eval/maxdamage', row['maxdamage'], step)
                w.add_scalar('eval/minutes', row['minutes'], step)
            if args.save:
                stem = pathlib.Path(args.save)
                fileio.write_text(stem.with_name(stem.stem + '_curve.jsonl'), ''.join(json.dumps(x) + '\n' for x in curve))
                discrete_sac.save(net, args.save)
    if w:
        w.close()
    if args.save:
        discrete_sac.save(net, args.save)
        print(f'saved {args.save}')


if __name__ == '__main__':
    main()
