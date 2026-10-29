"""The update rule of the league trainer: what to do with a rollout once it is collected.

`train_league.py` plays the league and builds the rollout (observations, legal
masks, actions, the old log-probabilities, normalized advantages and returns,
all for the learner's own real decisions); an `Algo` turns it into optimizer
steps. The rollout, the opponents, the network and the evaluation do not depend
on the algorithm, so a comparison between two of them changes nothing else.

    update(net, opt, batch, adv, ret, amp) -> (loss, entropy), of the last step

`batch` holds `obs`, `legal`, `act`, `logp` and (with `--opp-action`) `foe_act`,
one row per decision; `adv` and `ret` are the rows' advantages and returns.
`amp` is the trainer's autocast context factory. A new algorithm subclasses
`Algo`, defines `policy_loss` and `update`, and is added to `ALGOS`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from league_targets import foe_action_loss


def gae(rew, val, done, trunc, last_val, gamma: float, lam: float):
    """Advantages and returns over [T, B, 2] streams, one per side of a slot:
    a game end cuts both, and a truncation bootstraps from the value there."""
    T = rew.shape[0]
    adv = torch.zeros_like(rew)
    running = torch.zeros_like(last_val)
    next_val = last_val
    for t in reversed(range(T)):
        end = (done[t] | trunc[t]).unsqueeze(-1)
        nv = torch.where(end, torch.zeros_like(next_val), next_val)
        nv = torch.where(trunc[t].unsqueeze(-1), val[t], nv)
        delta = rew[t] + gamma * nv - val[t]
        running = delta + gamma * lam * torch.where(end, torch.zeros_like(running), running)
        adv[t] = running
        next_val = val[t]
    return adv, adv + val


class Algo:
    """An actor-critic update: the value loss, the entropy bonus and the
    optional opponent-action loss are shared, the policy loss is the subclass's."""

    name = ''

    def __init__(self, args) -> None:
        self.args = args

    def policy_loss(self, dist, batch: dict, adv: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """The loss on the rows `i` for their new policy `dist`."""
        raise NotImplementedError

    def update(self, net, opt, batch: dict, adv: torch.Tensor, ret: torch.Tensor, amp):
        raise NotImplementedError

    def step(self, net, opt, batch: dict, adv: torch.Tensor, ret: torch.Tensor, idx: torch.Tensor, amp):
        """One optimizer step on the rows `idx`, the backward in `--micro` chunks
        (gradient accumulation: the same step)."""
        args = self.args
        micro = args.micro or len(idx)
        opt.zero_grad()
        for c0 in range(0, len(idx), micro):
            i = idx[c0:c0 + micro]
            with amp():
                if args.opp_action:
                    dist, v, extras = net(batch['obs'][i], batch['legal'][i], opp_action=True)
                else:
                    dist, v = net(batch['obs'][i], batch['legal'][i])
            pg = self.policy_loss(dist, batch, adv, i)
            vf = ((v - ret[i]) ** 2).mean()
            if args.opp_action:
                vf = vf + args.opp_action * foe_action_loss(extras['opp_action'], batch['foe_act'][i])
            ent = dist.entropy().mean()
            loss = pg + 0.5 * vf - args.ent * ent
            (loss * (len(i) / len(idx))).backward()
        nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        opt.step()
        return loss, ent


class PPO(Algo):
    """Clipped-ratio PPO: `--epochs` passes over the rollout in minibatches of `--minibatch` rows."""

    name = 'ppo'

    def policy_loss(self, dist, batch, adv, i):
        ratio = (dist.log_prob(batch['act'][i]) - batch['logp'][i]).exp()
        return -torch.min(ratio * adv[i], ratio.clamp(0.8, 1.2) * adv[i]).mean()

    def update(self, net, opt, batch, adv, ret, amp):
        n = adv.shape[0]
        for _ in range(self.args.epochs):
            perm = torch.randperm(n, device=adv.device)
            for s0 in range(0, n, self.args.minibatch):
                loss, ent = self.step(net, opt, batch, adv, ret, perm[s0:s0 + self.args.minibatch], amp)
        return loss, ent


class A2C(Algo):
    """Advantage actor-critic: one step on the whole rollout, on the policy that
    collected it. No ratio, no clip and no epochs, so nothing holds a step back
    from moving the policy far: the learning rate matters more than for PPO.
    `--epochs` and `--minibatch` do not apply; `--micro` still sets the chunks."""

    name = 'a2c'

    def policy_loss(self, dist, batch, adv, i):
        return -(dist.log_prob(batch['act'][i]) * adv[i]).mean()

    def update(self, net, opt, batch, adv, ret, amp):
        return self.step(net, opt, batch, adv, ret, torch.arange(adv.shape[0], device=adv.device), amp)


ALGOS = {cls.name: cls for cls in (PPO, A2C)}
