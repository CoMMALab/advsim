"""Discrete SAC: the network and its acting interface. Training is in tools/train/train_discrete_sac.py.

The contract (docs/RL_EVAL.md): forward(obs int16 [n, OBS_DIM], legal bool [n, 12])
-> (Categorical over 12 actions, zero mass on illegal ones; value or None).
"""
from __future__ import annotations

import pathlib

import numpy as np
import torch
from torch import nn

from advsim import fileio
from advsim.engine import obs_layout as L
from advsim.net import ACTIONS, mlp


class Net(nn.Module):
    """A categorical actor and twin Q critics, each an MLP over the scaled observation like
    advsim.net.Policy. Acting uses the actor alone; the critics are for training."""

    def __init__(self, hidden: int = 512, layers: int = 2, inputs: int = L.OBS_DIM) -> None:
        super().__init__()
        self.actor = nn.Sequential(mlp(inputs, *[hidden] * layers), nn.Linear(hidden, ACTIONS))
        self.q1 = nn.Sequential(mlp(inputs, *[hidden] * layers), nn.Linear(hidden, ACTIONS))
        self.q2 = nn.Sequential(mlp(inputs, *[hidden] * layers), nn.Linear(hidden, ACTIONS))

    def inputs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[..., :self.actor[0][0].in_features].float() / 64.0

    def logits(self, obs: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        return self.actor(self.inputs(obs)).float().masked_fill(~legal.bool(), -1e9)

    def qs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.inputs(obs)
        return self.q1(x).float(), self.q2(x).float()

    def forward(self, obs: torch.Tensor, legal: torch.Tensor):
        return torch.distributions.Categorical(logits=self.logits(obs, legal)), None


def save(net: Net, path) -> None:
    fileio.write_npz(pathlib.Path(path), {k: v.detach().float().cpu().numpy() for k, v in net.state_dict().items()})


def load(path, device) -> Net:
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f'{path} missing: train one with tools/train/train_discrete_sac.py --save')
    arrays = fileio.read_npz(path)
    layers = sum(1 for k in arrays if k.startswith('actor.0.') and k.endswith('.weight'))
    net = Net(hidden=arrays['actor.1.weight'].shape[1], layers=layers, inputs=arrays['actor.0.0.weight'].shape[1])
    net.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in arrays.items()})
    return net.to(device).eval()
