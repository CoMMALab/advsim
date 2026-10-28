"""DQN and Double DQN: the network and its acting interface. Training is in tools/train/train_dqn.py.

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
    """An MLP over the scaled observation, like advsim.net.Policy, with one Q-value per action."""

    def __init__(self, hidden: int = 512, layers: int = 2, inputs: int = L.OBS_DIM) -> None:
        super().__init__()
        self.body = mlp(inputs, *[hidden] * layers)
        self.head = nn.Linear(hidden, ACTIONS)

    def q(self, obs: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        """Q-values [n, 12], illegal ones at -1e9."""
        x = obs[..., :self.body[0].in_features].float() / 64.0
        return self.head(self.body(x)).float().masked_fill(~legal.bool(), -1e9)

    def forward(self, obs: torch.Tensor, legal: torch.Tensor):
        """The greedy action as a one-hot Categorical, and no value."""
        q = self.q(obs, legal)
        greedy = torch.nn.functional.one_hot(q.argmax(-1), ACTIONS).bool()
        return torch.distributions.Categorical(logits=torch.zeros_like(q).masked_fill(~greedy, -1e9)), None


def save(net: Net, path) -> None:
    fileio.write_npz(pathlib.Path(path), {k: v.detach().float().cpu().numpy() for k, v in net.state_dict().items()})


def load(path, device) -> Net:
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f'{path} missing: train one with tools/train/train_dqn.py --save')
    arrays = fileio.read_npz(path)
    layers = sum(1 for k in arrays if k.startswith('body.') and k.endswith('.weight'))
    net = Net(hidden=arrays['head.weight'].shape[1], layers=layers, inputs=arrays['body.0.weight'].shape[1])
    net.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in arrays.items()})
    return net.to(device).eval()
