"""DQN and Double DQN: the network and its acting interface. Training is in tools/train/train_dqn.py.

The contract (docs/RL_EVAL.md): forward(obs int16 [n, OBS_DIM], legal float [n, 12])
-> (Categorical over 12 actions, zero mass on illegal ones; value or None).
"""
from __future__ import annotations

import torch
from torch import nn


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO: layers. Reuse advsim.net.mlp, or advsim.net_tf for a transformer encoder.

    def forward(self, obs: torch.Tensor, legal: torch.Tensor):
        # TODO: Q-values [n, 12], illegal ones masked to -inf; return Categorical(logits=q / tau) with a small tau, and None.
        raise NotImplementedError('DQN and Double DQN: forward')


def save(net: Net, path) -> None:
    # TODO: weights to npz through advsim.fileio.write_npz (see advsim.net.save).
    raise NotImplementedError('DQN and Double DQN: save')


def load(path, device) -> Net:
    # TODO: the inverse of save (see advsim.net.load).
    raise NotImplementedError('DQN and Double DQN: load')
