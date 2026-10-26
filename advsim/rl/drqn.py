"""DRQN (recurrent Q-learning): the network and its acting interface. Training is in tools/train/train_drqn.py.

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
        # TODO: per-game hidden state across decisions. Evaluator.score must pass game indices (docs/RL_EVAL.md).
        raise NotImplementedError('DRQN (recurrent Q-learning): forward')


def save(net: Net, path) -> None:
    # TODO: weights to npz through advsim.fileio.write_npz (see advsim.net.save).
    raise NotImplementedError('DRQN (recurrent Q-learning): save')


def load(path, device) -> Net:
    # TODO: the inverse of save (see advsim.net.load).
    raise NotImplementedError('DRQN (recurrent Q-learning): load')
