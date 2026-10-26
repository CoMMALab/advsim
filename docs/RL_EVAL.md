# RL evaluation harness

Author: Sai

`tools/policy_eval.py` scores RL policies against each other and against the
scripted baselines. It uses the evaluator the league trainer uses
(`tools/league.py`, `Evaluator`): mirrored couples on fixed deals, so both
players get the same teams and the same random draws, with the sides swapped.
A score is the win rate, with a tie (a game still going after 1000 decisions)
counting as half a win. 0.5 means the two policies are even.

## Running it

```sh
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv   # pick a free GPU
CUDA_VISIBLE_DEVICES=1 uv run --group rl python tools/policy_eval.py \
    ppo:dmg46,ppo:long,dqn --against random,maxdamage --games 4096
```

```
ppo:dmg46 at temperature 1: random 1.000  maxdamage 0.980
ppo:long at temperature 1: random ...  maxdamage ...
dqn: skipped (DQN and Double DQN: load)
```

The harness needs `advsim build` and, for the `ppo` policies, the networks in
`artifacts/` (see the README).

| Flag | Default | Meaning |
|---|---|---|
| *(positional)* | | Policies to score, comma-separated |
| `--temps` | `1` | Temperatures for network policies, comma-separated. 0 takes the top action. Baselines play once |
| `--against` | `maxdamage` | Opponents, comma-separated. They play at temperature 1 |
| `--games` | 4096 | Games per policy, opponent and temperature. Use an even number, since games come in couples |
| `--seed` | 11 | The deals. Keep it fixed when comparing algorithms |
| `--device` | `cuda:0` | The torch/Warp device |

## Policy names

| Name | What it is |
|---|---|
| `random`, `emerald`, `maxdamage`, `status`, `switchaverse` | The scripted baselines, run in the engine's kernel |
| `ppo[:tag]` | A PPO network, loaded from `artifacts/ppo_<tag>.npz`. With no tag it loads `ppo_baseline.npz`. A tag ending in `.npz` is used as the path. A bare name that isn't registered, like `dmg46`, is also a PPO tag |
| `dqn`, `double_dqn`, `drqn`, `discrete_sac` | Placeholders that raise `NotImplementedError` |

## The policy contract

A network policy is a `torch.nn.Module` whose forward pass looks like this:

```python
dist, value = net(obs, legal)
# obs:   int16 [n, OBS_DIM], from the engine's obs kernel
# legal: float [n, 12], 1 for each legal action (net.legal_bits(mask))
# dist:  torch.distributions.Categorical over the 12 actions, zero mass on illegal ones
```

`players.net_act` samples from `dist`. The harness doesn't use `value`, so any
placeholder works there.

A value-based algorithm (DQN, Double DQN, DRQN) has to return its greedy
action as a `Categorical`. For example, mask the illegal Q-values to `-inf`
and pass `logits=q / tau` with a small `tau`, or one-hot logits. Discrete SAC
already has a `Categorical` actor.

A recurrent policy (DRQN) needs per-game hidden state across decisions. The
evaluator calls the network only for live games that have a real choice, so
it doesn't pass game indices. Supporting recurrence means extending
`Evaluator.score` to pass the game index `g` along. Expect that when DRQN is
implemented.

## Where the code goes

| File | What goes in it |
|---|---|
| `advsim/rl/<algo>.py` | The network (`Net`), `save` and `load`. Importable, no CLI |
| `tools/train/train_<algo>.py` | The training loop. Reuse `Gen3Env`, `players` and `league.Evaluator` |
| `artifacts/<algo>_<tag>.npz` | Checkpoints, written through `advsim/fileio.py` |
| `tests/test_<algo>.py` | A few iterations run, and a save/load round trip matches |

Placeholders with `TODO:` markers exist for `dqn` (which covers Double DQN through
`train_dqn.py --double`, since only the target differs), `drqn` and `discrete_sac`. The
harness already loads `<algo>[:tag]` through `advsim.rl.<algo>.load`, so a
policy is ready to score once `Net.forward` and `load` are implemented.

## Adding an algorithm

Add the files above and register it in `tools/policy_eval.py`, or register it from your own module:

```python
import policy_eval
from advsim import net as net_

policy_eval.register('dqn', lambda device, tag: MyDQN.load(f'artifacts/dqn_{tag}.npz', device))
```

The loader gets `(device, tag)`, where `tag` is the part after the colon (or
`None`). It returns a network that meets the contract above, or a baseline
name. Save weights with `advsim.net.save` or `advsim.fileio.write_npz`. File
I/O goes through `advsim/fileio.py` (CONTRIBUTING.md).

## Comparing algorithms fairly

- Use the same `--seed` and `--games` for every algorithm. Then each pairing
  plays the same deals.
- At 4096 games, the standard error of a score near 0.5 is about 0.008.
  Treat differences under about 0.02 as noise, or raise `--games`.
- For a single rating across many players, use the Elo bracket
  (`tools/elo.py`). It only knows its own player names, so a new algorithm
  needs a branch in `elo.play` next to the `ppo` one.
- Record results in `docs/RESULTS.md`.

## Checks

```sh
uv run --group dev --group rl pytest -q tests/test_policy_eval.py
```
