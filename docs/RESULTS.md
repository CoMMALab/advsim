# Results

What has been measured, with the commands that measure it. Dates are when a
result was taken (2026); hardware is named where speed is part of the result.
The design itself is in [SPEC.md](SPEC.md).

## Parity with Showdown

Every check compares the engine against `pokemon-showdown` 0.11.11 itself.

| Check | Scale | Result |
|---|---|---|
| L1 damage, through the engine's own hit path, abilities and items live | 10,000,000 cases | 0 mismatches |
| L2 scenarios (`tests/scenarios/*.json`), scripted draws at named call sites | 29 positions | full canonical hash at every decision point |
| L3 whole battles, random play (`--seed 200001 --seeds 200 --battles 500 --turns 100`) | 100,000 battles, 6,782,378 decision points | 0 divergences |
| L3, six earlier seed ranges | 893,181 decision points | 0 divergences |
| L3, `--policy mix` (random, Emerald, max-damage, status-first, switch-averse) | 100,000 battles, 5,666,760 decision points | 0 divergences |
| Emerald replacement chooser | 2,000 battles, 133,406 decision points | 0 divergences |
| Perft through `fork` (500 positions) | 1,976,061 leaf cases | all match |
| CPU against CUDA (`parity.py devices`) | 2,000 battles, 111,479 decision points | 0 hashes differ |
| Live observation converter against the `obs` kernel | 276,366 observations (version 4) | 0 differences |

Each decision point is judged in order: draw counts, then the full canonical
hash over every field Showdown can speak to, then the legal-action mask. A
clean range says nothing about the next one: every new seed range found
something the last did not, until the counts above.

```
uv run python tools/parity.py sweep --policy mix --seed 2000001 --seeds 200 --battles 500 --turns 100 --jobs 20
uv run python tools/parity.py damage --seeds 200 --battles 50000 --jobs 20
uv run python tools/parity.py perft --seed 7000001 --seeds 500 --battles 3000 --jobs 16
```

## Speed

- **Gate (before the engine was written).** The dense step at batch 16K on an
  RTX 4070: Warp 10.7x JAX, within 5% of hand-written CUDA, all three
  bit-identical (`bench/gate.py`).
- **PPO smoke run.** `examples/ppo_smoke.py` takes p1 from 0.49 to 0.925
  against random play in 40 iterations, at about 745K decisions/s on the 4070.
- **League training.** The 2x512 MLP: about 430K decisions/s; 3x1024: 135K. On
  rented RTX 4090 hosts the loop is CPU-bound (Python at 100%, GPU at 23%);
  removing the rollout's waits on the GPU took it from 142K to 172K
  decisions/s, and two runs can share one GPU at almost no cost to either.
- **Build.** Turning off Warp's adjoints (`enable_backward=False`) took the
  cold CUDA build from 30m40s to 2m45s. Keeping one kernel that calls the
  step (`step_idx`) took the step module from over 36 minutes and ~43 GB of
  host memory to about 7.

## The Elo bracket

`tools/elo.py` plays every pair 2,048 games in mirrored couples (both games
deal the same teams and draws, with the players swapped) and fits a
Bradley-Terry model, `random` anchored at 1000. `--add NAME` rates a new
player against the existing field. Ranges in square brackets are 95%
intervals from resampling the couples. Ratings below are from the current
fit of the whole field; each refit moves them by a few points.

Current field (38 players, 1,437,696 games), top and baselines:

| Player | Elo |
|---|---|
| `search-64-pnet-t25-mix-g25-prune02:dmg46` | 2275 [2268, 2285] |
| `search-64-pnet-t25-mix-g25-prune02:tf96cdmglr` | 2256 |
| `ppo-t25:dmg46` | 2240 |
| `search-64-pnet-t25-mix-g25-prune02:tf96cnight` | 2217 |
| `ppo-t25:tf96cdmglr` | 2215 |
| `ppo-t25:tf96cnight` | 2170 |
| `ppo-t25:night` (PolicyV2) | 2110 |
| `ppo-t25:long` (PolicyV2) | 2009 |
| `ppo:league` (MLP) | 1611 |
| `maxdamage` | 1595 |
| `search-64` (no network) | 1549 |
| `switchaverse` | 1309 |
| `emerald` | 1002 |
| `random` | 1000 |
| `status` | 758 |

Player names: `ppo:<tag>` is the network `artifacts/ppo_<tag>.npz` sampling
at temperature 1, `ppo-t25:` at 0.25. `search-N` is the device MCTS with N
iterations of 8 lanes; `-net` scores leaves with the network's value head,
`-pnet` also uses its policy as every node's prior, `-t25` plays at
temperature 0.25, `-mix` mixes the root equilibrium with the network's root
policy, `-gNN` and `-pruneNN` set the prior weight and the pruning ratio.

## Networks

### MLP and PolicyV2

- The league trainer (`tools/train/train_league.py`: self-play, past snapshots and
  the five baselines) on the MLP: `ppo:league` 1611 after 12 minutes; a 3x1024
  MLP rates 1607-1637 at 1k-3k iterations, barely better and 3.2x slower.
- **PolicyV2** (embeddings, one shared Pokemon encoder, pointer heads): 45
  minutes of league training rates 1942 against 1637 for the best MLP, and 150
  minutes more 1979 (`long`). Playing at temperature 0.25 adds about 30 for free.
- **Experiments at equal wall-clock** (45 minutes and one hour, scored against
  a reference network; seed noise about 0.005): width 1024 0.427 against 0.5;
  entropy decay 0.455; two attention layers 0.394 (3x slower per iteration);
  a value head over both sides' observations 0.425; belief heads for the foe's
  set 0.461 (within noise); the last move each side used 0.442; two PPO epochs
  instead of four 0.440. None beat the default; at this budget speed per
  iteration wins.
- **Overnight** (780 minutes from `long`, annealed): `night` rates 2110 at
  temperature 0.25.
- **Batch on rented GPUs** (scored against `long`): reward shaping by HP
  difference speeds early learning but is level by 90 minutes; an
  opponent-action auxiliary head is neutral at weight 0.1 and harmful at 0.3;
  a KL pull toward uniform is harmful (0.364 against 0.424); GAE lambda 0.98
  harmful (-0.043); gamma 1.0 no different.
- **Expert iteration** from the search was tried and parked: three
  generations made the network flat (policy entropy 0.44 to 1.63) and the
  result rated 1322. The search's targets were sharp but agreed with the
  network's top action only 41% of the time.

### PolicyTF

A transformer over the observation's tokens (`advsim/net_tf.py`). The
working recipe, which is `train_league.py`'s default: compact (17 tokens),
width 96, 3 layers, reacting to its own prediction of the opponent's action
(`--opp-action 0.1`), with damage ranges as inputs.

- **Four hours from scratch** (two seeds each, against `long`): d96 3-layer
  reacting 0.635 / 0.597, d128 4-layer 0.609 / 0.617, not reacting 0.595 /
  0.590, PolicyV2 0.598 / 0.613. Level at equal time on 4-7x fewer
  iterations: per iteration it learns about 4x faster, and runs 3.8-6x slower.
- **Overnight** (780 minutes): `tfnight` (warm-started) ends at 0.741 against
  `long`, `tf96cnight` (compact, from scratch) at 0.752; they rate 2175 and
  2170.
- **A warm start restarts the learning rate.** At the default 3e-4 with fresh
  Adam state the network first drops well below where it began (0.635 to
  0.557 against `long`, about four hours to recover); at `--lr 1e-4` it stays
  near its start. Warm-start at 1e-4.
- **Damage inputs** (`advsim/damage.py`): about +0.03 per iteration early, but
  nothing measurable at equal time: +4 Elo [-3, +10] at 150 minutes, and over
  23 hours on an A100 from `tf96cnight`, 2215 with them (`tf96cdmglr`) against
  2212 without (`tf96cctl`). Their cost was mostly a compile bug (tables loaded
  inside the forward split the graph in ten); fixed, they cost 1.3% of
  throughput. Kept: free and not worse.
- **Longer training is what pays.** `tf96cnight` (13 hours) 2170, +23 hours
  2215, +46 hours more (`dmg46`, 33,476 iterations) 2240: each doubling pays
  less. `dmg46` is the strongest network alone.
- **Architecture experiments** (two hours from scratch, against `long`;
  baseline 0.585): a speed comparison against the foe active 0.602, SwiGLU,
  type-matchup attention bias, dex features and a set posterior each +0.02 to
  +0.05 per iteration early; all five together 0.641. Trained 23 hours
  (`combo24`), the combination ended where the baseline did (0.753 against
  0.752 for `tf96cnight`; 2193, below the 36-hour networks' 2212-2215), so it
  was removed. The move order, auxiliary value heads and a memory token did
  nothing.

### DQN and Double DQN

`tools/train/train_dqn.py` (`--double` for Double DQN): the 2x512 MLP with a
Q head, in the same league as PPO (self-play, snapshots, baselines), 4,096
battles, epsilon-greedy from 1.0 to 0.05 over 5,000 decisions, 1-step Huber TD,
a 2M-transition replay on the GPU, target sync every 1,000 updates. 30 minutes
each on one RTX A5000 (`dqn_mlp`, `double_dqn_mlp`, 37k and 43k decisions):

- Against `maxdamage` (`tools/train/train_dqn.py`'s curve, 2,048 games, eval seed 7):
  0.08 at 2 minutes, 0.38-0.40 at 7-8, then flat at 0.51-0.54 from 13 minutes on, both
  variants alike.
- `tools/policy_eval.py --games 4096`: DQN random 0.959, maxdamage 0.530,
  switchaverse 0.850; Double DQN 0.962, 0.531, 0.852. No difference.
- A small bracket (256 games per pair; random, maxdamage, `ppo:dmg46`,
  `search-32`, `search-32-net:dmg46`): `double_dqn:mlp` 1567 [1529, 1616], `dqn:mlp`
  1558 [1516, 1609], `maxdamage` 1570. Head to head, Double DQN 0.539 against DQN.
  Both level with `maxdamage` and with the PPO MLP (`ppo:league` 1611 in the
  full field); far below the PolicyV2 and transformer networks.
- **Dueling** (`--dueling`: Q = V + A - mean(A) over all 12 actions), same
  settings, 30 minutes each: `policy_eval` dueling random 0.969, maxdamage 0.528,
  switchaverse 0.849; dueling + double 0.970, 0.547, 0.865. In the same bracket
  refit: `dueling_double_dqn:mlp` 1591 [1555, 1631], `dueling_dqn:mlp` 1574,
  `dqn:mlp` 1570, `double_dqn:mlp` 1568, `maxdamage` 1566. Dueling + double is
  the best of the four by about 20, inside the intervals; head to head it scores
  0.508 against DQN. The architecture does not lift the MLP's ceiling.
- **Rerun** (the current `artifacts/*dqn_mlp.npz`, logged to TensorBoard): the
  same four, one after another on one A5000, 30 minutes each, seed 0. Final
  `eval/maxdamage` 0.53-0.54 for all four. Refit bracket (256 games per pair):
  `dqn:mlp` 1610 [1575, 1661], `maxdamage` 1595, `double_dqn:mlp` 1588,
  `dueling_double_dqn:mlp` 1585, `dueling_dqn:mlp` 1584. The order among the
  four changed from the first run; all sit inside each other's intervals, so
  none is better than the others.

### REINFORCE, GRPO and discrete SAC

Same MLP (2x512), league mix (0.5 self, 0.25 snapshots, 0.25 baselines), seed 0,
30 minutes each, one after another on one A5000.

- **REINFORCE** (`tools/train/train_pg.py --algo reinforce`: batch-mean
  baseline, one full-batch step per batch of finished games): `eval/maxdamage`
  0.14-0.16 from 2.5 minutes to the end, flat. `policy_eval` random 0.861,
  maxdamage 0.161, switchaverse 0.500.
- **GRPO** (`--algo grpo`: groups of 8 forks of one deal, group-normalized
  result, clipped ratio, KL 0.01 to a reference refreshed every 50 iterations):
  0.14 at 5 minutes, 0.31 at 19, then flat at 0.28-0.31. `policy_eval` random
  0.866, maxdamage 0.299, switchaverse 0.647.
- **Discrete SAC** (`tools/train/train_discrete_sac.py`: twin critics, alpha from
  0.005, entropy target 0.5 log(legal actions)): 0.51 at 10 minutes, then flat
  at 0.49-0.52, like DQN. `policy_eval` random 0.969, maxdamage 0.515,
  switchaverse 0.852. It makes about 55k decisions/s against DQN's 90-140k (three
  networks per update).
- Bracket refit with these three (256 games per pair): `dqn:mlp` 1608,
  `maxdamage` 1598, the other DQN variants 1581-1592, `discrete_sac:mlp` 1578
  [1550, 1608], `grpo:mlp` 1400 [1370, 1430], `search-32` 1390,
  `reinforce:mlp` 1295 [1271, 1323].
- Without a critic, GRPO's forked-group baseline roughly doubles REINFORCE's
  score against maxdamage, but both stay far below the value-based methods in
  30 minutes. The off-policy methods (DQN, discrete SAC) all stop at the same
  level as maxdamage.
- **PPO on the same MLP** (`train_league.py --arch mlp`, defaults otherwise,
  30 minutes, `ppo_mlp`): `eval/maxdamage` 0.46 at 6 minutes, 0.54 at the end.
  `policy_eval` random 0.975, maxdamage 0.537, switchaverse 0.871. Elo 1600
  [1564, 1631], inside the DQN/SAC cluster (1580-1609) and level with
  `maxdamage` (1598). At 30 minutes on a 2x512 MLP, PPO, DQN and discrete SAC
  are indistinguishable; the gap to `ppo:dmg46` (2177) comes from the
  transformer, the 80 hours of training, or both.

## Search

The device search (`engine/tree.py`, `search/mcts.py`) is open-loop
simultaneous-move MCTS with decoupled regret matching; every iteration
re-plays its path from a freshly determinized copy of the root.

- With random rollouts and no network, `search-128` rates 1678, above
  `maxdamage`. Scoring leaves with a value head lifts it to 1757-1864, above
  the MLPs it used, but on PolicyV2 it stays below the network playing alone
  (1864 against 1979).
- **Three things make it beat its network**, each measured: the policy as
  every node's prior (`-pnet`), a final move at temperature 0.25 (`-t25`),
  and the final move mixed with the network's root policy (`-mix`). The root
  equilibrium alone plays for a perfect opponent and gives points to
  predictable ones (0.80 against `maxdamage` where the network scores 0.93);
  the mix takes them back (0.95).
- **Knob sweep** (67 runs on `night`): prior weight 0.25 beats 0.10 by about
  7; pruning actions under 2% of the node's best prior adds about 5;
  iterations stop paying at 32; depth 3-6 ties, 8-10 loses; a fading prior,
  a policy rollout and visit counts as the final move were no better and
  were removed.
- **On the transformers** the tuned search adds 35-47 over its network:
  2217 on `tf96cnight`, 2256 on `tf96cdmglr`, 2275 on `dmg46`, the top of the
  field.

Rating a search player takes 40 minutes to 3 hours; a transformer under
search needs `--capacity 4096` on a 12 GB card.
