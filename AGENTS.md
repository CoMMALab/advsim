# AGENTS.md

advsim is a GPU battle engine for Showdown [Gen 3] Random Battle, used for RL
training and test-time search. The rules for working here are in
`CONTRIBUTING.md` (hard rules) and `CLAUDE.md`; the design is `docs/SPEC.md`;
measured results go in `docs/RESULTS.md`. This file covers TensorBoard for the
RL policies. The evaluation harness itself is in `docs/RL_EVAL.md`.

## TensorBoard

Every RL run and the Elo bracket log to one directory, `runs/` (gitignored),
one subdirectory per run.

### What writes where

| Source | Run directory | Tags |
|---|---|---|
| `tools/train/train_league.py` (PPO) | `runs/<--save stem>`, or `runs/ppo_<time>` without `--save` | `train/loss`, `train/entropy`, `train/transitions`, `train/decisions_per_s`, `win/<baseline>`, `eval/maxdamage`, `eval/minutes`, `eval/<--eval-ref>` |
| `tools/train/train_dqn.py` (DQN, `--double`, `--dueling`) | `runs/<--save stem>`, or `runs/<algo>_<time>` | `train/loss`, `train/epsilon`, `train/updates`, `train/decisions_per_s`, `win/<baseline>`, `eval/maxdamage`, `eval/minutes` |
| `tools/train/train_pg.py` (`--algo reinforce\|grpo`) | `runs/<--save stem>`, or `runs/<algo>_<time>` | `train/loss`, `train/entropy`, `train/kl`, `train/transitions`, `train/finished`, `train/decisions_per_s`, `win/<baseline>`, `eval/maxdamage`, `eval/minutes` |
| `tools/train/train_discrete_sac.py` | `runs/<--save stem>`, or `runs/discrete_sac_<time>` | `train/critic_loss`, `train/actor_loss`, `train/entropy`, `train/alpha`, `train/updates`, `train/decisions_per_s`, `win/<baseline>`, `eval/maxdamage`, `eval/minutes` |
| `tools/elo.py --tb runs` | `runs/elo` | `elo/<player>` scalars; text `elo/ratings` (sorted, with 95% intervals) and `elo/head_to_head` |
| `tools/tb.py` | `runs/<stem>` per `artifacts/*_curve.jsonl`; `runs/elo` from `artifacts/elo/summary.json` | `eval/*` only, for runs trained before logging existed |

- **The step is decisions per battle slot** for every trainer. One PPO
  iteration is `--steps` decisions (64 by default), one DQN step is one. That
  puts PPO and DQN on the same x-axis. For wall-clock comparisons, use
  `eval/minutes` or TensorBoard's "Relative" axis.
- **`win/*` is not a score to report.** These are running averages against
  opponents that the league picks for being hard. `eval/maxdamage` (fixed teams,
  mirrored couples, eval seed 7) is the curve to compare.
- **The Elo step is the number of games in the fit**, so each `elo.py --add`
  adds a point.
- Trainers log by default. Pass `--tb ''` to turn logging off, which tests and
  throwaway smoke runs should do so they don't litter `runs/`. Pass `--tb DIR`
  to log somewhere else.

### Recipes

```sh
# Train with logging (check GPUs first; pick a free one with CUDA_VISIBLE_DEVICES)
CUDA_VISIBLE_DEVICES=2 uv run --group rl python tools/train/train_dqn.py --dueling --double \
    --minutes 30 --save artifacts/dueling_double_dqn_mlp.npz

# Rate a new checkpoint and update the Elo view
CUDA_VISIBLE_DEVICES=2 uv run --group rl python tools/elo.py --add dueling_double_dqn:mlp \
    --games 256 --tb runs

# Refit the bracket without playing, and log it
uv run --group rl python tools/elo.py --ratings artifacts/elo/games.jsonl --tb runs

# Import curves and the bracket written before TensorBoard logging existed
uv run --group rl python tools/tb.py
```

To redo a run, delete `runs/<name>` first. Importing or logging the same run
twice appends a second copy of every point.

### Viewing

For a person:

```sh
uv run --group rl tensorboard --logdir runs --port 6006 --bind_all
# from a laptop: ssh -L 6006:localhost:6006 <box>, then http://localhost:6006
```

An agent cannot see the web page, so it should read the event files directly:

```python
from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer

m = EventMultiplexer()
m.AddRunsFromDirectory('runs')
m.Reload()
for run in sorted(m.Runs()):
    if 'eval/maxdamage' in m.GetAccumulator(run).Tags()['scalars']:
        pts = m.Scalars(run, 'eval/maxdamage')
        print(run, [(p.step, round(p.value, 3)) for p in pts[-3:]])
```

Run it with `uv run --group rl python`. Text tags such as `elo/ratings` are
under `Tags()['tensors']` as `elo/ratings/text_summary`. For the bracket
itself, `artifacts/elo/summary.json` is simpler to read.

### Process hygiene

To stop a TensorBoard server you started, kill it by PID (`$!` right after
launching it in the background). Never use `pkill -f tensorboard`: the pattern
matches your own shell's command line, so it kills that shell too. If you must
match by name, build the pattern from a variable, as `CLAUDE.md` describes.

### Adding a new algorithm

In its trainer, create `w = tb.writer(args.tb, tb.run_name(args.save, '<algo>'))`
(`tools/tb.py`). Log with the same tag names and the same step unit, decisions
per battle slot, and call `w.close()` at the end. Then the new runs overlay the
existing ones without any other change.
