#!/usr/bin/env bash
# Train PPO, A2C and REINFORCE for the same wall-clock time (default lr, same seed and recipe), then
# rate all three in an Elo bracket against the five scripted baselines.
#
#   tools/train/compare_ppo_a2c_reinforce.sh
#   MINUTES=30 GAMES=2048 SEED=0 tools/train/compare_ppo_a2c_reinforce.sh
#
# Writes artifacts/ppo_{ppo,a2c,reinforce}30.npz (tags ppo30, a2c30 and reinforce30; the trainer's
# checkpoints are `ppo_<tag>.npz` whatever the algorithm), logs in logs/, and the bracket in
# artifacts/elo_ppo_a2c_reinforce/ (games.jsonl and summary.json).
#
# REINFORCE learns from whole games (no critic, so nothing bootstraps), so it gets a rollout as long
# as a game: --steps at --max-steps (400) instead of the default 64. PPO and A2C keep the default.
set -euo pipefail
cd "$(dirname "$0")/../.."

MINUTES=${MINUTES:-30}
GAMES=${GAMES:-2048}   # per pair, even
SEED=${SEED:-0}
DEVICE=${DEVICE:-cuda:0}
RUN="uv run --group rl python"
ELO_DIR=artifacts/elo_ppo_a2c_reinforce

mkdir -p logs "$ELO_DIR"

for algo in ppo a2c reinforce; do
    extra=()
    [ "$algo" = reinforce ] && extra=(--steps 400)
    echo "=== training $algo for $MINUTES minutes ==="
    $RUN tools/train/train_league.py --algo "$algo" --anneal --minutes "$MINUTES" --seed "$SEED" \
        --device "$DEVICE" "${extra[@]}" --save "artifacts/ppo_${algo}30.npz" 2>&1 | tee "logs/train_${algo}30.log"
done

echo "=== Elo bracket ==="
# `ppo-t25:TAG` plays artifacts/ppo_TAG.npz at temperature 0.25, how the repo rates its networks.
$RUN tools/elo.py --games "$GAMES" --device "$DEVICE" --out "$ELO_DIR/games.jsonl" \
    --players random,emerald,maxdamage,status,switchaverse,ppo-t25:ppo30,ppo-t25:a2c30,ppo-t25:reinforce30 \
    2>&1 | tee logs/elo_ppo_a2c_reinforce.log
