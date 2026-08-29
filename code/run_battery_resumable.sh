#!/bin/bash
# Resumable battery: skips trainings whose model file already exists.
PY=${PYTHON:-python}
cd "$(dirname "$0")"
train () {  # train <alpha> <seed> <steps> <action-set...>
  local A=$1 S=$2 N=$3; shift 3
  local tag="ppo_a${A}_s${S}_$(echo $@ | tr -d ' ' | sed 's/acts/acts/')"
  local f="../models/ppo_a${A}_s${S}_acts$(echo $@ | tr -d ' ').zip"
  if [ -f "$f" ]; then echo "[skip] $f exists"; return; fi
  echo "[train] alpha=$A seed=$S steps=$N acts=$@ $(date '+%H:%M:%S')"
  $PY train_ppo.py --alpha $A --seed $S --steps $N --action-set "$@"
}
echo "=== battery start $(date) ==="
train 0.1 0 400000 0 1 2
train 0.1 1 400000 0 1 2
train 0.1 2 400000 0 1 2
for A in 0.0 0.05 0.2 0.5; do train $A 0 300000 0 1 2; done
train 0.1 0 300000 0 1
train 0.1 0 300000 0 2
echo "=== trainings done $(date), starting evaluations ==="
$PY evaluate.py --reps 10
$PY sensitivity.py --mode robust
$PY sensitivity.py --mode alpha --alphas 0.0 0.05 0.1 0.2 0.5
$PY sensitivity.py --mode ablation
$PY make_figures.py
echo "BATTERY_DONE $(date)"
