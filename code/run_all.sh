#!/bin/bash
# Full experiment battery (sequential). Run from this directory with a
# Python environment that satisfies ../requirements.txt.
PY=${PYTHON:-python}
cd "$(dirname "$0")"
set -x
# 1. main models: alpha=0.1, 3 seeds, 400k steps
for S in 0 1 2; do
  $PY train_ppo.py --alpha 0.1 --seed $S --steps 400000
done
# 2. alpha sweep (seed 0), 300k steps
for A in 0.0 0.05 0.2 0.5; do
  $PY train_ppo.py --alpha $A --seed 0 --steps 300000
done
# 3. action ablation (seed 0), 300k steps
$PY train_ppo.py --alpha 0.1 --seed 0 --steps 300000 --action-set 0 1
$PY train_ppo.py --alpha 0.1 --seed 0 --steps 300000 --action-set 0 2
# 4. evaluation + sensitivity tables
$PY evaluate.py --reps 10
$PY sensitivity.py --mode robust
$PY sensitivity.py --mode alpha --alphas 0.0 0.05 0.1 0.2 0.5
$PY sensitivity.py --mode ablation
$PY make_figures.py
echo "BATTERY_DONE"
