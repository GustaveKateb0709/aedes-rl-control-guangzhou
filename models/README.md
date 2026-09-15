# Trained policies

**The trained policies are not distributed.** The full set is about 1.4 GB,
dominated by the action-space control arms, whose flattened action head has
59,049 outputs (88 MB per policy). They are regenerated exactly by the
commands below, which use fixed seeds; `code/` and `results/` are what the
repository and the archived release carry.

File naming: `models/<name>.zip` is the policy and `<name>_vecnorm.pkl` its
observation-normalisation statistics.

Approximate cost per arm, measured on one 8-core machine with three runs in
parallel: ~200-380 s at K = 56, ~210 s at K = 486, ~400 s at K = 2,193,
~930 s at K = 6,498, ~3,500 s at K = 25,149, and ~7,700-8,000 s for the two
control arms at K = 59,049.

```bash
cd code

# --- main capacity sweep: PPO at every ceiling, 10 seeds each --------------
python train_ppo.py --algo ppo --seed 0 --steps 400000 --name ppo_c2_s0  --C-max 2
python train_ppo.py --algo ppo --seed 0 --steps 400000 --name ppo_c4_s0  --C-max 4
python train_ppo.py --algo ppo --seed 0 --steps 400000 --name ppo_c6_s0  --C-max 6
python train_ppo.py --algo ppo --seed 0 --steps 400000 --name ppo_c8_s0  --C-max 8
python train_ppo.py --algo ppo --seed 0 --steps 400000 --name ppo_c12_s0 --C-max 12
# no ceiling, factorized per-zone head (the original formulation)
python train_ppo.py --algo ppo --seed 0 --steps 400000 --name ppo_cinf_s0
# repeat each with --seed 1 ... 9; the second and third algorithms use
# --algo a2c and --algo dqn with three seeds each

# --- action-space control arms (nested tables, full space, projection) -----
python k_confound.py --mode train --workers 3

# --- leave-one-year-out folds at C_max = 2 ---------------------------------
python loyo.py --mode train --workers 3

# --- adjacency-feature ablation -------------------------------------------
python spatial_ablation.py --mode train-adjac --workers 3

# --- cost-weight sweep (alpha) and the original sensitivity runs -----------
# see run_battery_resumable.sh
```

After training, the evaluation steps in the top-level README reproduce every
number, table and figure in the manuscript.
