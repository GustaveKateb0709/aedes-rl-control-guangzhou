# Code and data: deep reinforcement learning for weekly mosquito-control allocation under an operational capacity ceiling

This package reproduces every number, table and figure in the manuscript.

Version 2.0.0 extends the original submission with a weekly operational-capacity
ceiling, additional algorithms, cross-year validation and a spatial-structure
analysis. See "What is new in 2.0.0" below.

## Layout

```
code/
  mosquito_env.py       Gymnasium environment: MDP formulation, intervention
                        effect processes, reward. Supports the weekly capacity
                        ceiling (C_max), nested action tables
                        (action_table_size, subset_core), greedy budget
                        projection, adjacency-augmented observations, and a
                        flattened Discrete action space.
  train_ppo.py          PPO / A2C training driver (Stable-Baselines3).
                        Flags: --C-max, --action-table-size, --subset-core,
                        --budget-projection, --augment-adjacency.
  baselines.py          fixed rules: no intervention / threshold / calendar
  baselines_extra.py    grid-tuned rules, one-step-lookahead controller
                        (MPC-1) with exhaustive feasible-action enumeration,
                        SB3 policy loading helpers
  capacity_sweep.py     weekly capacity sweep (C_max x algorithm), budget
                        utilisation and infeasibility accounting
  loyo.py               leave-one-year-out cross-validation (8 folds)
  spatial_ablation.py   exact decomposability checks + adjacency-feature
                        ablation (--mode decompose | train-adjac | eval-adjac)
  k_confound.py         nested-action control: isolates action-table size from
                        the ceiling (--mode train | eval | tables | all)
  evaluate.py           paired counterfactual evaluation
  sensitivity.py        alpha sweep / parameter perturbation / action ablation
  tune_rules.py         grid-search tuning of the static rules on training years
  ppo_return.py         PPO episode return under the tuning protocol
  eval_transmission.py, make_transmission_figure.py   transmission-potential layer
  make_figures.py       original figures (Figure1..Figure8)
  make_figures_v2.py    manuscript figures (Figure1..Figure13, TIFF 600 dpi LZW + PDF)
  run_all.sh, run_battery_resumable.sh
models/                 trained policies (*.zip) + VecNormalize stats (*.pkl).
                        Large (~600 MB); the published archive is on Zenodo
                        (see "Data availability").
results/                every evaluation output, including
  panel_data.csv        10 grids x 416 weeks (2015-2022): ERA5 weather, PFI,
                        urbanisation scale, simulated baseline BI
  capacity_summary.csv  the weekly capacity sweep, per arm and per ceiling
  capacity_budget_use_summary.csv   how often the ceiling binds
  loyo_fold_summary.csv, loyo_stability.csv   leave-one-year-out results
  spatial_decomposability.csv, spatial_adjacency_summary.csv
  kconfound_*.csv       nested-action control (action-space size vs ceiling)
  alpha_sweep.csv, robustness_perturbation.csv, action_ablation.csv
  trajectory_*.csv, transmission_weekly.npy
  monitor/              training logs behind the learning-curve figure
requirements.txt
```

## What is new in 2.0.0

1. **Weekly operational-capacity ceiling.** The cost of the joint action in a
   week, summed over the ten zones, may not exceed `C_max` (source reduction
   costs 1 unit, space spraying 3). The ceiling is imposed at sampling time by
   pre-enumerating all `3^10 = 59,049` action vectors and restricting the policy
   to the feasible subset, so constraint-violating weeks are never emitted. The
   feasible-set size is `K = 56, 486, 2,193, 6,498, 25,149` for
   `C_max = 2, 4, 6, 8, 12`, and `59,049` when the ceiling is removed.
2. **Capacity sweep** over `C_max in {2, 4, 6, 8, 12, none}` for PPO, A2C,
   Double DQN, the one-step-lookahead controller and the static rules.
3. **Ten training seeds** for the main configurations (was three).
4. **Leave-one-year-out cross-validation**, eight folds (2015-2022).
5. **Decomposability.** Without a ceiling the joint problem separates exactly
   into ten independent single-zone problems; this is verified numerically and
   confirmed by a negative adjacency-feature ablation.
6. **Nested-action control.** A control arm removes the ceiling and varies only
   the size of the flattened action table, so that the effect of the ceiling can
   be separated from the effect of action-space size.

## Quick start

```bash
pip install -r requirements.txt
cd code

# main paired evaluation on the held-out year
python evaluate.py --reps 10

# weekly capacity sweep (uses shipped models where available)
python capacity_sweep.py --mode eval

# leave-one-year-out cross-validation
python loyo.py --mode eval

# exact decomposability checks and the adjacency ablation
python spatial_ablation.py --mode decompose
python spatial_ablation.py --mode eval-adjac

# nested-action control: evaluate what is on disk, no retraining
python k_confound.py --mode eval

# manuscript figures (TIFF 600 dpi LZW + PDF) into ../figures_v2/
python make_figures_v2.py
```

Retraining is CPU-bound and slow for the largest action tables:

```bash
python train_ppo.py --algo ppo --seed 0 --steps 400000 --name ppo_cinf_s0
python train_ppo.py --algo ppo --seed 0 --steps 400000 --name ppo_c2_s0 --C-max 2
bash run_all.sh
```

## Conventions

- One environment step = one calendar week; one episode = one calendar year,
  simulated jointly for all ten zones.
- Training years 2015-2021; 2022 is held out.
- Action codes: 0 = no intervention, 1 = source reduction, 2 = space spraying.
- Costs are relative units: 1 per zone-week of source reduction, 3 per
  zone-week of space spraying.
- Random seeds are fixed throughout. Training seeds are 0-9 for the main
  configurations; evaluation noise seeds are 10000-10009, paired across arms.
- `C_max = none` uses a factorized per-zone action head (`MultiDiscrete`), which
  is the original formulation; every capacity-constrained arm uses a flattened
  `Discrete(K)` table, because a global ceiling makes the feasible set a
  non-product set that cannot be expressed per zone.

## Scope of the shipped evaluations

Two control arms defined in `k_confound.py` are **not** part of the reported
results and are not evaluated here: the flattened full joint space
(`nest59049`, 3 seeds) and the full space with greedy budget projection
(`c2proj`, 3 seeds) were trained but not evaluated before the manuscript was
finalised. The reported action-space analysis rests on the nested tables
`K = 56, 486, 6,498` at `C_max = none`, together with the factorized
`MultiDiscrete` reference and the uniform-random 56-action negative control.
The script defaults will retrain all arms if the manuscript numbers are to be
extended.

## Data availability

Code and data: https://github.com/GustaveKateb0709/aedes-rl-control-guangzhou
Archived with a persistent identifier: https://doi.org/10.5281/zenodo.22158605

ERA5 reanalysis data were obtained through the Open-Meteo archive
(https://open-meteo.com/).
