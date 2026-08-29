# Code and data: deep reinforcement learning for proactive mosquito control

This package reproduces every number, table and figure in the manuscript.

## Layout

```
code/
  mosquito_env.py   MosquitoEnv: Gymnasium environment (MDP formulation,
                    intervention effect processes, reward)
  train_ppo.py      PPO training driver (Stable-Baselines3)
  baselines.py      rule-based arms: no intervention / threshold / calendar
  evaluate.py       paired counterfactual evaluation on the held-out 2022 year
  sensitivity.py    alpha sweep / parameter perturbation / action ablation
  tune_rules.py     tuned-rule fairness analysis: grid-search-tuned
                    threshold/calendar rules vs fixed rules (Table 5)
  eval_transmission.py  transmission-potential computation (Table 6 / Figure 9)
  make_transmission_figure.py  Figure 9
  ppo_return.py     PPO 2022 episode return under the exact tune_rules
                    protocol (paired arms in Table 5)
  make_figures.py   publication figures (Figure1..Figure8, PNG 300 dpi + PDF)
  run_all.sh        full battery, sequential
  run_battery_resumable.sh  resumable battery (skips existing model files)
models/             trained PPO policies (*.zip) + VecNormalize stats (*.pkl);
                    included in this archive; if cloning from git, download
                    them from the Zenodo archive (see repository README)
results/
  panel_data.csv    10 grids x 416 weeks (2015-2022): ERA5 weather, PFI,
                    urbanisation scale, simulated baseline BI
  counterfactual_2022.csv / counterfactual_summary.csv
  alpha_sweep.csv / robustness_perturbation.csv / action_ablation.csv
  rule_tuning_train.csv / rule_tuned_2022.csv
                    tuned-rule fairness analysis outputs (Table 5)
  ppo_return_2022.csv   PPO held-out 2022 episode return (from ppo_return.py)
  trajectory_*.csv  week-by-week 2022 trajectories per arm (noise rep 0)
  transmission_weekly.npy   weekly transmission-potential index (Figure 9)
  monitor/          PPO training logs (episode rewards) behind Figure 2
requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
# models/ are included in this archive; if cloned from git, download them from
# the Zenodo archive (see repository README).
cd code
python evaluate.py --reps 10          # Table 1 + trajectories (uses shipped models)
python sensitivity.py --mode robust   # Table 3 (no retraining needed)
python tune_rules.py                  # Table 5, tuned-rule arms (no models needed)
python ppo_return.py                  # Table 5, PPO episode return (needs models/)
python make_figures.py                # Figure1..Figure8
python eval_transmission.py           # Table 6 / Figure 9 data
python make_transmission_figure.py    # Figure9
```

Retraining from scratch (CPU, hours per run):

```bash
python train_ppo.py --alpha 0.1 --seed 0 --steps 400000   # main configuration
bash run_all.sh                                           # everything
```

The environment locates `results/panel_data.csv` automatically relative to
`code/`; set `MOSQ_DATA_DIR` to override. Random seeds are fixed throughout:
training seeds 0-2, evaluation noise seeds 10000-10009 (paired across arms).

## Conventions

- One environment step = one calendar week; one episode = one calendar year,
  simulated jointly for all ten grids.
- Training years 2015-2021; 2022 is held out entirely for evaluation.
- Action codes: 0 = no intervention, 1 = source reduction, 2 = space spraying.
- Costs are relative units: 1 per grid-week of source reduction, 3 per
  grid-week of space spraying.
