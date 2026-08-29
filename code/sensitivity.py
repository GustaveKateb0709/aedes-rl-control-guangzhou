#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sensitivity / robustness / ablation drivers.

  --mode alpha      sweep reward cost-weight alpha, retrain PPO per value
  --mode robust     perturb intervention effect parameters and re-evaluate
                    the already-trained main policy WITHOUT retraining
  --mode ablation   retrain with restricted action sets {0,1} / {0,2}
"""
import os, argparse
import numpy as np
import pandas as pd

from mosquito_env import MosquitoEnv, INTERVENTION_PARAMS
from evaluate import evaluate_arm, ppo_policy_factory, EVAL_YEAR, RES, MODELS
from train_ppo import train, tag


def mode_alpha(alphas, steps, seed=0, reps=10):
    rows = []
    for a in alphas:
        name = tag(a, seed, (0, 1, 2))
        if not os.path.exists(os.path.join(MODELS, f'{name}.zip')):
            print(f'[alpha] training {name} ...', flush=True)
            train(alpha=a, seed=seed, steps=steps)
        policy = ppo_policy_factory(name)
        df, _ = evaluate_arm(lambda: MosquitoEnv(eval_year=EVAL_YEAR), policy,
                             reps=reps)
        m = df.drop(columns=['rep']).mean(numeric_only=True)
        rows.append({'alpha': a, 'mean_bi': m['mean_bi'],
                     'weeks_bi_ge10_pct': m['weeks_bi_ge10_pct'],
                     'total_cost': m['total_cost']})
        print(f'[alpha={a}] meanBI={m["mean_bi"]:.2f} cost={m["total_cost"]:.0f}',
              flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, 'alpha_sweep.csv'), index=False)
    return out


def mode_robust(model_name='ppo_a0.1_s0_acts012', reps=10):
    """Re-evaluate the trained policy under perturbed environment dynamics."""
    policy = ppo_policy_factory(model_name)
    base = dict(INTERVENTION_PARAMS)
    scenarios = {'nominal': {}}
    for k, rel in [('sr_efficacy', 0.3), ('sp_efficacy', 0.3)]:
        scenarios[f'{k}-30%'] = {k: base[k] * (1 - rel)}
        scenarios[f'{k}+30%'] = {k: base[k] * (1 + rel)}
    scenarios['sr_decay_fast'] = {'sr_decay': 0.70}
    scenarios['sp_decay_slow'] = {'sp_decay': 0.70}
    scenarios['both_efficacy-30%'] = {'sr_efficacy': base['sr_efficacy'] * 0.7,
                                      'sp_efficacy': base['sp_efficacy'] * 0.7}
    rows = []
    for label, override in scenarios.items():
        params = dict(base)
        params.update(override)
        df, _ = evaluate_arm(lambda: MosquitoEnv(eval_year=EVAL_YEAR,
                                                 params=params),
                             policy, reps=reps)
        m = df.drop(columns=['rep']).mean(numeric_only=True)
        rows.append({'scenario': label, 'mean_bi': m['mean_bi'],
                     'weeks_bi_ge10_pct': m['weeks_bi_ge10_pct'],
                     'total_cost': m['total_cost']})
        print(f'[robust] {label:20s} meanBI={m["mean_bi"]:.2f}', flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, 'robustness_perturbation.csv'), index=False)
    return out


def mode_ablation(steps, seed=0, reps=10):
    rows = []
    for acts in [(0, 1, 2), (0, 1), (0, 2)]:
        name = tag(0.1, seed, acts)
        if not os.path.exists(os.path.join(MODELS, f'{name}.zip')):
            print(f'[ablation] training {name} ...', flush=True)
            train(alpha=0.1, seed=seed, steps=steps, action_set=acts)
        policy = ppo_policy_factory(name)
        df, _ = evaluate_arm(lambda: MosquitoEnv(eval_year=EVAL_YEAR,
                                                 action_set=acts),
                             policy, reps=reps)
        m = df.drop(columns=['rep']).mean(numeric_only=True)
        rows.append({'action_set': str(acts), 'mean_bi': m['mean_bi'],
                     'weeks_bi_ge10_pct': m['weeks_bi_ge10_pct'],
                     'total_cost': m['total_cost']})
        print(f'[ablation] acts={acts} meanBI={m["mean_bi"]:.2f} '
              f'cost={m["total_cost"]:.0f}', flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, 'action_ablation.csv'), index=False)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True,
                    choices=['alpha', 'robust', 'ablation'])
    ap.add_argument('--alphas', type=float, nargs='*',
                    default=[0.0, 0.05, 0.1, 0.2, 0.5])
    ap.add_argument('--steps', type=int, default=300_000)
    args = ap.parse_args()
    if args.mode == 'alpha':
        mode_alpha(args.alphas, args.steps)
    elif args.mode == 'robust':
        mode_robust()
    else:
        mode_ablation(args.steps)
