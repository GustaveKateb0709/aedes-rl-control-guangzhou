#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tuned-rule fairness analysis.

The static rule baselines in the main text use fixed, hand-picked
parameters (threshold trigger BI>=15; calendar sr_trigger=5, season weeks
18-40, period=4), while PPO was trained/optimized. This script removes
that asymmetry: both rules are grid-search TUNED ON THE TRAINING YEARS
(2015-2021, same noise-seed protocol), the best configuration by the same
objective the PPO agent optimizes (episode return at alpha=0.1) is
selected, and only that tuned rule is carried to the held-out 2022 year.

Outputs:
  results/rule_tuning_train.csv  - all configurations, training-year performance
  results/rule_tuned_2022.csv    - tuned rules vs original fixed rules, 2022
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Package root: parent of this code/ directory (override with MOSQ_DATA_DIR)
RL_PKG = os.environ.get('MOSQ_DATA_DIR', os.path.join(HERE, '..'))
sys.path.insert(0, HERE)

from mosquito_env import MosquitoEnv, N_GRIDS, BI_THRESHOLD, REWARD_SCALE
from baselines import threshold_rule, calendar_rule

ALPHA = 0.1
TRAIN_YEARS = tuple(range(2015, 2022))
N_SEEDS = 10
RES = os.path.join(HERE, '..', 'results')
os.makedirs(RES, exist_ok=True)


def episode_return(env, policy_fn):
    """Run one episode with a rule; return (return, mean BI, cost, %weeks>=10)."""
    obs, _ = env.reset()
    done, ret = False, 0.0
    bis, w10 = [], []
    while not done:
        a = policy_fn(env)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        ret += r
        bis.append(info['mean_bi'])
        w10.append(info['weeks_above10'])
    return ret, float(np.mean(bis)), env.cost_total, float(np.mean(w10) * 100)


def eval_rule(policy_fn, years=TRAIN_YEARS, seeds=range(N_SEEDS)):
    """Mean performance over training years x noise seeds (paired protocol)."""
    R, B, C, W = [], [], [], []
    for sd in seeds:
        for yr in years:
            env = MosquitoEnv(alpha=ALPHA, train_years=years,
                              eval_year=yr, seed=sd)
            env.reset(seed=sd)
            r, b, c, w = episode_return(env, policy_fn)
            R.append(r); B.append(b); C.append(c); W.append(w)
    return (float(np.mean(R)), float(np.mean(B)), float(np.mean(C)),
            float(np.mean(W)), len(R))


def eval_2022(policy_fn, seeds=range(N_SEEDS)):
    R, B, C, W = [], [], [], []
    for sd in seeds:
        env = MosquitoEnv(alpha=ALPHA, train_years=TRAIN_YEARS,
                          eval_year=2022, seed=sd)
        env.reset(seed=sd)
        r, b, c, w = episode_return(env, policy_fn)
        R.append(r); B.append(b); C.append(c); W.append(w)
    sd_ = lambda x: float(np.std(x, ddof=1))
    return {'return': (float(np.mean(R)), sd_(R)),
            'mean_bi': (float(np.mean(B)), sd_(B)),
            'cost': (float(np.mean(C)), sd_(C)),
            'pct_weeks_ge10': (float(np.mean(W)), sd_(W))}


def main():
    rows = []
    print('== Stage 1: tune rules on TRAINING years 2015-2021 ==', flush=True)
    for trig in [5, 8, 10, 12, 15, 20, 25]:
        ret, bi, cost, w10, n = eval_rule(
            lambda env, t=trig: threshold_rule(env, spray_trigger=t))
        rows.append({'rule': 'threshold', 'params': f'trigger={trig}',
                     'train_return': round(ret, 3), 'train_mean_bi': round(bi, 3),
                     'train_cost': round(cost, 1), 'train_pct_w10': round(w10, 2),
                     'n_eps': n})
        print(f'  threshold trigger={trig:4.1f}: return={ret:8.3f} '
              f'BI={bi:6.2f} cost={cost:6.1f}', flush=True)
    for sr_trig in [3, 5, 8]:
        for period in [2, 4]:
            def mk(env, st=sr_trig, pd_=period):
                return calendar_rule(env, sr_trigger=st,
                                     season=(18, 40), period=pd_)
            ret, bi, cost, w10, n = eval_rule(mk)
            rows.append({'rule': 'calendar',
                         'params': f'sr_trigger={sr_trig},period={period}',
                         'train_return': round(ret, 3),
                         'train_mean_bi': round(bi, 3),
                         'train_cost': round(cost, 1),
                         'train_pct_w10': round(w10, 2), 'n_eps': n})
            print(f'  calendar sr={sr_trig} period={period}: return={ret:8.3f} '
                  f'BI={bi:6.2f} cost={cost:6.1f}', flush=True)
    train = pd.DataFrame(rows)
    train.to_csv(os.path.join(RES, 'rule_tuning_train.csv'), index=False)

    best_thr = train[train['rule'] == 'threshold'].loc[
        lambda d: d['train_return'].idxmax()]
    best_cal = train[train['rule'] == 'calendar'].loc[
        lambda d: d['train_return'].idxmax()]
    print(f'\nbest threshold on train: {best_thr["params"]}', flush=True)
    print(f'best calendar  on train: {best_cal["params"]}', flush=True)

    print('\n== Stage 2: evaluate tuned rules on held-out 2022 ==', flush=True)
    out = []
    bt = float(best_thr['params'].split('=')[1])
    cfg = best_cal['params']
    cs = int(cfg.split('sr_trigger=')[1].split(',')[0])
    cp = int(cfg.split('period=')[1])
    arms = {
        'Threshold (original, trigger=15)':
            lambda env: threshold_rule(env, spray_trigger=15.0),
        f'Threshold (tuned, {best_thr["params"]})':
            lambda env: threshold_rule(env, spray_trigger=bt),
        'Calendar (original, sr=5, period=4)':
            lambda env: calendar_rule(env, sr_trigger=5.0, period=4),
        f'Calendar (tuned, {best_cal["params"]})':
            lambda env: calendar_rule(env, sr_trigger=float(cs), period=cp),
    }
    for name, fn in arms.items():
        r = eval_2022(fn)
        out.append({'arm': name,
                    'return': round(r['return'][0], 3),
                    'mean_bi': round(r['mean_bi'][0], 2),
                    'mean_bi_sd': round(r['mean_bi'][1], 2),
                    'cost': round(r['cost'][0], 1),
                    'pct_weeks_ge10': round(r['pct_weeks_ge10'][0], 2)})
        print(f'  {name}: BI={r["mean_bi"][0]:.2f}±{r["mean_bi"][1]:.2f} '
              f'cost={r["cost"][0]:.1f} return={r["return"][0]:.2f}', flush=True)
    # PPO reference from the main table (alpha = 0.1, seed 0): BI 2.14±0.08, cost 158.4
    out.append({'arm': 'PPO (seed 0, from main Table 1)', 'return': '',
                'mean_bi': 2.14, 'mean_bi_sd': 0.08, 'cost': 158.4,
                'pct_weeks_ge10': 0.3})
    fin = pd.DataFrame(out)
    fin.to_csv(os.path.join(RES, 'rule_tuned_2022.csv'), index=False)
    print(f'\nsaved -> {RES}/rule_tuning_train.csv, rule_tuned_2022.csv', flush=True)


if __name__ == '__main__':
    main()
