#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Leave-one-year-out (LOYO) cross-validation over 2015-2022.

For every fold year y in 2015..2022 the agent is retrained on the OTHER
seven years and evaluated on y with 10 paired noise replicates
(seeds 10000-10009).  The capacity regime is C_max = 2, the tightest ceiling of
the sweep and the only one that genuinely binds (the budget is spent to the
limit in ~48% of weeks; see results/capacity_budget_use_summary.csv).

The rule baselines are re-tuned ON EACH FOLD'S TRAINING YEARS (same
objective as the agent, alpha = 0.1) before being applied to the held-out
year, so the comparison stays fair inside every fold.

Seeds: 10 per fold, matching the main configuration.  One 400k-step PPO run at
C_max = 2 (K = 56) costs ~200-320 s on this machine, so 8 folds x 10 seeds
= 80 runs.

Usage:
  python loyo.py --mode train  [--workers 3] [--seeds 0 1 2 3 4 5 6 7 8 9]
  python loyo.py --mode eval
  python loyo.py --mode all
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mosquito_env import MosquitoEnv, N_GRIDS, BI_THRESHOLD            # noqa: E402
import baselines                                                       # noqa: E402
import baselines_extra as bx                                           # noqa: E402

RES = os.path.abspath(os.path.join(HERE, '..', 'results'))
MODELS = os.path.abspath(os.path.join(HERE, '..', 'models'))
ALPHA = 0.1
YEARS = list(range(2015, 2023))
# The manuscript reports the cross-validation at C_max = 2, the tightest
# ceiling and the only one that binds; the default must match it.
REGIMES = [2]
SEEDS = list(range(10))
EVAL_SEEDS = [10_000 + r for r in range(10)]
STEPS = 400_000
PY = sys.executable


def c_tag(C):
    return 'inf' if C is None else str(C)


def model_name(fold_year, C, seed):
    return f'ppo_loyo{fold_year}_c{c_tag(C)}_s{seed}'


def train_years_for(fold_year):
    return tuple(y for y in YEARS if y != fold_year)


# --------------------------------------------------------------------- train
def train_mode(workers=3, seeds=SEEDS, regimes=REGIMES):
    jobs = []
    for fy in YEARS:
        for C in regimes:
            for s in seeds:
                name = model_name(fy, C, s)
                if not os.path.exists(os.path.join(MODELS, f'{name}.zip')):
                    jobs.append((fy, C, s, name))
    print(f'[loyo-train] {len(jobs)} runs to launch, {workers} workers', flush=True)
    running, q, t0 = [], list(jobs), time.time()
    while q or running:
        while q and len(running) < workers:
            fy, C, s, name = q.pop(0)
            tys = train_years_for(fy)
            cmd = [PY, os.path.join(HERE, 'train_ppo.py'), '--algo', 'ppo',
                   '--seed', str(s), '--steps', str(STEPS), '--name', name,
                   '--alpha', str(ALPHA),
                   '--train-years', *[str(y) for y in tys],
                   '--eval-year', str(fy)]
            if C is not None:
                cmd += ['--C-max', str(C)]
            log = os.path.join(RES, 'monitor', f'{name}.train.log')
            fh = open(log, 'w')
            p = subprocess.Popen(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
            running.append((p, name, fh, time.time()))
            print(f'[loyo-train] start {name} (fold {fy}, train {tys[0]}-{tys[-1]})',
                  flush=True)
        time.sleep(2)
        for it in list(running):
            p, name, fh, ts = it
            if p.poll() is not None:
                fh.close()
                print(f'[loyo-train] done {name} rc={p.returncode} '
                      f'{time.time()-ts:.0f}s', flush=True)
                running.remove(it)
    print(f'[loyo-train] all done in {(time.time()-t0)/60:.1f} min', flush=True)


# ---------------------------------------------------------------------- eval
def _episode(env, policy_fn, reset_seed=None):
    """One episode.  `reset_seed` is passed straight to `env.reset` so the
    noise realisation matches the protocol used by capacity_sweep.py
    (seeds 10000-10009); no implicit second reset."""
    env.reset(seed=reset_seed) if reset_seed is not None else env.reset()
    done, ret, B, C, W = False, 0.0, [], [], []
    while not done:
        a = policy_fn(env)
        _, r, te, tr, info = env.step(a)
        done = te or tr
        ret += r
        B.append(info['mean_bi'])
        C.append(info['cost'])
        W.append(info['weeks_above10'])
    return ret, float(np.mean(B)), float(np.sum(C)), float(np.mean(W) * 100)


def run_model_on_year(name, fold_year, C, reps=EVAL_SEEDS):
    pol = bx.sb3_policy_factory(name, algo='ppo',
                                env_kwargs={'C_max': C, 'alpha': ALPHA,
                                            'train_years': train_years_for(fold_year),
                                            'eval_year': fold_year})
    rows = []
    for r, sd in enumerate(reps):
        env = MosquitoEnv(alpha=ALPHA, train_years=train_years_for(fold_year),
                          eval_year=fold_year, seed=sd, C_max=C)
        ret, bi, cost, w10 = _episode(env, pol, reset_seed=sd)
        rows.append({'rep': r, 'eval_seed': sd, 'episode_return': ret,
                     'mean_bi': bi, 'total_cost': cost, 'weeks_bi_ge10_pct': w10})
    return rows


def tune_rules_for_fold(fold_year, C, seeds=SEEDS):
    tys = train_years_for(fold_year)
    best = {}
    rows = []
    for trig in [8, 10, 12, 15, 20, 25]:
        R = []
        for sd in seeds:
            for yr in tys:
                env = MosquitoEnv(alpha=ALPHA, train_years=tys, eval_year=yr,
                                  seed=sd, C_max=C)
                R.append(_episode(env, bx.make_rule_policy(
                    lambda e, t=trig: baselines.threshold_rule(e, spray_trigger=t)))[0])
        rows.append({'rule': 'threshold', 'params': f'trigger={trig}',
                     'train_return': float(np.mean(R))})
    for sr_trig in [3, 5, 8]:
        for period in [2, 4]:
            R = []
            for sd in seeds:
                for yr in tys:
                    env = MosquitoEnv(alpha=ALPHA, train_years=tys, eval_year=yr,
                                      seed=sd, C_max=C)
                    R.append(_episode(env, bx.make_rule_policy(
                        lambda e, s=sr_trig, p=period:
                        baselines.calendar_rule(e, sr_trigger=float(s), period=p)))[0])
            rows.append({'rule': 'calendar',
                         'params': f'sr_trigger={sr_trig},period={period}',
                         'train_return': float(np.mean(R))})
    df = pd.DataFrame(rows)
    bt = df[df['rule'] == 'threshold'].loc[lambda d: d['train_return'].idxmax()]
    bc = df[df['rule'] == 'calendar'].loc[lambda d: d['train_return'].idxmax()]
    best['threshold_trigger'] = float(bt['params'].split('=')[1])
    best['calendar_sr'] = int(bc['params'].split('sr_trigger=')[1].split(',')[0])
    best['calendar_period'] = int(bc['params'].split('period=')[1])
    return best, df


def eval_mode(seeds=SEEDS, regimes=REGIMES):
    all_rows, tune_rows = [], []
    for fy in YEARS:
        tys = train_years_for(fy)
        for C in regimes:
            cfg, tdf = tune_rules_for_fold(fy, C, seeds)
            tdf.insert(0, 'fold_year', fy)
            tdf.insert(1, 'C_max', c_tag(C))
            tune_rows.append(tdf)
            arms = {
                'none': (lambda env, c=C: np.zeros(N_GRIDS, dtype=int), {}),
                'threshold': (bx.make_rule_policy(
                    lambda e, t=cfg['threshold_trigger']:
                    baselines.threshold_rule(e, spray_trigger=t)), {}),
                'calendar': (bx.make_rule_policy(
                    lambda e, s=cfg['calendar_sr'], p=cfg['calendar_period']:
                    baselines.calendar_rule(e, sr_trigger=float(s), period=p)), {}),
                'mpc1': (bx.mpc_one_step, {}),
            }
            for arm, (fn, kw) in arms.items():
                for r, sd in enumerate(EVAL_SEEDS):
                    env = MosquitoEnv(alpha=ALPHA, train_years=tys,
                                      eval_year=fy, seed=sd, C_max=C, **kw)
                    ret, bi, cost, w10 = _episode(env, fn, reset_seed=sd)
                    all_rows.append({'fold_year': fy, 'C_max': c_tag(C),
                                     'arm': arm, 'seed': -1, 'rep': r,
                                     'episode_return': ret, 'mean_bi': bi,
                                     'total_cost': cost,
                                     'weeks_bi_ge10_pct': w10})
            for s in seeds:
                name = model_name(fy, C, s)
                if not os.path.exists(os.path.join(MODELS, f'{name}.zip')):
                    print(f'[loyo-eval] missing {name}', flush=True)
                    continue
                for d in run_model_on_year(name, fy, C):
                    all_rows.append({'fold_year': fy, 'C_max': c_tag(C),
                                     'arm': 'ppo', 'seed': s, **d})
            sub = pd.DataFrame(all_rows)
            sub = sub[(sub.fold_year == fy) & (sub.C_max == c_tag(C))]
            g = sub[sub.arm == 'ppo'].groupby('seed').episode_return.mean()
            print(f'[loyo-eval] fold {fy} C_max={c_tag(C)}: '
                  f'ppo return={g.mean():.4f} (n_seeds={len(g)})  '
                  f'thr={sub[sub.arm=="threshold"].episode_return.mean():.4f}  '
                  f'mpc1={sub[sub.arm=="mpc1"].episode_return.mean():.4f}',
                  flush=True)
            pd.DataFrame(all_rows).to_csv(os.path.join(RES, 'loyo_runs.csv'),
                                          index=False)
    out = pd.DataFrame(all_rows)
    out.to_csv(os.path.join(RES, 'loyo_runs.csv'), index=False)
    pd.concat(tune_rows, ignore_index=True).to_csv(
        os.path.join(RES, 'loyo_rule_tuning.csv'), index=False)

    # per fold x arm aggregation (seeds are the unit for PPO, reps otherwise)
    agg = (out.groupby(['C_max', 'fold_year', 'arm'])
           .agg(n=('episode_return', 'size'),
                ret_mean=('episode_return', 'mean'),
                ret_sd=('episode_return', 'std'),
                bi_mean=('mean_bi', 'mean'),
                cost_mean=('total_cost', 'mean'))
           .reset_index())
    agg.to_csv(os.path.join(RES, 'loyo_fold_summary.csv'), index=False)

    # cross-year stability: mean over folds and its spread
    stab = (agg.groupby(['C_max', 'arm'])
            .agg(n_folds=('ret_mean', 'size'),
                 crossyear_mean=('ret_mean', 'mean'),
                 crossyear_sd=('ret_mean', 'std'),
                 crossyear_min=('ret_mean', 'min'),
                 crossyear_max=('ret_mean', 'max'),
                 bi_mean=('bi_mean', 'mean'),
                 cost_mean=('cost_mean', 'mean'))
            .reset_index())
    stab['crossyear_se'] = stab['crossyear_sd'] / np.sqrt(stab['n_folds'])
    stab['ci95_low'] = stab['crossyear_mean'] - 1.96 * stab['crossyear_se']
    stab['ci95_high'] = stab['crossyear_mean'] + 1.96 * stab['crossyear_se']
    stab.to_csv(os.path.join(RES, 'loyo_stability.csv'), index=False)
    print('\n', stab.round(4).to_string(index=False), flush=True)
    print(f'\nsaved -> {RES}/loyo_runs.csv, loyo_fold_summary.csv, loyo_stability.csv',
          flush=True)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['train', 'eval', 'all'])
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--seeds', type=int, nargs='*', default=SEEDS)
    ap.add_argument('--regimes', type=str, nargs='*', default=['2'])
    args = ap.parse_args()
    regs = [None if r in ('inf', 'None', 'none') else int(r) for r in args.regimes]
    if args.mode in ('train', 'all'):
        train_mode(args.workers, args.seeds, regs)
    if args.mode in ('eval', 'all'):
        eval_mode(args.seeds, regs)
