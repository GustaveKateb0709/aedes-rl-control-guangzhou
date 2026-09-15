#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Capacity sweep: how tight the weekly city-wide budget decides who wins.

Grid:  C_max in {2, 4, 6, 8, 12, inf}  x  10 training seeds (0-9), PPO.
Every C_max is evaluated against the SAME paired protocol on the held-out
2022 weather year with 10 noise replicates (seeds 10000-10009):

  none        no intervention (trivially feasible)
  threshold   spray when BI >= trigger; capacity-aware greedy projection
  calendar    preventive rotation in the transmission season; capacity-aware
  mpc1        exhaustive one-step lookahead (= degenerate MPC) over ALL
              feasible action vectors -- exact argmax, no learning
  ppo         PPO, 10 seeds
  a2c         second RL algorithm (A2C), 3 seeds
  dqn         third RL algorithm (DQN) on the flattened Discrete space,
              only where the flat space is small enough (see k_confound.py)

Modes
-----
  --mode train       launch resumable training (skips existing model files)
  --mode eval        evaluate every C_max x arm, write capacity_*.csv
  --mode mpc-timing  micro-benchmark of the exhaustive enumeration
  --mode regression  new-vs-old consistency: C_max=None, legacy models
  --mode tune-rules  re-tune the rule parameters on the training years,
                     per C_max, under the same objective as the agent

Nothing written here overwrites a pre-existing results/ or models/ file.
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

from mosquito_env import MosquitoEnv, GRIDS, N_GRIDS, BI_THRESHOLD  # noqa: E402
import baselines                                                     # noqa: E402
import baselines_extra as bx                                         # noqa: E402

RES = os.path.abspath(os.path.join(HERE, '..', 'results'))
MODELS = os.path.abspath(os.path.join(HERE, '..', 'models'))
os.makedirs(RES, exist_ok=True)

ALPHA = 0.1
TRAIN_YEARS = tuple(range(2015, 2022))
EVAL_YEAR = 2022
CMAX_VALUES = [2, 4, 6, 8, 12, None]
PPO_SEEDS = list(range(10))
# The flattened action space is Discrete(K) with K = 56 / 486 / 2193 / 6498 /
# 25149 / 59049.  Training cost grows with K in the policy head: one
# 400k-step PPO run costs ~216 s at K=56, ~370 s at K=2193, ~800 s at K=6498
# and ~2500 s at K=25149 (measured on this machine, 3 runs in parallel).
# C_max = 12 is provably NON-BINDING: the unconstrained per-grid optimum never
# demands more than 7 cost units in any week (results/spatial_budget_demand.csv),
# so C_max = 8, 12 and inf sit on the same plateau.  It is kept in the sweep and
# trained with the full 10 seeds, like every other ceiling, so that an
# across-seed spread can be reported for every row.
PPO_SEEDS_BY_CMAX = {}


def ppo_seeds(C):
    return PPO_SEEDS_BY_CMAX.get(c_tag(C), PPO_SEEDS)


A2C_SEEDS = list(range(3))
DQN_SEEDS = list(range(3))
DQN_CMAX = ['2', '6']            # flat Discrete space small enough to be usable
STEPS = 400_000
A2C_STEPS = 400_000
DQN_STEPS = 200_000
REPS = 10
EVAL_SEEDS = [10_000 + r for r in range(REPS)]
PY = sys.executable


def c_tag(C):
    return 'inf' if C is None else str(C)


def _cfg_key(v):
    """Canonical C_max tag for a value that may have been read back from CSV
    as a float (2.0 -> '2', inf -> 'inf')."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if np.isinf(f):
        return 'inf'
    return str(int(f)) if float(f).is_integer() else str(f)


def model_name(algo, C, seed):
    return f'{algo}_c{c_tag(C)}_s{seed}'


# --------------------------------------------------------------------- train
def job_list():
    jobs = []
    for C in CMAX_VALUES:
        for s in ppo_seeds(C):
            jobs.append(('ppo', C, s, STEPS))
    for C in CMAX_VALUES:
        for s in A2C_SEEDS:
            jobs.append(('a2c', C, s, A2C_STEPS))
    for ctok in DQN_CMAX:
        C = int(ctok)
        for s in DQN_SEEDS:
            jobs.append(('dqn', C, s, DQN_STEPS))
    return jobs


def train_mode(workers=3, only=None):
    jobs = job_list()
    if only:
        jobs = [j for j in jobs if j[0] == only]
    todo = [j for j in jobs
            if not os.path.exists(os.path.join(MODELS, f'{model_name(j[0], j[1], j[2])}.zip'))]
    print(f'[train] {len(todo)}/{len(jobs)} jobs to run, {workers} workers',
          flush=True)
    running, t0 = [], time.time()
    queue = list(todo)
    while queue or running:
        while queue and len(running) < workers:
            algo, C, seed, steps = queue.pop(0)
            name = model_name(algo, C, seed)
            cmd = [PY, os.path.join(HERE, 'train_ppo.py'), '--algo', algo,
                   '--seed', str(seed), '--steps', str(steps),
                   '--name', name, '--alpha', str(ALPHA)]
            if C is not None:
                cmd += ['--C-max', str(C)]
            log = os.path.join(RES, 'monitor', f'{name}.train.log')
            os.makedirs(os.path.dirname(log), exist_ok=True)
            fh = open(log, 'w')
            p = subprocess.Popen(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
            running.append((p, name, seed, C, fh, time.time()))
            print(f'[train] start {name} ({len(running)} running)', flush=True)
        time.sleep(2)
        for item in list(running):
            p, name, seed, C, fh, tstart = item
            if p.poll() is not None:
                fh.close()
                ok = os.path.exists(os.path.join(MODELS, f'{name}.zip'))
                print(f'[train] done {name} rc={p.returncode} '
                      f'{time.time()-tstart:.0f}s ok={ok}', flush=True)
                running.remove(item)
    print(f'[train] all done in {time.time()-t0:.0f}s', flush=True)


# ---------------------------------------------------------------------- eval
def metrics(bis, costs, n_sr, n_sp, rewards, C_max, overflows):
    B = np.asarray(bis)
    return {'mean_bi': float(B.mean()),
            'peak_bi': float(B.max()),
            'weeks_bi_ge10_pct': float((B >= 10).mean() * 100),
            'weeks_bi_ge5_pct': float((B >= 5).mean() * 100),
            'total_cost': float(np.sum(costs)),
            'mean_weekly_cost': float(np.mean(costs)),
            'n_sr': int(np.sum(n_sr)), 'n_sp': int(np.sum(n_sp)),
            'episode_return': float(np.sum(rewards)),
            'n_infeasible_weeks': int(np.sum(np.asarray(overflows) > 1e-9)),
            'budget_utilisation': (np.nan if C_max is None
                                   else float(np.mean(costs) / C_max))}


def run_arm(env_factory, policy_fn, C_max):
    rows = []
    for rep, sd in enumerate(EVAL_SEEDS):
        env = env_factory()
        env.reset(seed=sd)
        bis, costs, nsr, nsp, rew, ovf = [], [], [], [], [], []
        done = False
        while not done:
            a = policy_fn(env)
            _, r, te, tr, info = env.step(a)
            done = te or tr
            bis.append(env.bi.copy())
            costs.append(info['cost'])
            nsr.append(info['n_sr'])
            nsp.append(info['n_sp'])
            rew.append(r)
            ovf.append(info['capacity_overflow'])
        d = metrics(bis, costs, nsr, nsp, rew, C_max, ovf)
        d.update({'rep': rep, 'eval_seed': sd})
        rows.append(d)
    return pd.DataFrame(rows)


def env_factory(C, **kw):
    return lambda: MosquitoEnv(alpha=ALPHA, eval_year=EVAL_YEAR,
                               C_max=C, **kw)


def evaluate_all(rules_cfg=None):
    rules_cfg = rules_cfg or {}
    all_rows, ppo_rows = [], []
    for C in CMAX_VALUES:
        cfg = rules_cfg.get(c_tag(C), {})
        tthr = cfg.get('threshold_trigger', 15.0)
        csr, cper = cfg.get('calendar_sr', BI_THRESHOLD), cfg.get('calendar_period', 4)
        print(f'[eval] C_max={c_tag(C)} rules: thr={tthr} cal(sr={csr},p={cper})',
              flush=True)
        arms = {
            'none': lambda env: np.zeros(N_GRIDS, dtype=int),
            'threshold': bx.make_rule_policy(
                lambda env: baselines.threshold_rule(env, spray_trigger=tthr)),
            'calendar': bx.make_rule_policy(
                lambda env: baselines.calendar_rule(env, sr_trigger=float(csr),
                                                    period=int(cper))),
            'mpc1': bx.mpc_one_step,
        }
        for arm, fn in arms.items():
            df = run_arm(env_factory(C), fn, C)
            df.insert(0, 'arm', arm)
            df.insert(0, 'seed', -1)
            df.insert(0, 'C_max', c_tag(C))
            all_rows.append(df)
            print(f'   {arm:10s} return={df.episode_return.mean():8.3f} '
                  f'BI={df.mean_bi.mean():5.2f} cost={df.total_cost.mean():6.1f}',
                  flush=True)
        for algo, seeds, arm in [('ppo', ppo_seeds(C), 'ppo'),
                                 ('a2c', A2C_SEEDS, 'a2c'),
                                 ('dqn', DQN_SEEDS, 'dqn')]:
            if algo == 'dqn' and c_tag(C) not in DQN_CMAX:
                continue
            for sd in seeds:
                name = model_name(algo, C, sd)
                if not os.path.exists(os.path.join(MODELS, f'{name}.zip')):
                    print(f'   [skip] {name} missing', flush=True)
                    continue
                pol = bx.sb3_policy_factory(name, algo=algo,
                                            env_kwargs={'C_max': C,
                                                        'alpha': ALPHA})
                df = run_arm(env_factory(C), pol, C)
                df.insert(0, 'arm', arm)
                df.insert(0, 'seed', sd)
                df.insert(0, 'C_max', c_tag(C))
                df.insert(1, 'model', name)
                all_rows.append(df)
                if algo == 'ppo':
                    ppo_rows.append(df)
                print(f'   {name:16s} return={df.episode_return.mean():8.3f} '
                      f'BI={df.mean_bi.mean():5.2f} cost={df.total_cost.mean():6.1f}',
                      flush=True)
    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(os.path.join(RES, 'capacity_arms.csv'), index=False)
    if ppo_rows:
        pd.concat(ppo_rows, ignore_index=True).to_csv(
            os.path.join(RES, 'capacity_ppo.csv'), index=False)
    agg = (out.groupby(['C_max', 'arm'])
           .agg(seeds=('seed', lambda s: len(set(s))),
                n=('episode_return', 'size'),
                ret_mean=('episode_return', 'mean'),
                ret_sd=('episode_return', 'std'),
                bi_mean=('mean_bi', 'mean'), bi_sd=('mean_bi', 'std'),
                peak_bi=('peak_bi', 'mean'),
                w10=('weeks_bi_ge10_pct', 'mean'),
                w5=('weeks_bi_ge5_pct', 'mean'),
                cost=('total_cost', 'mean'), cost_sd=('total_cost', 'std'),
                budget_util=('budget_utilisation', 'mean'),
                infeasible_weeks=('n_infeasible_weeks', 'mean'))
           .reset_index())
    agg.to_csv(os.path.join(RES, 'capacity_summary.csv'), index=False)

    # seed-level aggregation: the seed is the unit of replication for the
    # learning arms, so averaging over seed x replicate pools two variance
    # sources and understates the seed-to-seed spread.  Error bars in the
    # paper must come from this table.
    seed_lvl = (out.groupby(['C_max', 'arm', 'seed'])
                .agg(ret=('episode_return', 'mean'),
                     bi=('mean_bi', 'mean'),
                     cost=('total_cost', 'mean'),
                     w10=('weeks_bi_ge10_pct', 'mean'))
                .reset_index())
    seed_lvl.to_csv(os.path.join(RES, 'capacity_seed_level.csv'), index=False)
    agg_seed = (seed_lvl.groupby(['C_max', 'arm'])
                .agg(n_seeds=('seed', 'nunique'),
                     ret_mean=('ret', 'mean'), ret_sd=('ret', 'std'),
                     ret_min=('ret', 'min'), ret_max=('ret', 'max'),
                     bi_mean=('bi', 'mean'), cost_mean=('cost', 'mean'))
                .reset_index())
    if len(agg_seed):
        agg_seed['ret_se'] = agg_seed['ret_sd'] / np.sqrt(agg_seed['n_seeds'])
    agg_seed.to_csv(os.path.join(RES, 'capacity_summary_seed_level.csv'),
                    index=False)
    print('\n', agg_seed.round(4).to_string(index=False), flush=True)
    print(f'\nsaved -> {RES}/capacity_arms.csv, capacity_ppo.csv, '
          f'capacity_summary.csv, capacity_summary_seed_level.csv', flush=True)
    return out


# ------------------------------------------------------------- budget usage
def budget_use(rules_cfg=None, reps=None):
    """Distribution of the WEEKLY cost actually spent by every arm.

    The capacity sweep only stored episode means; this mode records, for each
    arm and C_max, the full weekly-cost distribution (mean / median / p90 /
    p99 / max), the share of weeks sitting exactly at the cap, and the share
    of weeks with no spending at all.  This is what answers "does the optimal
    policy even use the budget it is given?".
    """
    from mosquito_env import MosquitoEnv as _E
    rules_cfg = rules_cfg or {}
    reps = EVAL_SEEDS if reps is None else reps
    rows = []
    for C in CMAX_VALUES:
        cfg = rules_cfg.get(c_tag(C), {})
        tthr = cfg.get('threshold_trigger', 15.0)
        csr, cper = cfg.get('calendar_sr', BI_THRESHOLD), cfg.get('calendar_period', 4)
        arms = {
            'none': lambda env: np.zeros(N_GRIDS, dtype=int),
            'threshold': bx.make_rule_policy(
                lambda env: baselines.threshold_rule(env, spray_trigger=tthr)),
            'calendar': bx.make_rule_policy(
                lambda env: baselines.calendar_rule(env, sr_trigger=float(csr),
                                                    period=int(cper))),
            'mpc1': bx.mpc_one_step,
        }
        for arm, fn in arms.items():
            wk, nsr, nsp = [], [], []
            for sd in reps:
                env = MosquitoEnv(alpha=ALPHA, eval_year=EVAL_YEAR, C_max=C)
                env.reset(seed=sd)
                done = False
                while not done:
                    _, _, te, tr, info = env.step(fn(env))
                    done = te or tr
                    wk.append(info['cost']); nsr.append(info['n_sr'])
                    nsp.append(info['n_sp'])
            rows.append(_use_row(arm, -1, C, wk, nsr, nsp))
        for sd in ppo_seeds(C):
            name = model_name('ppo', C, sd)
            if not os.path.exists(os.path.join(MODELS, f'{name}.zip')):
                continue
            pol = bx.sb3_policy_factory(name, algo='ppo',
                                        env_kwargs={'C_max': C, 'alpha': ALPHA})
            wk, nsr, nsp = [], [], []
            for rep_sd in reps:
                env = MosquitoEnv(alpha=ALPHA, eval_year=EVAL_YEAR, C_max=C)
                env.reset(seed=rep_sd)
                done = False
                while not done:
                    _, _, te, tr, info = env.step(pol(env))
                    done = te or tr
                    wk.append(info['cost']); nsr.append(info['n_sr'])
                    nsp.append(info['n_sp'])
            rows.append(_use_row('ppo', sd, C, wk, nsr, nsp))
        print(f'[budget-use] C_max={c_tag(C)} done', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, 'capacity_budget_use.csv'), index=False)
    agg = (df.groupby(['C_max', 'arm'])
           .agg(seeds=('seed', lambda s: len(set(s))),
                weekly_cost_mean=('weekly_cost_mean', 'mean'),
                weekly_cost_p90=('weekly_cost_p90', 'mean'),
                weekly_cost_p99=('weekly_cost_p99', 'mean'),
                weekly_cost_max=('weekly_cost_max', 'max'),
                frac_weeks_at_cap=('frac_weeks_at_cap', 'mean'),
                frac_weeks_zero=('frac_weeks_zero', 'mean'),
                total_cost=('total_cost', 'mean'),
                n_sr_per_week=('n_sr_per_week', 'mean'),
                n_sp_per_week=('n_sp_per_week', 'mean'))
           .reset_index())
    agg.to_csv(os.path.join(RES, 'capacity_budget_use_summary.csv'), index=False)
    sub = agg[agg['arm'].isin(['ppo', 'calendar', 'mpc1'])]
    print('\n' + sub.round(4).to_string(index=False), flush=True)
    return df


def _use_row(arm, seed, C, weekly, n_sr, n_sp):
    w = np.asarray(weekly, dtype=float)
    cap = np.nan if C is None else float(C)
    return {'C_max': c_tag(C), 'arm': arm, 'seed': seed,
            'weekly_cost_mean': float(w.mean()),
            'weekly_cost_sd': float(w.std()),
            'weekly_cost_median': float(np.median(w)),
            'weekly_cost_p90': float(np.percentile(w, 90)),
            'weekly_cost_p99': float(np.percentile(w, 99)),
            'weekly_cost_max': float(w.max()),
            'frac_weeks_at_cap': (np.nan if C is None
                                  else float(np.mean(np.abs(w - cap) < 1e-9))),
            'frac_weeks_zero': float(np.mean(w <= 1e-9)),
            'total_cost': float(w.sum()),
            'budget_utilisation': (np.nan if C is None else float(w.mean() / cap)),
            'n_sr_per_week': float(np.mean(n_sr)),
            'n_sp_per_week': float(np.mean(n_sp))}


# --------------------------------------------------------------- rule tuning
def _episode_summary(env, policy_fn):
    env.reset()
    done, ret, bis, w10, cost = False, 0.0, [], [], []
    while not done:
        a = policy_fn(env)
        _, r, te, tr, info = env.step(a)
        done = te or tr
        ret += r
        bis.append(info['mean_bi'])
        w10.append(info['weeks_above10'])
        cost.append(info['cost'])
    return ret, float(np.mean(bis)), float(np.sum(cost)), float(np.mean(w10) * 100)


def tune_rules(C_values=None, seeds=range(10)):
    """Re-tune both rules on the TRAINING years for every C_max, using the
    same objective the agent optimises (episode return at alpha = 0.1)."""
    C_values = CMAX_VALUES if C_values is None else C_values
    rows = []
    for C in C_values:
        for trig in [5, 8, 10, 12, 15, 20, 25]:
            R = []
            for sd in seeds:
                for yr in TRAIN_YEARS:
                    env = MosquitoEnv(alpha=ALPHA, train_years=TRAIN_YEARS,
                                      eval_year=yr, seed=sd, C_max=C)
                    R.append(_episode_summary(
                        env, bx.make_rule_policy(
                            lambda e, t=trig: baselines.threshold_rule(e, spray_trigger=t)))[0])
            rows.append({'C_max': c_tag(C), 'rule': 'threshold',
                         'params': f'trigger={trig}', 'train_return': float(np.mean(R))})
        for sr_trig in [3, 5, 8]:
            for period in [2, 4]:
                R = []
                for sd in seeds:
                    for yr in TRAIN_YEARS:
                        env = MosquitoEnv(alpha=ALPHA, train_years=TRAIN_YEARS,
                                          eval_year=yr, seed=sd, C_max=C)
                        R.append(_episode_summary(
                            env, bx.make_rule_policy(
                                lambda e, s=sr_trig, p=period:
                                baselines.calendar_rule(e, sr_trigger=float(s),
                                                        period=p)))[0])
                rows.append({'C_max': c_tag(C), 'rule': 'calendar',
                             'params': f'sr_trigger={sr_trig},period={period}',
                             'train_return': float(np.mean(R))})
        print(f'[tune] C_max={c_tag(C)} done', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, 'capacity_rule_tuning.csv'), index=False)
    best = {}
    for C in C_values:
        sub = df[df['C_max'] == c_tag(C)]
        bt = sub[sub['rule'] == 'threshold'].loc[
            lambda d: d['train_return'].idxmax()]
        bc = sub[sub['rule'] == 'calendar'].loc[
            lambda d: d['train_return'].idxmax()]
        cfg = {'threshold_trigger': float(bt['params'].split('=')[1])}
        cfg['calendar_sr'] = int(bc['params'].split('sr_trigger=')[1].split(',')[0])
        cfg['calendar_period'] = int(bc['params'].split('period=')[1])
        cfg['threshold_params'] = bt['params']
        cfg['calendar_params'] = bc['params']
        best[c_tag(C)] = cfg
        print(f'[tune] best @ C_max={c_tag(C)}: {bt["params"]} | {bc["params"]}',
              flush=True)
    pd.DataFrame([{'C_max': k, **v} for k, v in best.items()]).astype(
        {'C_max': str}).to_csv(os.path.join(RES, 'capacity_rule_best.csv'),
                               index=False)
    return best


# ----------------------------------------------------------------- regression
def regression():
    """C_max = None: re-evaluate the LEGACY models and compare with the
    archived results/counterfactual_2022.csv (new-vs-old consistency)."""
    from evaluate import ppo_policy_factory, evaluate_arm
    rows = []
    for arm in ['none', 'threshold', 'calendar']:
        df, _ = evaluate_arm(env_factory(None), baselines.RULES[arm], reps=REPS)
        df['arm'] = arm
        rows.append(df)
    for name in ['ppo_a0.1_s0_acts012', 'ppo_a0.1_s1_acts012',
                 'ppo_a0.1_s2_acts012']:
        if not os.path.exists(os.path.join(MODELS, f'{name}.zip')):
            print(f'[regression] missing {name}', flush=True)
            continue
        df, _ = evaluate_arm(env_factory(None), ppo_policy_factory(name), reps=REPS)
        df['arm'] = name
        rows.append(df)
    new = pd.concat(rows, ignore_index=True)
    new.to_csv(os.path.join(RES, 'regression_cmax_none_2022.csv'), index=False)
    old = pd.read_csv(os.path.join(RES, 'counterfactual_2022.csv'))
    keys = ['arm', 'rep', 'mean_bi', 'peak_bi', 'weeks_bi_ge10_pct',
            'total_cost']
    m = old[keys].merge(new[keys], on=['arm', 'rep'], suffixes=('_old', '_new'))
    print(f'[regression] matched rows: {len(m)} / old {len(old)} / new {len(new)}',
          flush=True)
    worst = {}
    for c in ['mean_bi', 'peak_bi', 'weeks_bi_ge10_pct', 'total_cost']:
        d = np.abs(m[f'{c}_old'] - m[f'{c}_new'])
        worst[c] = float(np.nanmax(d)) if len(d) else np.nan
    print('[regression] max |old-new| per metric:', worst, flush=True)
    m.to_csv(os.path.join(RES, 'regression_cmax_none_compare.csv'), index=False)
    return worst


# ---------------------------------------------------------------- mpc timing
def mpc_timing():
    rows = []
    for C in CMAX_VALUES:
        env = MosquitoEnv(alpha=ALPHA, eval_year=EVAL_YEAR, C_max=C)
        env.reset(seed=EVAL_SEEDS[0])
        r = bx.mpc_timing(env, n_repeat=10, note=f'C_max={c_tag(C)}')
        rows.append({'C_max': c_tag(C), **{k: r[k] for k in
                                           ['n_candidates',
                                            'vectorised_sec_per_step',
                                            'naive_sec_per_full_enum_est']}})
        print(f"[mpc] C_max={c_tag(C):>4}: K={r['n_candidates']:6d} "
              f"vectorised={r['vectorised_sec_per_step']*1000:.3f} ms/step "
              f"naive(est)={r['naive_sec_per_full_enum_est']*1000:.1f} ms/step",
              flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, 'capacity_mpc_timing.csv'),
                              index=False)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True,
                    choices=['train', 'eval', 'mpc-timing', 'regression',
                             'tune-rules', 'budget-use'])
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--only', default=None, choices=['ppo', 'a2c', 'dqn'])
    args = ap.parse_args()
    _cfg_path = os.path.join(RES, 'capacity_rule_best.csv')
    _rules_cfg = None
    if os.path.exists(_cfg_path):
        _rb = pd.read_csv(_cfg_path)
        _rules_cfg = {_cfg_key(r['C_max']): r.to_dict() for _, r in _rb.iterrows()}
    if args.mode == 'train':
        train_mode(args.workers, args.only)
    elif args.mode == 'mpc-timing':
        mpc_timing()
    elif args.mode == 'regression':
        regression()
    elif args.mode == 'tune-rules':
        tune_rules()
    elif args.mode == 'budget-use':
        budget_use(_rules_cfg)
    else:
        if _rules_cfg:
            print('[eval] tuned rule configs loaded for:',
                  sorted(_rules_cfg), flush=True)
        evaluate_all(_rules_cfg)
