#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Isolate the ACTION-TABLE SIZE from the BUDGET constraint.

Motivation
----------
In the plain capacity sweep C_max and the size K of the (flattened) Discrete
action table are perfectly collinear: C_max = 2 -> K = 56, C_max = 12 ->
K = 25149.  PPO's return degrades monotonically along that same axis, so the
headline claim "tighter budget -> RL more valuable" cannot be separated from
"smaller action head -> easier optimisation".  This script breaks the
collinearity by holding C_max FIXED and varying only K.

Arms (all alpha = 0.1, 400k PPO steps unless noted)
---------------------------------------------------
  md       C_max = None, legacy MultiDiscrete([3]*10)  (10 seeds, REUSED:
           models/ppo_cinf_s{0..9}.zip -- this, not Discrete(59049), is the
           real status quo of the unconstrained arm)
  nest56   C_max = None, Discrete(56)   nested  (10 seeds, NEW)
  nest486  C_max = None, Discrete(486)  nested  (10 seeds, NEW)
  nest6498 C_max = None, Discrete(6498) nested  (10 seeds, NEW)
  nest59049 C_max = None, Discrete(59049) = full joint space  (3 seeds, NEW)
  rnd56    C_max = None, Discrete(56) uniform-random subset (3 seeds, REUSED)
           -- NEGATIVE CONTROL ONLY, see the warning below
  c2proj   C_max = 2,    Discrete(59049) full space + greedy projection onto
           the budget  (3 seeds, NEW)  (arm 'e': large action head, tight budget)

NESTED vs RANDOM tables
-----------------------
"Nested" means: every action whose weekly cost is <= 2 is ALWAYS in the table,
and only the remaining K-56 slots are drawn at random.  The core (cost <= 2) is
exactly the C_max = 2 feasible set, so `nest56` at C_max = None is *bit-identical*
to the C_max = 2 arm of the main sweep -- verified: identical action table and
max |delta reward| = 0.0 over an identical random action stream.  That gives a
free end-to-end consistency anchor.

A plain uniform-random subset (the `rnd` arms) was the FIRST design and it is
INVALID: a random 56-subset of 3**10 almost never contains the all-zero action
(frac_cost0 = 0.0), so the agent is forced to intervene every week.  Measured
return -3.317 +- 0.012 with total cost 326 (vs 56 for `md`) -- the penalty is
the alpha * cost term, not an action-head effect.  It is kept as a documented
negative control showing that table COMPOSITION, not just size, dominates.

Read-out
--------
  results/kconfound_tables.csv       what each table actually contains
                                     (size, cost distribution, best achievable
                                     one-step reward = "richness")
  results/kconfound_arms.csv         raw per-replicate metrics
  results/kconfound_seed_level.csv   seed-level means
  results/kconfound_summary.csv      mean +/- sd per arm + paired differences
                                     vs the `md` reference

Pre-specified interpretation
---------------------------
A SMALL spread between nest56 and nest59049 at C_max = None would mean the
parameterisation effect is small and the ceiling remains the dominant
explanation.  A LARGE spread (comparable to or bigger than the between-C_max
spread) would mean that the capacity sweep is largely an artefact of
action-table size and has to be read as such.

Nothing here overwrites a pre-existing results/ or models/ file.
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

from mosquito_env import MosquitoEnv, N_GRIDS, BI_THRESHOLD     # noqa: E402
import baselines_extra as bx                                    # noqa: E402
from capacity_sweep import (run_arm, metrics, EVAL_SEEDS, ALPHA,  # noqa: E402
                            EVAL_YEAR, RES, MODELS, _cfg_key)

STEPS = 400_000
SUBSET_SEED = 0
# Core of the nested control: every action with weekly cost <= 2 must be in the
# table.  That core is EXACTLY the C_max = 2 feasible set, so the nested K = 56
# arm at C_max = None is the same MDP as the C_max = 2 arm of the main sweep
# (verified bit-exactly: identical table, max |delta reward| = 0.0).  It is
# therefore both the smallest-K arm and a free consistency anchor.
CORE_COST = 2.0
# K = 59049 costs ~1-1.4 h per 400k-step run on this machine.  10 seeds x 2
# such arms would be ~28 h of serial compute, so the two largest-K arms run
# 3 seeds and that deviation is reported explicitly.
K_SEEDS = list(range(10))
BIG_K_SEEDS = list(range(3))
# The uniform-random subsample (no core) is kept ONLY as a documented negative
# control: it contains no cheap action, hence forces the agent to spend every
# week.  It was the first design and it is invalid; it is kept only as a
# documented negative control.
RND_SEEDS = list(range(3))
PY = sys.executable


def c_tag(C):
    return 'inf' if C is None else str(C)


# --------------------------------------------------------------- arm registry
def nested_kwargs(K):
    """C_max = None, Discrete(K), nested around every action of cost <= 2."""
    return {'C_max': None, 'alpha': ALPHA, 'action_table_size': K,
            'subset_seed': SUBSET_SEED, 'subset_core': CORE_COST}


def rnd_kwargs(K):
    """C_max = None, Discrete(K) over a uniform-random subset (NEGATIVE
    CONTROL -- no do-nothing action, so the agent is forced to spend)."""
    return {'C_max': None, 'alpha': ALPHA, 'action_table_size': K,
            'subset_seed': SUBSET_SEED}


ARMS = [
    # key,        label,                              env kwargs,                          seeds
    ('md',        'C=inf  MultiDiscrete (status quo)', {'C_max': None, 'alpha': ALPHA},    K_SEEDS),
    ('nest56',    'C=inf  Discrete(56) nested',        nested_kwargs(56),                  K_SEEDS),
    ('nest486',   'C=inf  Discrete(486) nested',       nested_kwargs(486),                 K_SEEDS),
    ('nest6498',  'C=inf  Discrete(6498) nested',      nested_kwargs(6498),                K_SEEDS),
    ('nest59049', 'C=inf  Discrete(59049) = full',     nested_kwargs(59049),               BIG_K_SEEDS),
    ('rnd56',     'C=inf  Discrete(56) RANDOM [neg ctl]', rnd_kwargs(56),                  RND_SEEDS),
    ('c2proj',    'C=2    Discrete(59049)+project',    {'C_max': 2, 'alpha': ALPHA,
                                                        'budget_projection': True},      BIG_K_SEEDS),
]
ARM_BY_KEY = {a[0]: a for a in ARMS}

# models already on disk under a different key (reused, never retrained)
REUSE = {('md', s): f'ppo_cinf_s{s}' for s in K_SEEDS}
REUSE.update({('rnd56', s): f'ppo_k56_s{s}' for s in RND_SEEDS})


def model_name(key, seed):
    if (key, seed) in REUSE:
        return REUSE[(key, seed)]
    if key == 'c2proj':
        return f'ppo_c2proj_s{seed}'
    if key.startswith('nest'):
        return f'ppo_nest{key[4:]}_s{seed}'
    if key.startswith('rnd'):
        return f'ppo_k{key[3:]}_s{seed}'
    return f'ppo_{key}_s{seed}'


def train_cmd(key, seed):
    """CLI to (re)produce the model of arm `key`, seed `seed`."""
    name = model_name(key, seed)
    cmd = [PY, os.path.join(HERE, 'train_ppo.py'), '--algo', 'ppo',
           '--seed', str(seed), '--steps', str(STEPS), '--name', name,
           '--alpha', str(ALPHA)]
    if key.startswith('nest'):
        cmd += ['--action-table-size', key[4:], '--subset-core', str(CORE_COST)]
    elif key.startswith('rnd'):
        cmd += ['--action-table-size', key[3:]]
    elif key == 'c2proj':
        cmd += ['--C-max', '2', '--budget-projection']
    return cmd


# ------------------------------------------------------------------- training
def enqueue(only=None):
    jobs = []
    for key, _label, _kw, seeds in ARMS:
        if only and key not in only:
            continue
        for s in seeds:
            if (key, s) in REUSE:               # already on disk, never retrain
                continue
            jobs.append((key, s))
    return jobs


def train_mode(workers=3, only=None):
    jobs = enqueue(only)
    todo = [j for j in jobs
            if not os.path.exists(os.path.join(MODELS, f'{model_name(*j)}.zip'))]
    print(f'[k-train] {len(todo)}/{len(jobs)} jobs to run, {workers} workers',
          flush=True)
    running, t0, queue, timings = [], time.time(), list(todo), []
    while queue or running:
        while queue and len(running) < workers:
            key, seed = queue.pop(0)
            name = model_name(key, seed)
            log = os.path.join(RES, 'monitor', f'{name}.train.log')
            os.makedirs(os.path.dirname(log), exist_ok=True)
            fh = open(log, 'w')
            p = subprocess.Popen(train_cmd(key, seed), cwd=HERE, stdout=fh,
                                 stderr=subprocess.STDOUT)
            running.append((p, key, seed, fh, time.time()))
            print(f'[k-train] start {name} ({len(running)} running)', flush=True)
        time.sleep(2)
        for item in list(running):
            p, key, seed, fh, tstart = item
            if p.poll() is not None:
                fh.close()
                dt = time.time() - tstart
                ok = os.path.exists(os.path.join(MODELS,
                                                 f'{model_name(key, seed)}.zip'))
                timings.append({'arm': key, 'seed': seed, 'wall_sec': dt,
                                'ok': ok, 'rc': p.returncode})
                print(f'[k-train] done {model_name(key, seed)} rc={p.returncode} '
                      f'{dt:.0f}s ok={ok}', flush=True)
                running.remove(item)
    print(f'[k-train] all done in {time.time()-t0:.0f}s', flush=True)
    if timings:
        pd.DataFrame(timings).to_csv(
            os.path.join(RES, 'kconfound_train_timing.csv'), index=False)
    return timings


# ----------------------------------------------------------------- table meta
def _probe_env(**kw):
    """Environment positioned at the seasonal PEAK week of the eval year with
    empty protection stocks, so that `peek_rewards` compares candidate plans
    on a genuinely high-risk next week rather than on an easy one."""
    e = MosquitoEnv(alpha=ALPHA, eval_year=EVAL_YEAR, **kw)
    e.reset(seed=EVAL_SEEDS[0])
    block = e.block
    peak = max(block[:-1], key=lambda t: e._base_bi_mean(t + 1).max())
    e.tpos = int(np.where(block == peak)[0][0])
    e.p_sr = np.zeros(N_GRIDS)
    e.p_sp = np.zeros(N_GRIDS)
    e.bi = e._base_bi_mean(block[e.tpos + 1]).copy()
    e.bi_prev = e.bi.copy()
    return e


def table_meta():
    """What each action table actually contains, plus a richness proxy.

    Richness = the best achievable IMMEDIATE reward over the table at the
    seasonal-peak state.  It is the cheapest honest way to see whether a small
    random subset is *able* to express a good plan at all -- without it, a poor
    small-K return could be mistaken for an optimisation effect when it is
    really table poverty.
    """
    rows = []
    for key, label, kw, _seeds in ARMS:
        if key == 'md':
            continue
        e = _probe_env(**{k: v for k, v in kw.items() if k != 'alpha'})
        tab = e.feasible_actions
        cost = e.feasible_costs
        r, nb = e.peek_rewards(tab)
        best = int(np.argmax(r))
        opt_health = float(np.mean((np.maximum(nb - BI_THRESHOLD, 0.0)
                                    / 5.0) ** 2))
        rows.append({'arm': key, 'label': label, 'K': int(len(tab)),
                     'cost_mean': float(cost.mean()),
                     'cost_sd': float(cost.std()),
                     'cost_max': float(cost.max()),
                     'frac_cost0': float(np.mean(cost <= 0)),
                     'peak_next_base_mean': float(e._base_bi_mean(
                         e.block[e.tpos + 1]).mean()),
                     'best_immediate_reward': float(r[best]),
                     'best_action_cost': float(cost[best]),
                     'best_action_excess': opt_health,
                     'median_immediate_reward': float(np.median(r))})
        print(f"[tables] {key:8s} K={len(tab):6d} cost {cost.mean():5.2f}"
              f"+-{cost.std():5.2f} max={cost.max():5.1f} "
              f"best_r={r[best]:+.4f} (cost {cost[best]:4.1f}, "
              f"excess {opt_health:.5f})", flush=True)
    e = _probe_env(flatten_actions=True)
    full = e.feasible_actions
    r, nb = e.peek_rewards(full)
    best = int(np.argmax(r))
    rows.append({'arm': 'full_space', 'label': '3**10 joint space (K=59049)',
                 'K': len(full), 'cost_mean': float(e.feasible_costs.mean()),
                 'cost_sd': float(e.feasible_costs.std()),
                 'cost_max': float(e.feasible_costs.max()),
                 'frac_cost0': float(np.mean(e.feasible_costs <= 0)),
                 'peak_next_base_mean': float(e._base_bi_mean(
                     e.block[e.tpos + 1]).mean()),
                 'best_immediate_reward': float(r[best]),
                 'best_action_cost': float(e.feasible_costs[best]),
                 'best_action_excess': float(np.mean(
                     (np.maximum(nb[best] - BI_THRESHOLD, 0.0) / 5.0) ** 2)),
                 'median_immediate_reward': float(np.median(r))})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, 'kconfound_tables.csv'), index=False)
    return df


# -------------------------------------------------------------------- evaluate
def evaluate_all(only=None):
    pre = 'kconfound' if not only else 'kconfound_partial'
    rows = []
    for key, label, kw, seeds in ARMS:
        if only and key not in only:
            continue
        env_kwargs = dict(kw)
        for sd in seeds:
            name = model_name(key, sd)
            if not os.path.exists(os.path.join(MODELS, f'{name}.zip')):
                print(f'  [skip] {name} missing', flush=True)
                continue
            pol = bx.sb3_policy_factory(name, algo='ppo', env_kwargs=env_kwargs)
            df = run_arm(lambda: MosquitoEnv(eval_year=EVAL_YEAR, **env_kwargs),
                         pol, kw.get('C_max'))
            df.insert(0, 'arm', key)
            df.insert(1, 'label', label)
            df.insert(2, 'seed', sd)
            df.insert(3, 'model', name)
            rows.append(df)
            util = (df.budget_utilisation.mean()
                    if df.budget_utilisation.notna().any() else float('nan'))
            print(f'  {name:20s} return={df.episode_return.mean():8.3f} '
                  f'BI={df.mean_bi.mean():5.2f} cost={df.total_cost.mean():6.1f} '
                  f'util={util:.3f}', flush=True)

    if not rows:
        print('[k-eval] nothing to evaluate', flush=True)
        return None
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(RES, f'{pre}_arms.csv'), index=False)

    seed_lvl = (out.groupby(['arm', 'label', 'seed'])
                .agg(ret=('episode_return', 'mean'),
                     bi=('mean_bi', 'mean'),
                     peak_bi=('peak_bi', 'mean'),
                     w10=('weeks_bi_ge10_pct', 'mean'),
                     cost=('total_cost', 'mean'),
                     util=('budget_utilisation', 'mean'),
                     infeas=('n_infeasible_weeks', 'mean'))
                .reset_index())
    seed_lvl.to_csv(os.path.join(RES, f'{pre}_seed_level.csv'), index=False)

    agg = (seed_lvl.groupby(['arm', 'label'])
           .agg(n_seeds=('seed', 'nunique'),
                ret_mean=('ret', 'mean'), ret_sd=('ret', 'std'),
                ret_min=('ret', 'min'), ret_max=('ret', 'max'),
                bi_mean=('bi', 'mean'), cost_mean=('cost', 'mean'),
                util_mean=('util', 'mean'),
                infeas_mean=('infeas', 'mean'))
           .reset_index())
    agg['ret_se'] = agg['ret_sd'] / np.sqrt(agg['n_seeds'])

    # paired differences vs the MultiDiscrete reference (same seed index)
    ref = (seed_lvl[seed_lvl['arm'] == 'md']
           .set_index('seed')['ret'] if (seed_lvl['arm'] == 'md').any() else None)
    diffs = []
    for key, label, _kw, _s in ARMS:
        if key == 'md' or ref is None:
            continue
        sub = seed_lvl[seed_lvl['arm'] == key].set_index('seed')['ret']
        common = sorted(set(sub.index) & set(ref.index))
        if not common:
            continue
        d = sub.loc[common].values - ref.loc[common].values
        n = len(d)
        sd = float(np.std(d, ddof=1)) if n > 1 else np.nan
        se = sd / np.sqrt(n) if n > 1 else np.nan
        diffs.append({'arm': key, 'label': label, 'n_paired_seeds': n,
                      'delta_vs_md': float(np.mean(d)),
                      'delta_sd': sd, 'delta_se': se,
                      't': (float(np.mean(d) / se) if se and se > 0 else np.nan)})
    dd = pd.DataFrame(diffs, columns=['arm', 'label', 'n_paired_seeds',
                                      'delta_vs_md', 'delta_sd', 'delta_se', 't'])
    agg = agg.merge(dd, on=['arm', 'label'], how='left')
    agg.to_csv(os.path.join(RES, f'{pre}_summary.csv'), index=False)
    print('\n' + agg.round(4).to_string(index=False), flush=True)
    print(f'\nsaved -> {RES}/{pre}_summary.csv (+ arms/seed_level/tables)',
          flush=True)
    return agg


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True,
                    choices=['train', 'eval', 'tables', 'all'])
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--only', nargs='*', default=None,
                    choices=[a[0] for a in ARMS])
    args = ap.parse_args()
    if args.mode in ('train', 'all'):
        train_mode(args.workers, args.only)
    if args.mode in ('tables', 'all'):
        table_meta()
    if args.mode in ('eval', 'all'):
        evaluate_all(args.only)
