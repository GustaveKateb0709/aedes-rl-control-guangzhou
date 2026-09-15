#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Spatial-structure analysis of the multi-zone control problem.

PART A -- exact decomposability of the unconstrained problem
------------------------------------------------------------
With C_max = None the environment is per-grid independent (`_base_bi` is
computed grid by grid; there is no cross-grid term anywhere) and the reward
is a sum of per-grid terms:

    R = sum_t [ -(1/N) sum_g (e_g/5)^2 - alpha * cost_total / N ]
      = sum_g sum_t [ -(e_g/5)^2 / N - alpha * cost_g / N ]
      = sum_g R_g .

Three numerical checks are run on the held-out 2022 year:

  A1  independent-simulator identity: replay random action sequences in the
      joint environment and in an independently coded per-grid simulator that
      shares the same noise realisation -> R_joint == sum_g R_g.
  A2  exact optimality: solve each grid's single-grid optimal control problem
      EXACTLY by dynamic programming over the reachable protection stocks
      (the base-BI sequence is deterministic once the noise realisation is
      fixed), giving V_g*.  Then sum_g V_g* is an upper bound on the joint
      optimum, and replaying the concatenated per-grid optimal actions in the
      joint environment attains it.  Therefore the joint optimum equals the
      per-grid optimum -- the joint MDP is exactly 10 decoupled MDPs.
  A3  one-step-lookahead decomposability: the exhaustive joint argmax over
      all 3^10 candidate vectors coincides (in value) with the per-grid
      argmax; with a finite C_max it does not, and the gap appears only in
      the constrained regime.

PART B -- adjacency-feature ablation
------------------------------------
Retrain PPO with the administrative-neighbour features added to the
observation (`augment_adjacency=True`) and compare on 2022 against the
matched baseline trained without them.

Usage:
  python spatial_ablation.py --mode decompose
  python spatial_ablation.py --mode train-adjac [--workers 3]
  python spatial_ablation.py --mode eval-adjac
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

from mosquito_env import (MosquitoEnv, GRIDS, N_GRIDS, ADJ_IDX,       # noqa: E402
                          BI_THRESHOLD, REWARD_SCALE, INTERVENTION_PARAMS)
import baselines_extra as bx                                          # noqa: E402

RES = os.path.abspath(os.path.join(HERE, '..', 'results'))
MODELS = os.path.abspath(os.path.join(HERE, '..', 'models'))
ALPHA = 0.1
EVAL_YEAR = 2022
EVAL_SEEDS = [10_000 + r for r in range(10)]
PY = sys.executable


# --------------------------------------------------------------- per-grid sim
def base_sequence(seed, years=(2015, 2016, 2017, 2018, 2019, 2020, 2021),
                  eval_year=EVAL_YEAR):
    """Realised 2022 baseline-BI sequence for one noise realisation.

    With the all-zero action the protection stocks stay at 0 and the observed
    BI equals the baseline, so replaying the no-intervention arm pins down the
    exact noise draws that any other action sequence will see.
    """
    env = MosquitoEnv(alpha=ALPHA, eval_year=eval_year, seed=seed)
    env.reset(seed=seed)
    seq = [env.bi.copy()]
    done = False
    while not done:
        _, _, te, tr, _ = env.step(np.zeros(N_GRIDS, dtype=int))
        done = te or tr
        seq.append(env.bi.copy())
    return np.asarray(seq)          # (L + 1, N_GRIDS)


def simulate_grid(b_g, actions, params, alpha=ALPHA):
    """Independent per-grid simulator, written from the environment's own
    equations.  b_g: baseline BI at weeks 0..L; actions: codes at weeks 0..L-1."""
    p = params
    p_sr = p_sp = 0.0
    total = 0.0
    for k, a in enumerate(actions):
        if a == 1:
            p_sr = min(0.95, p_sr * p['sr_decay'] + p['sr_efficacy'])
        else:
            p_sr = min(0.95, p_sr * p['sr_decay'])
        if a == 2:
            p_sp = min(0.95, p_sp * p['sp_decay'] + p['sp_efficacy'])
        else:
            p_sp = min(0.95, p_sp * p['sp_decay'])
        bi = b_g[k + 1] * (1.0 - p_sr) * (1.0 - p_sp)
        cost = p['sr_cost'] if a == 1 else (p['sp_cost'] if a == 2 else 0.0)
        total += -(max(0.0, bi - BI_THRESHOLD) / REWARD_SCALE) ** 2 / N_GRIDS \
            - alpha * cost / N_GRIDS
    return total


def _joint_return_from(base_seq, action_seq):
    """Vectorised replay of the environment's own transition and reward
    equations for a fixed action sequence."""
    p = INTERVENTION_PARAMS
    p_sr = np.zeros(N_GRIDS)
    p_sp = np.zeros(N_GRIDS)
    total = 0.0
    for k, a in enumerate(np.asarray(action_seq, dtype=int)):
        p_sr = np.clip(p_sr * p['sr_decay'] + p['sr_efficacy'] * (a == 1), 0, 0.95)
        p_sp = np.clip(p_sp * p['sp_decay'] + p['sp_efficacy'] * (a == 2), 0, 0.95)
        bi = base_seq[k + 1] * (1.0 - p_sr) * (1.0 - p_sp)
        cost = float(np.sum(a == 1) * p['sr_cost'] + np.sum(a == 2) * p['sp_cost'])
        e = np.maximum(bi - BI_THRESHOLD, 0.0)
        total += -float(np.mean((e / REWARD_SCALE) ** 2)) - ALPHA * cost / N_GRIDS
    return total


# ------------------------------------------------------- per-grid DP
def dp_grid_grid(b_g, params, alpha=ALPHA, delta=0.002):
    """Finite-horizon DP for ONE grid on a uniform grid of the protection
    stocks (p_sr, p_sp) in [0, 0.95].

    The reachable state set of the exact recurrence is exponential in the
    horizon (p_sr = eff * sum_j decay^j over the SR weeks), so an exact DP on
    the reachable set is not tractable; we therefore discretise the two
    protection stocks with step `delta` and verify convergence over delta in
    `decompose()`.  The policy returned by the DP is then rolled out with the
    EXACT (non-discretised) dynamics, so the reported per-grid return is an
    exact lower bound on the true per-grid optimum.
    """
    L = len(b_g) - 1
    M = int(round(0.95 / delta)) + 1
    grid = np.arange(M) * delta
    grid[-1] = 0.95
    p = params

    def idx(x):
        return np.clip(np.rint(np.asarray(x) / delta).astype(int), 0, M - 1)

    # successor index maps per action (p_sr depends only on p_sr, same for sp)
    succ = {}
    for a in (0, 1, 2):
        n_sr = np.clip(grid * p['sr_decay']
                       + (p['sr_efficacy'] if a == 1 else 0.0), 0, 0.95)
        n_sp = np.clip(grid * p['sp_decay']
                       + (p['sp_efficacy'] if a == 2 else 0.0), 0, 0.95)
        succ[a] = (idx(n_sr), idx(n_sp), n_sr, n_sp)

    V = np.zeros((M, M))
    pol = np.zeros((L, M, M), dtype=np.int8)
    for k in range(L - 1, -1, -1):
        best = np.full((M, M), -np.inf)
        best_a = np.zeros((M, M), dtype=np.int8)
        for a in (0, 1, 2):
            i_sr, i_sp, n_sr, n_sp = succ[a]
            bi = b_g[k + 1] * (1.0 - n_sr)[:, None] * (1.0 - n_sp)[None, :]
            cost = (p['sr_cost'] if a == 1 else
                    (p['sp_cost'] if a == 2 else 0.0))
            r = -(np.maximum(bi - BI_THRESHOLD, 0.0) / REWARD_SCALE) ** 2 / N_GRIDS \
                - alpha * cost / N_GRIDS
            val = r + V[np.ix_(i_sr, i_sp)]
            m = val > best
            best = np.where(m, val, best)
            best_a = np.where(m, np.int8(a), best_a)
        V = best
        pol[k] = best_a
    return grid, float(V[0, 0]), pol


def rollout_grid(b_g, pol, grid, params, alpha=ALPHA):
    """Exact-dynamics rollout of a DP policy for one grid -> true return."""
    delta = grid[1] - grid[0]
    L = len(b_g) - 1
    p = params
    p_sr = p_sp = 0.0
    total = 0.0
    acts = []
    for k in range(L):
        i = int(np.clip(round(p_sr / delta), 0, len(grid) - 1))
        j = int(np.clip(round(p_sp / delta), 0, len(grid) - 1))
        a = int(pol[k, i, j])
        acts.append(a)
        if a == 1:
            p_sr = min(0.95, p_sr * p['sr_decay'] + p['sr_efficacy'])
        else:
            p_sr = min(0.95, p_sr * p['sr_decay'])
        if a == 2:
            p_sp = min(0.95, p_sp * p['sp_decay'] + p['sp_efficacy'])
        else:
            p_sp = min(0.95, p_sp * p['sp_decay'])
        bi = b_g[k + 1] * (1.0 - p_sr) * (1.0 - p_sp)
        cost = p['sr_cost'] if a == 1 else (p['sp_cost'] if a == 2 else 0.0)
        total += -(max(0.0, bi - BI_THRESHOLD) / REWARD_SCALE) ** 2 / N_GRIDS \
            - alpha * cost / N_GRIDS
    return total, np.asarray(acts, dtype=int)


def per_grid_optimum(b_g, params, alpha=ALPHA, deltas=(0.005, 0.002)):
    """Per-grid optimum with a discretisation-convergence check."""
    best = None
    conv = []
    for d in deltas:
        grid, v_dp, pol = dp_grid_grid(b_g, params, alpha, d)
        v_true, acts = rollout_grid(b_g, pol, grid, params, alpha)
        conv.append({'delta': d, 'dp_value': v_dp, 'true_return_of_policy': v_true,
                     'n_sr': int((acts == 1).sum()), 'n_sp': int((acts == 2).sum())})
        if best is None or v_true > best[0]:
            best = (v_true, acts, d)
    return best, conv


# ------------------------------------------------------------------- Part A
def decompose(seeds=EVAL_SEEDS[:3], n_random=20, verbose=True):
    rows, conv_rows, cap_rows = [], [], []
    rng = np.random.default_rng(0)
    for sd in seeds:
        bs = base_sequence(sd)
        L = len(bs) - 1
        # ---- A1: independent-simulator identity --------------------------
        worst_a1 = 0.0
        for _ in range(n_random):
            acts = rng.integers(0, 3, size=(L, N_GRIDS))
            rj = _joint_return_from(bs, acts)
            rs = sum(simulate_grid(bs[:, g], acts[:, g], INTERVENTION_PARAMS)
                     for g in range(N_GRIDS))
            worst_a1 = max(worst_a1, abs(rj - rs))
        # ---- A2: per-grid optimum vs joint optimum -----------------------
        Vsum, opt_acts, per_grid_true = 0.0, [], []
        for g in range(N_GRIDS):
            (vt, ag, d), conv = per_grid_optimum(bs[:, g], INTERVENTION_PARAMS)
            Vsum += vt
            opt_acts.append(ag)
            per_grid_true.append(vt)
            for c in conv:
                conv_rows.append({'eval_seed': sd, 'grid': GRIDS[g], **c,
                                  'chosen_delta': d})
        opt_acts = np.stack(opt_acts, axis=1)          # (L, N_GRIDS)
        r_joint_of_pergrid_opt = _joint_return_from(bs, opt_acts)
        r_sum_grid = sum(simulate_grid(bs[:, g], opt_acts[:, g],
                                       INTERVENTION_PARAMS)
                         for g in range(N_GRIDS))
        # ---- A5: replay the same action sequence through the REAL env -----
        # (guards against the closed-form replay above disagreeing with the
        # environment itself, which would make A1/A2 vacuous)
        env_real = MosquitoEnv(alpha=ALPHA, eval_year=EVAL_YEAR, C_max=None)
        env_real.reset(seed=sd)
        tot_real = 0.0
        for k in range(L):
            _, r, te, tr, _ = env_real.step(opt_acts[k])
            tot_real += r
            if te or tr:
                break
        # ---- A3: joint one-step argmax vs per-grid one-step argmax -------
        env = MosquitoEnv(alpha=ALPHA, eval_year=EVAL_YEAR, C_max=None)
        env.reset(seed=sd)
        cands = bx.full_action_table()
        rew, _ = env.peek_rewards(cands)
        jbest = float(rew.max())
        per_grid_best = np.zeros(N_GRIDS, dtype=int)
        for g in range(N_GRIDS):
            mask = (cands[:, [i for i in range(N_GRIDS) if i != g]] == 0).all(axis=1)
            cg = cands[mask]
            rg, _ = env.peek_rewards(cg)
            per_grid_best[g] = cg[int(np.argmax(rg))][g]
        val_pg = float(rew[np.ravel_multi_index(per_grid_best, (3,) * N_GRIDS)])
        # joint one-step-greedy controller, full horizon, unfettered space
        env_g = MosquitoEnv(alpha=ALPHA, eval_year=EVAL_YEAR, C_max=None)
        env_g.reset(seed=sd)
        acts_g, done = [], False
        while not done:
            a = bx.mpc_one_step(env_g)
            acts_g.append(a)
            _, _, te, tr, _ = env_g.step(a)
            done = te or tr
        r_mpc = _joint_return_from(bs, np.asarray(acts_g, dtype=int))
        rows.append({'eval_seed': sd,
                     'A1_max_abs_diff_joint_vs_pergrid': worst_a1,
                     'A2_pergrid_optimum_sum': Vsum,
                     'A2_joint_return_of_pergrid_optima': r_joint_of_pergrid_opt,
                     'A2_sum_pergrid_of_same_actions': r_sum_grid,
                     'A2_gap_joint_minus_pergrid_sum': r_joint_of_pergrid_opt - Vsum,
                     'A3_joint_onestep_max_reward': jbest,
                     'A3_pergrid_onestep_reward': val_pg,
                     'A3_gap_joint_minus_pergrid': jbest - val_pg,
                     'A4_joint_mpc1_horizon_return': r_mpc,
                     'A4_pergrid_optimum_minus_mpc1': Vsum - r_mpc,
                     'A5_env_realised_return': tot_real,
                     'A5_replay_minus_env': r_joint_of_pergrid_opt - tot_real})
        if verbose:
            print(f'[A] seed={sd}: A1 max|diff|={worst_a1:.3e}  '
                  f'A2 joint(per-grid optima)={r_joint_of_pergrid_opt:.10f} vs '
                  f'sum V_g*={Vsum:.10f} (gap {r_joint_of_pergrid_opt-Vsum:.3e})  '
                  f'A3 gap={jbest-val_pg:.3e}  A4 mpc1={r_mpc:.4f}  '
                  f'A5 env-realised={tot_real:.10f} '
                  f'(replay-env={r_joint_of_pergrid_opt-tot_real:.2e})', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, 'spatial_decomposability.csv'), index=False)
    pd.DataFrame(conv_rows).to_csv(
        os.path.join(RES, 'spatial_decomposability_dp_convergence.csv'), index=False)

    # ---- the same comparison with a BINDING budget ----------------------
    sd0 = seeds[0]
    bs0 = base_sequence(sd0)
    L0 = len(bs0) - 1
    opt_acts0 = np.stack(
        [per_grid_optimum(bs0[:, g], INTERVENTION_PARAMS)[0][1]
         for g in range(N_GRIDS)], axis=1)              # (L0, N_GRIDS)
    p0 = INTERVENTION_PARAMS
    wk_cost = ((opt_acts0 == 1).sum(axis=1) * p0['sr_cost']
               + (opt_acts0 == 2).sum(axis=1) * p0['sp_cost'])
    pd.DataFrame([{'eval_seed': sd0,
                   'min_weekly_cost': float(wk_cost.min()),
                   'mean_weekly_cost': float(wk_cost.mean()),
                   'p95_weekly_cost': float(np.percentile(wk_cost, 95)),
                   'max_weekly_cost': float(wk_cost.max()),
                   'weeks': int(L0),
                   'n_sr_total': int((opt_acts0 == 1).sum()),
                   'n_sp_total': int((opt_acts0 == 2).sum())}]).to_csv(
        os.path.join(RES, 'spatial_budget_demand.csv'), index=False)
    print(f'[A] weekly budget demanded by the UNCONSTRAINED per-grid optimum: '
          f'mean={wk_cost.mean():.2f} max={wk_cost.max():.0f} (cost units/week)',
          flush=True)

    for C in [2, 4, 6, 8, 12]:
        sd, bs = sd0, bs0
        L = L0
        Vsum = sum(per_grid_optimum(bs[:, g], INTERVENTION_PARAMS)[0][0]
                   for g in range(N_GRIDS))
        env = MosquitoEnv(alpha=ALPHA, eval_year=EVAL_YEAR, C_max=C)
        env.reset(seed=sd)
        acts, done = [], False
        while not done:
            a = bx.mpc_one_step(env)
            acts.append(a)
            _, _, te, tr, _ = env.step(a)
            done = te or tr
        r_mpc = _joint_return_from(bs, np.asarray(acts, dtype=int))
        opt_acts = opt_acts0
        env2 = MosquitoEnv(alpha=ALPHA, eval_year=EVAL_YEAR, C_max=C)
        env2.reset(seed=sd)
        pa = np.zeros((L, N_GRIDS), dtype=int)
        for k in range(L):
            pa[k] = env2.project_to_budget(opt_acts[k], priority=env2.bi)
            env2.step(pa[k])
        r_proj = _joint_return_from(bs, pa)
        cap_rows.append({'C_max': C, 'unconstrained_pergrid_optimum': Vsum,
                         'joint_mpc1': r_mpc,
                         'pergrid_opt_projected': r_proj,
                         'mpc1_gap_to_unconstrained': r_mpc - Vsum,
                         'pct_of_unconstrained_optimum_achieved':
                             100.0 * r_mpc / Vsum if Vsum else np.nan})
        print(f'[A/C_max={C}] unconstrained sum V_g*={Vsum:.4f} '
              f'joint mpc1={r_mpc:.4f} projected-pergrid={r_proj:.4f} '
              f'gap={r_mpc-Vsum:.4f} '
              f'({100*(1-r_mpc/Vsum):.1f}% of the unconstrained optimum lost)',
              flush=True)
    dc = pd.DataFrame(cap_rows)
    dc.to_csv(os.path.join(RES, 'spatial_decomposability_capacity.csv'), index=False)
    return df, dc


# ------------------------------------------------------------------- Part B
# Regimes for the adjacency ablation.  Deliberately restricted to the two
# settings whose action parameterisation is NOT confounded with the size of
# the flat feasible-action table: C_max = inf (legacy MultiDiscrete, factored
# head) and C_max = 2 (K = 56).  At C_max = 6/8 the flat Discrete(2193/6498)
# head itself degrades PPO performance (see the capacity sweep), so an
# adjacency effect measured there could not be separated from that artefact.
ADJ_CMAX = [None, 2]
ADJ_SEEDS = list(range(5))


def adjac_name(C, seed):
    ctok = 'inf' if C is None else str(C)
    return f'ppo_c{ctok}_s{seed}_adj'


def baseline_name(C, seed):
    ctok = 'inf' if C is None else str(C)
    return f'ppo_c{ctok}_s{seed}'


def train_adjac(workers=3, steps=400_000):
    jobs = []
    for C in ADJ_CMAX:
        for s in ADJ_SEEDS:
            name = adjac_name(C, s)
            if os.path.exists(os.path.join(MODELS, f'{name}.zip')):
                continue
            jobs.append((C, s, name))
    print(f'[adjac] {len(jobs)} trainings to run', flush=True)
    running = []
    q = list(jobs)
    while q or running:
        while q and len(running) < workers:
            C, s, name = q.pop(0)
            cmd = [PY, os.path.join(HERE, 'train_ppo.py'), '--algo', 'ppo',
                   '--seed', str(s), '--steps', str(steps), '--name', name,
                   '--augment-adjacency']
            if C is not None:
                cmd += ['--C-max', str(C)]
            log = os.path.join(RES, 'monitor', f'{name}.train.log')
            fh = open(log, 'w')
            p = subprocess.Popen(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
            running.append((p, name, fh))
            print(f'[adjac] start {name}', flush=True)
        time.sleep(2)
        for it in list(running):
            p, name, fh = it
            if p.poll() is not None:
                fh.close()
                print(f'[adjac] done {name} rc={p.returncode}', flush=True)
                running.remove(it)
    print('[adjac] all done', flush=True)


def eval_adjac(reps=10):
    rows = []
    for C in ADJ_CMAX:
        ctok = 'inf' if C is None else str(C)
        for s in ADJ_SEEDS:
            for tag, name in [('baseline', baseline_name(C, s)),
                              ('adjacency', adjac_name(C, s))]:
                if not os.path.exists(os.path.join(MODELS, f'{name}.zip')):
                    print(f'[adjac-eval] missing {name}', flush=True)
                    continue
                pol = bx.sb3_policy_factory(name, algo='ppo',
                                            env_kwargs={'C_max': C,
                                                        'alpha': ALPHA,
                                                        'augment_adjacency':
                                                            tag == 'adjacency'})
                env = MosquitoEnv(alpha=ALPHA, eval_year=EVAL_YEAR, C_max=C,
                                  augment_adjacency=(tag == 'adjacency'))
                rets, bis, costs, w10 = [], [], [], []
                for rep, sd in enumerate(EVAL_SEEDS[:reps]):
                    env.reset(seed=sd)
                    done, ret, B, Cst, W = False, 0.0, [], [], []
                    while not done:
                        a = pol(env)
                        _, r, te, tr, info = env.step(a)
                        done = te or tr
                        ret += r
                        B.append(info['mean_bi'])
                        Cst.append(info['cost'])
                        W.append(info['weeks_above10'])
                    rets.append(ret)
                    bis.append(np.mean(B))
                    costs.append(np.sum(Cst))
                    w10.append(np.mean(W) * 100)
                rows.append({'C_max': ctok, 'seed': s, 'features': tag,
                             'model': name,
                             'mean_return': float(np.mean(rets)),
                             'mean_return_sd': float(np.std(rets, ddof=1)),
                             'mean_bi': float(np.mean(bis)),
                             'total_cost': float(np.mean(costs)),
                             'weeks_bi_ge10_pct': float(np.mean(w10))})
                print(f'[adjac-eval] {name:22s} return={rows[-1]["mean_return"]:.3f} '
                      f'BI={rows[-1]["mean_bi"]:.3f}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, 'spatial_adjacency_ablation.csv'), index=False)
    if len(df):
        agg = (df.groupby(['C_max', 'features'])
               .agg(n=('mean_return', 'size'),
                    ret=('mean_return', 'mean'), ret_sd=('mean_return', 'std'),
                    bi=('mean_bi', 'mean'), cost=('total_cost', 'mean'),
                    w10=('weeks_bi_ge10_pct', 'mean')).reset_index())
        agg.to_csv(os.path.join(RES, 'spatial_adjacency_summary.csv'), index=False)
        print('\n', agg.round(3).to_string(index=False), flush=True)
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True,
                    choices=['decompose', 'train-adjac', 'eval-adjac', 'all'])
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--seeds', type=int, nargs='*', default=None)
    args = ap.parse_args()
    if args.mode in ('decompose', 'all'):
        decompose(seeds=args.seeds or EVAL_SEEDS[:3])
    if args.mode in ('train-adjac', 'all'):
        train_adjac(args.workers)
    if args.mode in ('eval-adjac', 'all'):
        eval_adjac()
