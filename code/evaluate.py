#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Counterfactual evaluation on the held-out 2022 weather year.

Arms:
  A  PPO dynamic control (one or more trained seeds)
  B  static threshold rule (spray when BI >= 15)
  B2 preventive calendar rule
  C  no intervention

Every arm is replayed under the SAME set of episode noise seeds so that
comparisons are paired.  Metrics per episode: mean BI, peak BI, share of
grid-weeks with BI >= 10, total cost.  Averted-BI and cost-effectiveness
ratios are computed against the no-intervention arm.

Usage:
    python evaluate.py                      # default: all arms, 10 noise reps
    python evaluate.py --reps 10
"""
import os, argparse
import numpy as np
import pandas as pd

from mosquito_env import MosquitoEnv, GRIDS, N_GRIDS
import baselines

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.abspath(os.path.join(BASE, '..', 'models'))
RES = os.path.abspath(os.path.join(BASE, '..', 'results'))
os.makedirs(RES, exist_ok=True)
EVAL_YEAR = 2022


def evaluate_arm(make_env_fn, policy_fn, reps=10, record_last=False):
    """Run one arm over `reps` noise replicates of the eval year."""
    rows, trajs = [], []
    for rep in range(reps):
        env = make_env_fn()
        obs, _ = env.reset(seed=10_000 + rep)
        bis, costs, actions_sr, actions_sp = [], [], [], []
        done = False
        traj = []
        while not done:
            action = policy_fn(env)
            # policy outputs are indices into env.action_set; map to the
            # environment's action codes before counting / stepping
            mapped = np.array([env.action_set[int(np.clip(a, 0, len(env.action_set) - 1))]
                               for a in np.asarray(action).reshape(-1)], dtype=int)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            bis.append(env.bi.copy())
            costs.append(info['cost'])
            actions_sr.append(int(np.sum(mapped == 1)))
            actions_sp.append(int(np.sum(mapped == 2)))
            if record_last:
                traj.append({'week_in_year': int(env.weeks['week_in_year']
                                                 .values[env.block[env.tpos]]),
                             **{f'BI_{g}': float(env.bi[i])
                                for i, g in enumerate(GRIDS)},
                             'n_sr': actions_sr[-1], 'n_sp': actions_sp[-1],
                             'cost': info['cost']})
        B = np.array(bis)                      # (weeks, grids)
        rows.append({'rep': rep,
                     'mean_bi': float(B.mean()),
                     'peak_bi': float(B.max()),
                     'weeks_bi_ge10_pct': float((B >= 10).mean() * 100),
                     'weeks_bi_ge5_pct': float((B >= 5).mean() * 100),
                     'total_cost': float(np.sum(costs)),
                     'n_sr': int(np.sum(actions_sr)),
                     'n_sp': int(np.sum(actions_sp))})
        if record_last and rep == 0:
            trajs = traj
    return pd.DataFrame(rows), pd.DataFrame(trajs)


def parse_action_set(model_name):
    """model tag '..._acts012' -> (0,1,2)"""
    import re
    m = re.search(r'acts(\d+)$', model_name)
    return tuple(int(c) for c in m.group(1)) if m else (0, 1, 2)


def ppo_policy_factory(model_name):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    model_path = os.path.join(MODELS, f'{model_name}.zip')
    vec_path = os.path.join(MODELS, f'{model_name}_vecnorm.pkl')
    model = PPO.load(model_path)
    # rebuild a normalisation wrapper matching training stats for obs scaling
    venv = VecNormalize.load(vec_path, DummyVecEnv([lambda: MosquitoEnv()]))
    venv.training = False

    def policy(env):
        obs = env._obs(env.block[min(env.tpos, len(env.block) - 1)],
                       env.bi, env.bi_prev, env.p_sr, env.p_sp,
                       env.last_action)
        obs_n = venv.normalize_obs(obs.reshape(1, -1))
        action, _ = model.predict(obs_n, deterministic=True)
        return np.asarray(action).reshape(-1).astype(int)
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*',
                    default=['ppo_a0.1_s0_acts012',
                             'ppo_a0.1_s1_acts012',
                             'ppo_a0.1_s2_acts012'])
    ap.add_argument('--reps', type=int, default=10)
    args = ap.parse_args()

    def env_fn():
        return MosquitoEnv(eval_year=EVAL_YEAR)

    all_rows, traj_rows = [], {}
    for arm in ['none', 'threshold', 'calendar']:
        df, traj = evaluate_arm(env_fn, baselines.RULES[arm],
                                reps=args.reps, record_last=True)
        df['arm'] = arm
        all_rows.append(df)
        traj_rows[arm] = traj
        m = df.drop(columns=['rep', 'arm']).mean(numeric_only=True)
        print(f'  {arm:9s}: meanBI={m["mean_bi"]:.2f} '
              f'BI>=10%={m["weeks_bi_ge10_pct"]:.1f} cost={m["total_cost"]:.0f}',
              flush=True)
    for model_name in args.models:
        if not os.path.exists(os.path.join(MODELS, f'{model_name}.zip')):
            print(f'  [skip] model {model_name} not found', flush=True)
            continue
        acts = parse_action_set(model_name)
        env_fn_ppo = lambda: MosquitoEnv(eval_year=EVAL_YEAR, action_set=acts)
        policy = ppo_policy_factory(model_name)
        df, traj = evaluate_arm(env_fn_ppo, policy, reps=args.reps, record_last=True)
        df['arm'] = model_name
        all_rows.append(df)
        traj_rows[model_name] = traj
        m = df.drop(columns=['rep', 'arm']).mean(numeric_only=True)
        print(f'  {model_name}: meanBI={m["mean_bi"]:.2f} '
              f'BI>=10%={m["weeks_bi_ge10_pct"]:.1f} cost={m["total_cost"]:.0f}',
              flush=True)

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(os.path.join(RES, 'counterfactual_2022.csv'), index=False)
    for arm, traj in traj_rows.items():
        traj.to_csv(os.path.join(RES, f'trajectory_{arm}.csv'), index=False)

    # summary with averted BI and incremental cost per BI-unit averted
    summ = (out.groupby('arm')
            .agg(mean_bi=('mean_bi', 'mean'), mean_bi_sd=('mean_bi', 'std'),
                 cost=('total_cost', 'mean'), cost_sd=('total_cost', 'std'),
                 weeks10=('weeks_bi_ge10_pct', 'mean'),
                 weeks10_sd=('weeks_bi_ge10_pct', 'std'))
            .reset_index())
    base = summ.loc[summ['arm'] == 'none', 'mean_bi'].iloc[0]
    summ['bi_averted'] = base - summ['mean_bi']
    summ['cost_per_bi_averted'] = np.where(
        summ['bi_averted'] > 1e-9, summ['cost'] / summ['bi_averted'], np.nan)
    summ.to_csv(os.path.join(RES, 'counterfactual_summary.csv'), index=False)
    print('\n', summ.round(3).to_string(index=False), flush=True)
    print(f'\nsaved -> {RES}', flush=True)


if __name__ == '__main__':
    main()
