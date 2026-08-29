#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Measure the held-out 2022 episode return of the trained PPO policy
under the EXACT protocol of tune_rules.py (alpha=0.1, train_years
2015-2021, eval_year=2022, episode noise seeds 0-9), so the numbers are
paired with the tuned-rule arms in Table 5.

Outputs: results/ppo_return_2022.csv
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Package root: parent of this code/ directory (override with MOSQ_DATA_DIR)
RL_PKG = os.environ.get('MOSQ_DATA_DIR', os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
MODELS = os.path.join(RL_PKG, 'models')
RES = os.path.join(HERE, '..', 'results')
os.makedirs(RES, exist_ok=True)

from mosquito_env import MosquitoEnv

ALPHA = 0.1
TRAIN_YEARS = tuple(range(2015, 2022))
N_SEEDS = 10


def ppo_policy_factory(model_name):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    model = PPO.load(os.path.join(MODELS, f'{model_name}.zip'))
    venv = VecNormalize.load(
        os.path.join(MODELS, f'{model_name}_vecnorm.pkl'),
        DummyVecEnv([lambda: MosquitoEnv()]))
    venv.training = False

    def policy(env):
        obs = env._obs(env.block[min(env.tpos, len(env.block) - 1)],
                       env.bi, env.bi_prev, env.p_sr, env.p_sp,
                       env.last_action)
        obs_n = venv.normalize_obs(obs.reshape(1, -1))
        action, _ = model.predict(obs_n, deterministic=True)
        return np.asarray(action).reshape(-1).astype(int)
    return policy


def eval_2022(policy_fn, seeds=range(N_SEEDS)):
    R, B, C, W = [], [], [], []
    for sd in seeds:
        env = MosquitoEnv(alpha=ALPHA, train_years=TRAIN_YEARS,
                          eval_year=2022, seed=sd, action_set=(0, 1, 2))
        obs, _ = env.reset(seed=sd)
        done, ret = False, 0.0
        bis, w10 = [], []
        while not done:
            a = policy_fn(env)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            ret += r
            bis.append(info['mean_bi'])
            w10.append(info['weeks_above10'])
        R.append(ret); B.append(np.mean(bis)); C.append(env.cost_total)
        W.append(np.mean(w10) * 100)
        print(f'  seed {sd}: return={ret:8.3f} BI={np.mean(bis):.2f} '
              f'cost={env.cost_total:.1f}', flush=True)
    sd_ = lambda x: float(np.std(x, ddof=1))
    return {'return': (float(np.mean(R)), sd_(R)),
            'mean_bi': (float(np.mean(B)), sd_(B)),
            'cost': (float(np.mean(C)), sd_(C)),
            'pct_weeks_ge10': (float(np.mean(W)), sd_(W))}


def main():
    for name in ['ppo_a0.1_s0_acts012', 'ppo_a0.1_s1_acts012',
                 'ppo_a0.1_s2_acts012']:
        print(f'== {name} ==', flush=True)
        r = eval_2022(ppo_policy_factory(name))
        row = {'arm': name,
               'return': round(r['return'][0], 3),
               'return_sd': round(r['return'][1], 3),
               'mean_bi': round(r['mean_bi'][0], 2),
               'mean_bi_sd': round(r['mean_bi'][1], 2),
               'cost': round(r['cost'][0], 1),
               'cost_sd': round(r['cost'][1], 1),
               'pct_weeks_ge10': round(r['pct_weeks_ge10'][0], 2)}
        df_path = os.path.join(RES, 'ppo_return_2022.csv')
        df = pd.DataFrame([row])
        if os.path.exists(df_path):
            df = pd.concat([pd.read_csv(df_path), df], ignore_index=True)
        df.to_csv(df_path, index=False)
        print(f'  -> return {r["return"][0]:.2f}±{r["return"][1]:.2f} | '
              f'BI {r["mean_bi"][0]:.2f}±{r["mean_bi"][1]:.2f} | '
              f'cost {r["cost"][0]:.1f}±{r["cost"][1]:.1f}', flush=True)
    print(f'saved -> {RES}/ppo_return_2022.csv', flush=True)


if __name__ == '__main__':
    main()
