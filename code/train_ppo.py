#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PPO / A2C / DQN training for MosquitoEnv (Stable-Baselines3).

Usage:
    python train_ppo.py --alpha 0.1 --seed 0 --steps 400000
    python train_ppo.py --alpha 0.1 --seed 0 --steps 400000 --action-set 0 1
    python train_ppo.py --algo ppo --C-max 6 --seed 3 --steps 400000
    python train_ppo.py --algo a2c --C-max 6 --seed 0 --steps 400000
    python train_ppo.py --algo dqn --C-max 2 --seed 0 --steps 200000

`--C-max` (weekly city-wide cost budget) defaults to None, in which case the
environment and the saved artefact are bit-identical to the previous version
(legacy tag `ppo_a{alpha}_s{seed}_acts{...}`).  With a finite budget the tag is
`{algo}_c{C}_s{seed}` so that no existing artefact can be overwritten.

Saves the model + VecNormalize statistics under ../models/.
"""
import os, argparse
# pin BLAS/torch threading before heavy imports to avoid thread-pool stalls
# during long CPU trainings
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import torch
torch.set_num_threads(1)

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from mosquito_env import MosquitoEnv

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.abspath(os.path.join(BASE, '..', 'models'))
LOGS = os.path.abspath(os.path.join(BASE, '..', 'results', 'monitor'))
os.makedirs(MODELS, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

ALGOS = {'ppo': PPO, 'a2c': A2C, 'dqn': DQN}


def c_tag(C_max):
    return 'inf' if C_max is None else str(C_max)


def make_env(alpha, seed, action_set, log_name, C_max=None,
             augment_adjacency=False, train_years=None, eval_year=None,
             action_table_size=None, subset_seed=0, subset_core=None,
             budget_projection=False):
    """`eval_year` is deliberately NOT passed to the training environment:
    setting `eval_year` makes the environment lock onto that year at every
    reset, which would leak the held-out year into training.  It is accepted
    here only for provenance/logging (leave-one-year-out runs)."""
    def _f():
        kw = {}
        if train_years is not None:
            kw['train_years'] = tuple(train_years)
        env = MosquitoEnv(alpha=alpha, seed=seed, action_set=tuple(action_set),
                          C_max=C_max, augment_adjacency=augment_adjacency,
                          action_table_size=action_table_size,
                          subset_seed=subset_seed, subset_core=subset_core,
                          budget_projection=budget_projection, **kw)
        return Monitor(env, os.path.join(LOGS, log_name))
    return _f


def tag(alpha, seed, action_set):
    return f'ppo_a{alpha}_s{seed}_acts{"".join(map(str, action_set))}'


def tag_c(algo, C_max, seed, extra=''):
    return f'{algo}_c{c_tag(C_max)}_s{seed}{extra}'


def build_model(algo, venv, seed):
    if algo == 'ppo':
        return PPO('MlpPolicy', venv, seed=seed, verbose=0,
                   n_steps=2048, batch_size=256, n_epochs=10,
                   learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                   clip_range=0.2, ent_coef=0.005,
                   policy_kwargs=dict(net_arch=[128, 128]))
    if algo == 'a2c':
        return A2C('MlpPolicy', venv, seed=seed, verbose=0,
                   n_steps=16, learning_rate=7e-4, gamma=0.99,
                   gae_lambda=0.95, ent_coef=0.005, vf_coef=0.5,
                   max_grad_norm=0.5, use_rms_prop=True,
                   policy_kwargs=dict(net_arch=[128, 128]))
    if algo == 'dqn':
        return DQN('MlpPolicy', venv, seed=seed, verbose=0,
                   learning_rate=1e-4, gamma=0.99, batch_size=256,
                   buffer_size=200_000, learning_starts=10_000,
                   train_freq=4, gradient_steps=1, target_update_interval=2_000,
                   exploration_fraction=0.3, exploration_final_eps=0.05,
                   policy_kwargs=dict(net_arch=[128, 128]))
    raise ValueError(algo)


def train(alpha=0.1, seed=0, steps=400_000, action_set=(0, 1, 2), C_max=None,
          augment_adjacency=False, algo='ppo', name=None, train_years=None,
          eval_year=None, action_table_size=None, subset_seed=0,
          subset_core=None, budget_projection=False):
    if name is None:
        if (C_max is None and not augment_adjacency and algo == 'ppo'
                and action_table_size is None and not budget_projection):
            name = tag(alpha, seed, action_set)          # legacy naming
        else:
            extra = '_adj' if augment_adjacency else ''
            if action_table_size is not None:
                extra += f'_k{action_table_size}'
            if subset_core is not None:
                extra += f'_core{subset_core}'
            if budget_projection:
                extra += '_proj'
            name = tag_c(algo, C_max, seed, extra)
    venv = DummyVecEnv([make_env(alpha, seed, action_set, name, C_max,
                                 augment_adjacency, train_years, eval_year,
                                 action_table_size, subset_seed, subset_core,
                                 budget_projection)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = build_model(algo, venv, seed)
    model.learn(total_timesteps=steps)
    model.save(os.path.join(MODELS, f'{name}.zip'))
    venv.save(os.path.join(MODELS, f'{name}_vecnorm.pkl'))
    print(f'saved {name}', flush=True)
    return name


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=400_000)
    ap.add_argument('--action-set', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--C-max', type=str, default=None,
                    help="weekly cost budget, 'inf'/omitted = unconstrained")
    ap.add_argument('--augment-adjacency', action='store_true')
    ap.add_argument('--algo', choices=sorted(ALGOS), default='ppo')
    ap.add_argument('--name', type=str, default=None)
    ap.add_argument('--train-years', type=int, nargs='+', default=None,
                    help='leave-one-year-out support: which years to train on')
    ap.add_argument('--eval-year', type=int, default=None,
                    help='leave-one-year-out support: the held-out year')
    ap.add_argument('--action-table-size', type=int, default=None,
                    help='K: fix the Discrete action table to K rows (k confound control)')
    ap.add_argument('--subset-seed', type=int, default=0,
                    help='seed of the fixed action-table subsample')
    ap.add_argument('--subset-core', type=float, default=None,
                    help='action-table control: always keep every action whose '
                         'weekly cost is <= this value, pad the rest randomly')
    ap.add_argument('--budget-projection', action='store_true',
                    help='keep the full joint space, project samples onto C_max')
    args = ap.parse_args()
    C = None if args.C_max in (None, 'None', 'none', 'inf') else float(args.C_max)
    if C is not None and float(C).is_integer():
        C = int(C)
    train(args.alpha, args.seed, args.steps, tuple(args.action_set), C,
          args.augment_adjacency, args.algo, args.name, args.train_years,
          args.eval_year, args.action_table_size, args.subset_seed,
          args.subset_core, args.budget_projection)
