#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PPO training for MosquitoEnv (Stable-Baselines3).

Usage:
    python train_ppo.py --alpha 0.1 --seed 0 --steps 400000
    python train_ppo.py --alpha 0.1 --seed 0 --steps 400000 --action-set 0 1

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

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from mosquito_env import MosquitoEnv

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.abspath(os.path.join(BASE, '..', 'models'))
LOGS = os.path.abspath(os.path.join(BASE, '..', 'results', 'monitor'))
os.makedirs(MODELS, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)


def make_env(alpha, seed, action_set, log_name):
    def _f():
        env = MosquitoEnv(alpha=alpha, seed=seed, action_set=tuple(action_set))
        return Monitor(env, os.path.join(LOGS, log_name))
    return _f


def tag(alpha, seed, action_set):
    return f'ppo_a{alpha}_s{seed}_acts{"".join(map(str, action_set))}'


def train(alpha=0.1, seed=0, steps=400_000, action_set=(0, 1, 2)):
    name = tag(alpha, seed, action_set)
    venv = DummyVecEnv([make_env(alpha, seed, action_set, name)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO('MlpPolicy', venv, seed=seed, verbose=0,
                n_steps=2048, batch_size=256, n_epochs=10,
                learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.005,
                policy_kwargs=dict(net_arch=[128, 128]))
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
    args = ap.parse_args()
    train(args.alpha, args.seed, args.steps, tuple(args.action_set))
