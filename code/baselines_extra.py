#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Additional baselines used in this study.

(a) A SECOND reinforcement-learning algorithm
    * A2C  (stable-baselines3, native support for the MultiDiscrete action
      space of the unconstrained environment and for the Discrete space of the
      capacity-constrained environment).
    * DQN  (only meaningful once the joint action space is flattened to
      Discrete; the unconstrained space is 3**10 = 59049, which is reported
      explicitly as a computational limitation).

(b) A ONE-STEP LOOKAHEAD / EXHAUSTIVE-ALLOCATION baseline (= degenerate MPC)
    At every week we enumerate the action vectors that satisfy
    sum_g cost_g <= C_max, evaluate the immediate reward of each with the
    environment's own transition equations, and execute the argmax.  With no
    budget the enumeration is the full 3**10 = 59049 joint action space.

All non-learning arms are capacity-aware: their nominal action vector is
projected onto the weekly budget with a deterministic greedy rule
(`MosquitoEnv.project_to_budget`) so that every arm faces the same physical
constraint.  Learning arms do not need this: for C_max < inf their action
space contains feasible vectors only.
"""
import itertools
import os
import time

import numpy as np

from mosquito_env import MosquitoEnv, N_GRIDS

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.abspath(os.path.join(BASE, '..', 'models'))

_FULL_TABLE_CACHE = {}


def full_action_table(action_set=(0, 1, 2)):
    """All |action_set|^10 joint action vectors (cached).  At most 59049."""
    key = tuple(action_set)
    if key not in _FULL_TABLE_CACHE:
        _FULL_TABLE_CACHE[key] = np.asarray(
            list(itertools.product(key, repeat=N_GRIDS)), dtype=int)
    return _FULL_TABLE_CACHE[key]


def candidate_table(env):
    """Action vectors the MPC baseline may choose from this week."""
    if env.feasible_actions is not None:
        return env.feasible_actions
    return full_action_table(env.action_set)


# --------------------------------------------------------------------- (b)
def mpc_one_step(env):
    """Exhaustive one-step-lookahead allocation under the weekly budget."""
    cands = candidate_table(env)
    rewards, _ = env.peek_rewards(cands)
    return cands[int(np.argmax(rewards))].copy()


def mpc_timing(env, n_repeat=5, note=''):
    """Time a SINGLE full enumeration (both naive loop and vectorised)."""
    cands = candidate_table(env)
    # vectorised
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        env.peek_rewards(cands)
    t_vec = (time.perf_counter() - t0) / n_repeat
    # naive python loop over the same candidates, using the transition maths
    p = env.params
    t0 = time.perf_counter()
    worst = None
    for a in cands[:min(len(cands), 5000)]:
        n_sr = p['sr_decay'] * env.p_sr + p['sr_efficacy'] * (a == 1)
        n_sp = p['sp_decay'] * env.p_sp + p['sp_efficacy'] * (a == 2)
        _ = n_sr, n_sp
    t_naive_5000 = time.perf_counter() - t0
    return {'note': note, 'n_candidates': int(len(cands)),
            'vectorised_sec_per_step': t_vec,
            'naive_sec_per_5000': t_naive_5000,
            'naive_sec_per_full_enum_est': t_naive_5000 * len(cands) / 5000.0}


# ------------------------------------------------- capacity-aware rule wrap
def capacity_aware(rule_fn):
    """Wrap a nominal rule so its output respects the weekly budget."""
    def policy(env):
        nominal = np.asarray(rule_fn(env), dtype=int)
        if env.C_max is None:
            return nominal
        codes = env.action_codes[np.clip(nominal, 0, len(env.action_set) - 1)]
        return env.project_to_budget(codes)
    return policy


def _capacity_aware_codes(env, codes):
    if env.C_max is None:
        return codes
    return env.project_to_budget(codes)


# --------------------------------------------------------------------- (a)
def _vecnorm_path(name):
    return os.path.join(MODELS, f'{name}_vecnorm.pkl')


def sb3_policy_factory(model_name, algo='ppo', env_kwargs=None):
    """Load a trained sb3 model and return a policy(env) -> action CODES.

    `env_kwargs` must reproduce the environment the model was trained in
    (C_max / action_set / augment_adjacency), because the observation vector
    and the meaning of the action index depend on it.
    """
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    algo_cls = {'ppo': PPO, 'a2c': A2C, 'dqn': DQN}[algo]
    model = algo_cls.load(os.path.join(MODELS, f'{model_name}.zip'))
    env_kwargs = dict(env_kwargs or {})
    npz = _vecnorm_path(model_name)

    if algo == 'dqn' or not os.path.exists(npz):
        def policy(env):
            obs = env._obs(env.block[min(env.tpos, len(env.block) - 1)],
                           env.bi, env.bi_prev, env.p_sr, env.p_sp,
                           env.last_action)
            a, _ = model.predict(obs.reshape(1, -1), deterministic=True)
            return _decode(env, a)
        return policy

    venv = VecNormalize.load(npz, DummyVecEnv([lambda: MosquitoEnv(**env_kwargs)]))
    venv.training = False

    def policy(env):
        obs = env._obs(env.block[min(env.tpos, len(env.block) - 1)],
                       env.bi, env.bi_prev, env.p_sr, env.p_sp,
                       env.last_action)
        obs_n = venv.normalize_obs(obs.reshape(1, -1))
        a, _ = model.predict(obs_n, deterministic=True)
        return _decode(env, a)
    return policy


def _decode(env, act):
    """Raw sb3 action -> per-grid action CODES.

    Three cases, decided by the environment the model was TRAINED in:
      * budgeted / flattened: a single integer indexing the feasible table;
      * legacy MultiDiscrete: a length-N_GRIDS vector of action-set indices.
    """
    a = np.asarray(act).ravel().astype(int)
    n_act = len(env.action_set)
    if env.feasible_actions is not None:
        k = len(env.feasible_actions)
        return env.feasible_actions[int(np.clip(a[0], 0, k - 1))].copy()
    if a.size == 1:
        return env.action_codes[int(np.clip(a[0], 0, n_act - 1))].copy()
    return env.action_codes[np.clip(a, 0, n_act - 1)].copy()


# --------------------------------------------------------------------------
# Policy adapters for the rule baselines (return CODES, capacity-projected).
def make_rule_policy(rule_fn):
    return capacity_aware(rule_fn)
