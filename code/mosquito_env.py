#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MosquitoEnv: a Gymnasium environment for dynamic, spatially explicit
mosquito-control resource allocation.
======================================================================
The environment builds on a literature-parameterized mechanistic layer:
the weekly baseline Breteau index (BI) of each of the 10 Guangzhou grids
is driven by real ERA5 reanalysis weather through the PFI model
(BI_base = max(0, PFI * pop_scale + eps), eps ~ N(0, 1.5^2) re-sampled
every episode; the precomputed panel is shipped under results/panel_data.csv).

Interventions act on top of this baseline through literature-anchored,
explicitly parametrised effect processes (see INTERVENTION_PARAMS):

  * source reduction (action 1): suppresses aquatic-stage recruitment;
    moderate efficacy, slow onset (1 week), sustained (slow decay);
  * space spraying (action 2): kills adults; higher efficacy, transient
    (fast decay).

Both effects enter as multiplicative protection stocks with a one-week
onset lag and exponential decay, i.e. action at week t modifies BI from
week t+1 onwards:  BI(t+1) = BI_base(t+1) * (1 - P_sr) * (1 - P_sp).

Reward: r = -(1/N) * sum_g [ max(0, BI_g - BI_threshold) / S ]^2  - alpha * cost / N
with BI_threshold = 5 (national guideline transmission-risk threshold) and
S a scale constant.  alpha trades off health benefit vs. control cost and
is subjected to a dedicated sensitivity sweep (sensitivity.py).

Design note: alpha is NOT presented as universal; the effect parameters
carry explicit ranges and are perturbed in the robustness analysis; all
conclusions are framed as simulation-based.
"""
import os
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    raise ImportError("pip install gymnasium numpy") from e

BASE = os.path.dirname(os.path.abspath(__file__))


def _find_data_dir():
    """Locate the directory that holds results/panel_data.csv.

    Search order: $MOSQ_DATA_DIR, the package root (parent of code/).
    """
    cands = [os.environ.get('MOSQ_DATA_DIR', ''),
             os.path.abspath(os.path.join(BASE, '..'))]
    for c in cands:
        if c and os.path.exists(os.path.join(c, 'results', 'panel_data.csv')):
            return c
    raise FileNotFoundError('panel_data.csv not found; set MOSQ_DATA_DIR')


DATA_DIR = _find_data_dir()

GRIDS = ['conghua', 'huadu', 'zengcheng', 'baiyun', 'tianhe',
         'yuexiu', 'haizhu', 'panyu', 'nansha', 'huangpu']
N_GRIDS = len(GRIDS)

# ------------------------- intervention model -------------------------------
# effect parameters with literature-plausible ranges (perturbed in the
# robustness analysis; anchors documented in the manuscript).
INTERVENTION_PARAMS = {
    # source reduction (breeding-site elimination)
    'sr_efficacy': 0.35,     # peak proportional BI reduction (range 0.20-0.50)
    'sr_decay': 0.85,        # weekly retention of protection (half-life ~4 wk)
    'sr_cost': 1.0,          # relative cost units per grid-week
    # space spraying (e.g. ULV adulticiding)
    'sp_efficacy': 0.55,     # peak proportional BI reduction (range 0.30-0.70)
    'sp_decay': 0.55,        # weekly retention (half-life ~1.2 wk)
    'sp_cost': 3.0,          # relative cost units per grid-week
}

BI_THRESHOLD = 5.0           # national guideline: BI >= 5 implies transmission risk
CALIB_SCALE = 0.3476         # surveillance calibration: scales BI to published Guangzhou citywide means (2.90-4.50, 2019-2021); phase already matches
REWARD_SCALE = 5.0           # keeps squared-excess term in a PPO-friendly range
NOISE_SD = 1.5               # observation-noise sd of the mechanistic layer
FEATS_PER_GRID = 11


def load_panel(path=None):
    import pandas as pd
    path = path or os.path.join(DATA_DIR, 'results', 'panel_data.csv')
    panel = pd.read_csv(path)
    # weekly calendar shared by all grids
    weeks = (panel[panel['grid'] == GRIDS[0]]
             .sort_values('week_idx')[['week_idx', 'year', 'week_in_year']]
             .reset_index(drop=True))
    data = {}
    for g in GRIDS:
        sub = panel[panel['grid'] == g].sort_values('week_idx')
        data[g] = {k: sub[k].values for k in
                   ['PFI', 'pop_density', 'T', 'Rain', 'Hum', 'BI']}
    return weeks, data


class MosquitoEnv(gym.Env):
    """Weekly multi-grid control. Action = MultiDiscrete([3]*10): per grid
    0 = no intervention, 1 = source reduction, 2 = space spraying."""

    metadata = {'render_modes': []}

    def __init__(self, alpha=0.1, train_years=(2015, 2016, 2017, 2018, 2019,
                                               2020, 2021),
                 eval_year=None, seed=0, params=None, max_interventions=None,
                 action_set=(0, 1, 2)):
        super().__init__()
        self.alpha = float(alpha)
        self.train_years = tuple(train_years)
        self.eval_year = eval_year
        self.params = dict(INTERVENTION_PARAMS)
        if params:
            self.params.update(params)
        self.max_interventions = max_interventions   # optional hard weekly cap
        self.action_set = tuple(action_set)          # ablation support
        self.rng = np.random.default_rng(seed)
        self.weeks, self.data = load_panel()
        self.year_blocks = self._index_years()

        self.action_space = spaces.MultiDiscrete([len(self.action_set)] * N_GRIDS)
        self.observation_space = spaces.Box(
            low=0.0, high=5.0,
            shape=(FEATS_PER_GRID * N_GRIDS,), dtype=np.float32)

    # ------------------------------------------------------------------ util
    def _index_years(self):
        blocks = {}
        for y in np.unique(self.weeks['year']):
            idx = np.where(self.weeks['year'].values == y)[0]
            blocks[int(y)] = idx
        return blocks

    def _base_bi(self, t_idx):
        """Baseline BI for all grids at week index t_idx: PFI * pop_scale
        plus freshly sampled observation noise (per episode)."""
        out = np.zeros(N_GRIDS)
        for i, g in enumerate(GRIDS):
            mu = self.data[g]['PFI'][t_idx] * self.data[g]['pop_density'][0] * CALIB_SCALE
            out[i] = mu + self.rng.normal(0, NOISE_SD)
        return np.maximum(out, 0.0)

    def _obs(self, t_idx, bi, bi_prev, p_sr, p_sp, last_action):
        wiy = self.weeks['week_in_year'].values[t_idx]
        sin_w = (np.sin(2 * np.pi * wiy / 52.0) + 1) / 2
        cos_w = (np.cos(2 * np.pi * wiy / 52.0) + 1) / 2
        vec = np.zeros(N_GRIDS * FEATS_PER_GRID, dtype=np.float32)
        for i, g in enumerate(GRIDS):
            d = self.data[g]
            row = [d['T'][t_idx] / 35.0, d['Rain'][t_idx] / 120.0,
                   d['Hum'][t_idx] / 100.0, sin_w, cos_w,
                   bi[i] / 40.0, bi_prev[i] / 40.0,
                   p_sr[i], p_sp[i],
                   1.0 if last_action[i] == 1 else 0.0,
                   1.0 if last_action[i] == 2 else 0.0]
            vec[i * FEATS_PER_GRID:(i + 1) * FEATS_PER_GRID] = row
        return vec

    # ------------------------------------------------------------------ gym
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.eval_year is not None:
            year = self.eval_year
        else:
            year = int(self.rng.choice(self.train_years))
        self.block = self.year_blocks[year]
        self.year = year
        self.tpos = 0
        self.p_sr = np.zeros(N_GRIDS)
        self.p_sp = np.zeros(N_GRIDS)
        self.last_action = np.zeros(N_GRIDS, dtype=int)
        t_idx = self.block[0]
        self.bi = self._base_bi(t_idx)
        self.bi_prev = self.bi.copy()
        self.cost_total = 0.0
        return self._obs(t_idx, self.bi, self.bi_prev, self.p_sr, self.p_sp,
                         self.last_action), {'year': year}

    def step(self, action):
        action = np.asarray(action, dtype=int).clip(0, len(self.action_set) - 1)
        action = np.array([self.action_set[a] for a in action], dtype=int)
        p = self.params
        cost = float(np.sum(action == 1) * p['sr_cost']
                     + np.sum(action == 2) * p['sp_cost'])
        if self.max_interventions is not None:
            excess = int(np.sum(action > 0)) - self.max_interventions
            if excess > 0:
                cost += excess * 5.0 * p['sp_cost']      # soft capacity penalty

        # protection stocks update (action at t affects BI from t+1 on)
        self.p_sr = self.p_sr * p['sr_decay'] + p['sr_efficacy'] * (action == 1)
        self.p_sp = self.p_sp * p['sp_decay'] + p['sp_efficacy'] * (action == 2)
        self.p_sr = np.clip(self.p_sr, 0, 0.95)
        self.p_sp = np.clip(self.p_sp, 0, 0.95)

        # advance one week
        self.tpos += 1
        truncated = self.tpos >= len(self.block) - 1
        t_idx = self.block[min(self.tpos, len(self.block) - 1)]
        base = self._base_bi(t_idx)
        self.bi_prev = self.bi
        self.bi = base * (1.0 - self.p_sr) * (1.0 - self.p_sp)
        self.last_action = action
        self.cost_total += cost

        excess = np.maximum(self.bi - BI_THRESHOLD, 0.0)
        health_penalty = float(np.mean((excess / REWARD_SCALE) ** 2))
        reward = -health_penalty - self.alpha * cost / N_GRIDS

        info = {'mean_bi': float(np.mean(self.bi)),
                'cost': cost, 'cost_total': self.cost_total,
                'weeks_above10': float(np.mean(self.bi >= 10.0))}
        return self._obs(t_idx, self.bi, self.bi_prev, self.p_sr, self.p_sp,
                         action), reward, False, truncated, info
