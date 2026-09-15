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

Weekly capacity budget (C_max)
------------------------------
`C_max` is a hard weekly budget for the WHOLE city, expressed in the same
cost units as the intervention parameters (source reduction = 1 unit per
grid-week, space spraying = 3 units per grid-week):

    sum_g cost_g(t) <= C_max            (cost_g = 0 / 1 / 3)

alpha and C_max are conceptually distinct: alpha sets the *exchange rate*
between health and money, C_max sets what is *physically achievable* in a
given week (crews, vehicles, consumables).

When `C_max` is not None the constraint is enforced AT ACTION-SAMPLING
TIME, not by a penalty term: the action space is restricted to the
enumerable set of feasible action vectors,

    action_space = Discrete(K),  K = #{ a in A^10 : cost(a) <= C_max }

and the sampled integer indexes that table (`self.feasible_actions`), so an
infeasible plan is not representable and can never be emitted.  When
`C_max is None` nothing changes at all: the environment is bit-identical to
the previous version (MultiDiscrete action space, no budget term), which is
used as the new-vs-old consistency check.

Observation augmentation (adjacency ablation)
---------------------------------------------
`augment_adjacency=True` appends, per grid, the mean BI and mean previous BI
of the administrative neighbours defined in ADJACENCY (2 extra features per
grid, 13 * 10 = 130 dims).  This exists only to test whether graph/
neighbourhood information adds anything over the per-grid features.

Action-parameterisation controls (k_confound.py)
------------------------------------------------
Two extra flags exist ONLY to separate the effect of the *constraint* from the
effect of the *size of the action table*, which are collinear in the plain
sweep (smaller C_max <=> smaller K):

  * `action_table_size=K` restricts the action table to K rows WITHOUT changing
    C_max.  With `C_max=None` this yields `Discrete(K)`, i.e. a "large/small
    action head at zero budget pressure" control.  `subset_core=c` additionally
    makes the table NESTED: every action whose weekly cost is <= c is always
    kept and only the padding is drawn at random, so the useful action set is
    held fixed while the number of alternatives grows.
  * `budget_projection=True` keeps the FULL joint space as the table but
    greedily projects every sampled plan onto `C_max`, i.e. a "large action
    head under a tight budget" control.

Neither flag alters the dynamics, the reward or any calibrated parameter.

Design note: alpha is NOT presented as universal; the effect parameters
carry explicit ranges and are perturbed in the robustness analysis; all
conclusions are framed as simulation-based.
"""
import itertools
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

# Administrative-contiguity graph of the 10 Guangzhou districts, used ONLY by
# the adjacency-augmentation ablation (spatial_ablation.py).  Symmetric
# (undirected) by construction; keys/values are district names.
ADJACENCY = {
    'conghua':   ['huadu', 'zengcheng', 'baiyun'],
    'huadu':     ['conghua', 'baiyun', 'zengcheng'],
    'zengcheng': ['conghua', 'huadu', 'baiyun', 'huangpu'],
    'baiyun':    ['conghua', 'huadu', 'zengcheng', 'huangpu', 'tianhe', 'yuexiu'],
    'tianhe':    ['baiyun', 'yuexiu', 'haizhu', 'huangpu'],
    'yuexiu':    ['baiyun', 'tianhe', 'haizhu'],
    'haizhu':    ['yuexiu', 'tianhe', 'huangpu', 'panyu'],
    'panyu':     ['haizhu', 'huangpu', 'nansha'],
    'nansha':    ['panyu', 'huangpu'],
    'huangpu':   ['baiyun', 'zengcheng', 'tianhe', 'haizhu', 'panyu', 'nansha'],
}
ADJ_IDX = [[GRIDS.index(n) for n in ADJACENCY[g]] for g in GRIDS]
NEIGHBOUR_FEATS = 2          # mean neighbour BI, mean neighbour previous BI

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


_PANEL_CACHE = {}
_FEASIBLE_CACHE = {}
_CODE_TABLE_CACHE = {}
_SUBSET_CACHE = {}


def _subsample_table(table, size, seed=0, core_mask=None):
    """Deterministic `size`-row subsample of an action table.

    Used ONLY by the action-parameterisation control experiment
    (k_confound.py): it builds a Discrete(K) table of a prescribed SIZE so
    that K can be varied while the (absent) budget is held fixed.

    If `core_mask` is given, **every** masked row is kept and the table is
    padded with random unmasked rows up to `size`.  This is essential: a plain
    uniform subsample of the full 3**10 space almost never contains the
    all-zero action, so the agent would be *forced to spend* every week -- an
    artifact, not an action-head effect.  With a nested table the useful
    action set is held FIXED and only the number of extra alternatives grows.

    The padding choice is fixed by `seed`, hence identical across training
    seeds and across processes.
    """
    size = int(size)
    if size >= len(table):
        return table
    if core_mask is None:
        core_idx = np.zeros(0, dtype=int)
        pool = np.arange(len(table))
    else:
        core_mask = np.asarray(core_mask, dtype=bool)
        core_idx = np.flatnonzero(core_mask)
        pool = np.flatnonzero(~core_mask)
    if len(core_idx) > size:                      # core alone must fit
        core_idx = core_idx[:size]
        pool = np.arange(len(table))
    n_pad = size - len(core_idx)
    key = (size, int(seed), len(table), table.shape[1], len(core_idx),
           hash(table.tobytes()))
    if key not in _SUBSET_CACHE:
        rng = np.random.default_rng(1_000_003 + int(seed))
        pad = (np.sort(rng.choice(pool, size=n_pad, replace=False))
               if n_pad > 0 else np.zeros(0, dtype=int))
        idx = np.sort(np.concatenate([core_idx, pad]))
        _SUBSET_CACHE[key] = np.ascontiguousarray(table[idx])
    return _SUBSET_CACHE[key]


def _all_code_vectors(action_set):
    """All |action_set|^N_GRIDS joint action vectors as env action CODES
    (cached per action set)."""
    key = tuple(action_set)
    if key not in _CODE_TABLE_CACHE:
        n_act = len(key)
        combos = np.asarray(list(itertools.product(range(n_act),
                                                   repeat=N_GRIDS)), dtype=int)
        _CODE_TABLE_CACHE[key] = np.asarray(key, dtype=int)[combos]
    return _CODE_TABLE_CACHE[key]


def load_panel(path=None):
    """Load the weekly panel (cached: environments are constructed very many
    times during sweeps and the panel is read-only)."""
    import pandas as pd
    path = path or os.path.join(DATA_DIR, 'results', 'panel_data.csv')
    if path in _PANEL_CACHE:
        return _PANEL_CACHE[path]
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
    _PANEL_CACHE[path] = (weeks, data)
    return weeks, data


class MosquitoEnv(gym.Env):
    """Weekly multi-grid control. Action = MultiDiscrete([3]*10): per grid
    0 = no intervention, 1 = source reduction, 2 = space spraying.

    If `C_max` is given, the action space becomes Discrete(K) over the
    enumerable set of weekly-feasible action vectors (see module docstring).
    """

    metadata = {'render_modes': []}

    def __init__(self, alpha=0.1, train_years=(2015, 2016, 2017, 2018, 2019,
                                               2020, 2021),
                 eval_year=None, seed=0, params=None, max_interventions=None,
                 action_set=(0, 1, 2), C_max=None, augment_adjacency=False,
                 flatten_actions=False, action_table_size=None,
                 subset_seed=0, subset_core=None, budget_projection=False):
        super().__init__()
        self.alpha = float(alpha)
        self.train_years = tuple(train_years)
        self.eval_year = eval_year
        self.params = dict(INTERVENTION_PARAMS)
        if params:
            self.params.update(params)
        self.max_interventions = max_interventions   # optional legacy hard cap
        self.action_set = tuple(action_set)          # ablation support
        self.C_max = None if C_max is None else float(C_max)
        self.augment_adjacency = bool(augment_adjacency)
        self.flatten_actions = bool(flatten_actions)
        # --- action-parameterisation control (k_confound.py) -----------------
        # `action_table_size` fixes the SIZE of the Discrete action table to K
        # rows (a fixed-seed random subsample), which is what lets us vary K
        # while holding C_max fixed.  `budget_projection` keeps the FULL joint
        # space as the table but greedily projects every sampled plan onto the
        # budget, i.e. a large-K representation of a tight budget.
        self.action_table_size = (None if action_table_size is None
                                  else int(action_table_size))
        self.subset_seed = int(subset_seed)
        self.subset_core = None if subset_core is None else float(subset_core)
        self.projected_actions = bool(budget_projection and self.C_max is not None)
        self.feats_per_grid = FEATS_PER_GRID + (NEIGHBOUR_FEATS
                                               if self.augment_adjacency else 0)
        self.rng = np.random.default_rng(seed)
        self.weeks, self.data = load_panel()
        self.year_blocks = self._index_years()

        # action coding table: index inside action_set -> env action code
        self.action_codes = np.array([self.action_set[a]
                                      for a in range(len(self.action_set))],
                                     dtype=int)

        if (self.C_max is None and not self.flatten_actions
                and not self.projected_actions
                and self.action_table_size is None):
            # legacy behaviour: unrestricted joint action space
            self.feasible_actions = None
            self.feasible_costs = None
            self.action_space = spaces.MultiDiscrete(
                [len(self.action_set)] * N_GRIDS)
        else:
            table = self._build_feasible_actions()
            if self.projected_actions:
                # K = the FULL joint space (up to 3**10); feasibility is
                # restored downstream by greedy projection
                table = _all_code_vectors(self.action_set)
            if self.action_table_size is not None:
                core = None
                if self.subset_core is not None:
                    _c = ((table == 1).sum(axis=1) * self.params['sr_cost']
                          + (table == 2).sum(axis=1) * self.params['sp_cost'])
                    core = _c <= self.subset_core + 1e-9
                table = _subsample_table(table, self.action_table_size,
                                         self.subset_seed, core)
            self.feasible_actions = table
            self.feasible_costs = ((table == 1).sum(axis=1) * self.params['sr_cost']
                                   + (table == 2).sum(axis=1) * self.params['sp_cost'])
            self.action_space = spaces.Discrete(len(table))

        self.observation_space = spaces.Box(
            low=0.0, high=5.0,
            shape=(self.feats_per_grid * N_GRIDS,), dtype=np.float32)

    # ------------------------------------------------------- capacity budget
    def _build_feasible_actions(self):
        """All action-code vectors whose weekly cost is <= C_max.

        A = len(action_set); |A|^N_GRIDS is at most 3^10 = 59049, so the
        enumeration is exact and cheap.  Both the enumeration of all code
        vectors and the resulting feasibility table are cached at module level
        because sweeps construct thousands of identical environments.
        """
        key = (self.C_max, self.action_set)
        if key in _FEASIBLE_CACHE:
            return _FEASIBLE_CACHE[key]
        codes_all = _all_code_vectors(self.action_set)          # (K, N_GRIDS)
        p = self.params
        costs = ((codes_all == 1).sum(axis=1) * p['sr_cost']
                 + (codes_all == 2).sum(axis=1) * p['sp_cost'])
        if self.C_max is None:
            keep = codes_all
        else:
            keep = codes_all[costs <= self.C_max + 1e-9]
        if len(keep) == 0:                       # C_max too small for anything
            keep = np.zeros((1, N_GRIDS), dtype=int)
        _FEASIBLE_CACHE[key] = keep
        return keep

    def action_cost(self, codes):
        """Weekly cost in cost units (SR = sr_cost, SP = sp_cost)."""
        codes = np.asarray(codes, dtype=int)
        p = self.params
        return float(np.sum(codes == 1) * p['sr_cost']
                     + np.sum(codes == 2) * p['sp_cost'])

    def project_to_budget(self, codes, priority=None):
        """Greedy feasible projection: keep the highest-priority scheduled
        actions that fit the weekly budget; drop the rest.

        Used only by NON-learning arms (rules) so that every compared arm
        faces the same physical budget.  Learning arms never need it: their
        action space contains feasible vectors only.
        """
        codes = np.asarray(codes, dtype=int).copy()
        if self.C_max is None:
            return codes
        if self.action_cost(codes) <= self.C_max + 1e-9:
            return codes
        if priority is None:
            priority = np.asarray(self.bi, dtype=float)
        order = np.argsort(-np.asarray(priority, dtype=float), kind='stable')
        out = np.zeros(N_GRIDS, dtype=int)
        used = 0.0
        p = self.params
        for i in order:
            c = 0.0 if codes[i] == 0 else (
                p['sr_cost'] if codes[i] == 1 else p['sp_cost'])
            if c and used + c <= self.C_max + 1e-9:
                out[i] = codes[i]
                used += c
        return out

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

    def _base_bi_mean(self, t_idx):
        """Noise-free mean baseline BI (does NOT touch the RNG)."""
        out = np.array([self.data[g]['PFI'][t_idx]
                        * self.data[g]['pop_density'][0] * CALIB_SCALE
                        for g in GRIDS], dtype=float)
        return np.maximum(out, 0.0)

    def _obs(self, t_idx, bi, bi_prev, p_sr, p_sp, last_action):
        wiy = self.weeks['week_in_year'].values[t_idx]
        sin_w = (np.sin(2 * np.pi * wiy / 52.0) + 1) / 2
        cos_w = (np.cos(2 * np.pi * wiy / 52.0) + 1) / 2
        vec = np.zeros(N_GRIDS * self.feats_per_grid, dtype=np.float32)
        for i, g in enumerate(GRIDS):
            d = self.data[g]
            row = [d['T'][t_idx] / 35.0, d['Rain'][t_idx] / 120.0,
                   d['Hum'][t_idx] / 100.0, sin_w, cos_w,
                   bi[i] / 40.0, bi_prev[i] / 40.0,
                   p_sr[i], p_sp[i],
                   1.0 if last_action[i] == 1 else 0.0,
                   1.0 if last_action[i] == 2 else 0.0]
            if self.augment_adjacency:
                nb = ADJ_IDX[i]
                row.append(float(np.mean(bi[nb])) / 40.0)
                row.append(float(np.mean(bi_prev[nb])) / 40.0)
            vec[i * self.feats_per_grid:(i + 1) * self.feats_per_grid] = row
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

    def _decode_action(self, action):
        """Map an incoming action to per-grid action CODES.

        * `feasible_actions is None`  -> `action` is the legacy MultiDiscrete
          vector.
        * a scalar / 1-element array  -> it indexes `self.feasible_actions`.
        * a length-N_GRIDS vector     -> accepted for non-learning callers.

        Any plan whose cost exceeds `C_max` is greedily projected onto the
        budget and the pre-projection overflow is reported: this never happens
        for learning arms with a budgeted table (they can only emit feasible
        plans), but it is exactly what the `budget_projection` control arm
        relies on.
        """
        a = np.asarray(action, dtype=int).ravel()
        if self.feasible_actions is not None and a.size == 1:
            codes = self.feasible_actions[int(np.clip(
                a[0], 0, len(self.feasible_actions) - 1))]
        else:
            a = a.clip(0, len(self.action_set) - 1)
            codes = self.action_codes[a]
        if self.feasible_actions is None or self.C_max is None:
            return codes, 0.0
        over = max(0.0, self.action_cost(codes) - self.C_max)
        if over > 0:
            codes = self.project_to_budget(codes)
        return codes, float(over)

    def peek_rewards(self, candidate_codes, next_base=None):
        """Vectorised immediate reward for a BATCH of candidate action vectors.

        Returns (rewards[K], next_bi[K, N]) without mutating the environment.
        Used by the one-step-lookahead (degenerate MPC) baseline.  Assumes the
        same dynamics as `step` (action at t affects BI from t+1 onwards).
        """
        C = np.asarray(candidate_codes, dtype=int)
        p = self.params
        cost = ((C == 1).sum(axis=1) * p['sr_cost']
                + (C == 2).sum(axis=1) * p['sp_cost'])
        n_sr = p['sr_decay'] * self.p_sr[None, :] + p['sr_efficacy'] * (C == 1)
        n_sp = p['sp_decay'] * self.p_sp[None, :] + p['sp_efficacy'] * (C == 2)
        np.clip(n_sr, 0, 0.95, out=n_sr)
        np.clip(n_sp, 0, 0.95, out=n_sp)
        if next_base is None:
            t_idx = self.block[min(self.tpos + 1, len(self.block) - 1)]
            next_base = self._base_bi_mean(t_idx)   # noise-free forecast
        nb = next_base[None, :] * (1.0 - n_sr) * (1.0 - n_sp)
        excess = np.maximum(nb - BI_THRESHOLD, 0.0)
        health = np.mean((excess / REWARD_SCALE) ** 2, axis=1)
        return -health - self.alpha * cost / N_GRIDS, nb

    def step(self, action):
        action, capacity_overflow = self._decode_action(action)
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
                'n_sr': int(np.sum(action == 1)), 'n_sp': int(np.sum(action == 2)),
                'capacity_overflow': capacity_overflow,
                'capacity_slack': (None if self.C_max is None
                                   else float(self.C_max - cost)),
                'weeks_above10': float(np.mean(self.bi >= 10.0))}
        return self._obs(t_idx, self.bi, self.bi_prev, self.p_sr, self.p_sp,
                         action), reward, False, truncated, info
