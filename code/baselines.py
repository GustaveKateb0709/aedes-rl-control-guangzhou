#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rule-based control policies (counterfactual arms).

Each policy maps the env state to a per-grid action array.
Actions follow the env convention: 0 = none, 1 = source reduction,
2 = space spraying.
"""
import numpy as np
from mosquito_env import N_GRIDS, BI_THRESHOLD


def no_intervention(env):
    """Arm C: never intervene."""
    return np.zeros(N_GRIDS, dtype=int)


def threshold_rule(env, spray_trigger=15.0):
    """Arm B (static rule): spray a grid when its current BI exceeds the
    high-risk trigger (default 15); otherwise do nothing."""
    return np.where(env.bi >= spray_trigger, 2, 0).astype(int)


def calendar_rule(env, sr_trigger=BI_THRESHOLD, season=(18, 40), period=4):
    """Preventive calendar arm: during the high-transmission season, grids
    above the safety threshold receive source reduction on a fixed
    four-weekly rotation."""
    wiy = int(env.weeks['week_in_year'].values[env.block[min(env.tpos, len(env.block) - 1)]])
    a = np.zeros(N_GRIDS, dtype=int)
    if season[0] <= wiy <= season[1] and wiy % period == 0:
        a[env.bi >= sr_trigger] = 1
    return a


RULES = {'none': no_intervention,
         'threshold': threshold_rule,
         'calendar': calendar_rule}
