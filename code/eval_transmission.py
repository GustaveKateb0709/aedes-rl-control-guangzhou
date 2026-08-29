#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Transmission-potential evaluation (Table 6 / Figure 9 data source).

Converts each control arm's simulated 2022 BI trajectory into a weekly
relative index of chikungunya transmission potential, using the
temperature-trait model of Tegar et al. 2026 (J R Soc Interface,
DOI 10.1098/rsif.2025.0707) for CHIKV in Aedes albopictus:

  EIP50(T)  median extrinsic incubation period: linear interpolation
            between 8.74 days at 18 C and 1.74 days at 30 C (clamped to
            [1.5, 24] days);
  VC(T)     vector competence: Gaussian peak 0.96 at 25.6 C, support
            13.8-31.8 C;
  weekly adult survival p: two scenarios, 0.90 (main) and 0.80
            (sensitivity).

Weekly index for grid g:  BI_g(t) * VC(T_g(t)) * p ** EIP50(T_g(t)).
The yearly cumulative value sums the weekly index over weeks and grids.
Because all arms share the same weather series, ratios between arms do
not depend on the survival parameterization.

Outputs:
  results/transmission_weekly.npy   {(p, arm): (cumulative, weekly array)}
  results/transmission_summary.csv  per-arm totals and shares
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(BASE, '..', 'results'))
GRIDS = ['conghua', 'huadu', 'zengcheng', 'baiyun', 'tianhe',
         'yuexiu', 'haizhu', 'panyu', 'nansha', 'huangpu']
ARMS = ['none', 'threshold', 'calendar',
        'ppo_s0', 'ppo_s1', 'ppo_s2']
ARM_FILES = {'none': 'trajectory_none.csv',
             'threshold': 'trajectory_threshold.csv',
             'calendar': 'trajectory_calendar.csv',
             'ppo_s0': 'trajectory_ppo_a0.1_s0_acts012.csv',
             'ppo_s1': 'trajectory_ppo_a0.1_s1_acts012.csv',
             'ppo_s2': 'trajectory_ppo_a0.1_s2_acts012.csv'}
SCENARIOS = [0.90, 0.80]


def eip50(t):
    """Median extrinsic incubation period (days), linear between anchors."""
    return float(np.clip(np.interp(t, [18.0, 30.0], [8.74, 1.74]), 1.5, 24.0))


def vc(t):
    """Vector competence: Gaussian peak 0.96 at 25.6 C, support 13.8-31.8 C."""
    return 0.96 * np.exp(-((t - 25.6) ** 2) / (2 * 2.48 ** 2)) \
        * float(13.8 <= t <= 31.8)


def main():
    panel = pd.read_csv(os.path.join(RES, 'panel_data.csv'))
    temp = panel[panel['year'] == 2022].set_index(['week_in_year', 'grid'])['T'].to_dict()

    out = {}
    for p in SCENARIOS:
        for arm in ARMS:
            df = pd.read_csv(os.path.join(RES, ARM_FILES[arm]))
            weekly = []
            for _, row in df.iterrows():
                wk = int(row['week_in_year'])
                s = 0.0
                for g in GRIDS:
                    t = temp[(wk, g)]
                    s += row[f'BI_{g}'] * vc(t) * (p ** eip50(t))
                weekly.append(s)
            out[(p, arm)] = (float(np.sum(weekly)), np.asarray(weekly))

    np.save(os.path.join(RES, 'transmission_weekly.npy'), out,
            allow_pickle=True)

    rows = []
    for p in SCENARIOS:
        ref = out[(p, 'none')][0]
        for arm in ARMS:
            tot = out[(p, arm)][0]
            rows.append({'survival_p': p, 'arm': arm,
                         'cumulative_tp': round(tot, 1),
                         'pct_of_none': round(100 * tot / ref, 1)})
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(RES, 'transmission_summary.csv'),
                   index=False, encoding='utf-8-sig')
    print(summary.to_string(index=False))
    print('\nsaved transmission_weekly.npy / transmission_summary.csv ->', RES)


if __name__ == '__main__':
    main()
