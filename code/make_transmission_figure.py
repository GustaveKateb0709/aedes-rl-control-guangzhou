#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 9: transmission potential by control arm.

Reads results/transmission_weekly.npy (temperature-trait model of Tegar et
al. 2026, J R Soc Interface, DOI 10.1098/rsif.2025.0707) and writes
Figure9.png/.pdf into the figures directory next to code/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(BASE, '..', 'results'))
OUT = os.environ.get('FIG9_OUT',
                     os.path.abspath(os.path.join(BASE, '..', 'figures')))
os.makedirs(OUT, exist_ok=True)

INK = '#2c3e50'
C_NONE = '#95a5a6'
C_THRESH = '#5d6d7e'
C_CAL = '#e67e22'
C_PPO = '#c0392b'
P_MAIN = 0.90


def load(path):
    d = np.load(path, allow_pickle=True).item()
    return {(float(p), arm): (float(tot), np.asarray(wk, dtype=float))
            for (p, arm), (tot, wk) in d.items()}


def main():
    d = load(os.path.join(RES, 'transmission_weekly.npy'))
    p = P_MAIN

    arms = ['none', 'threshold', 'calendar', 'ppo_s0']
    labels = ['No intervention', 'Threshold rule\n(BI 15, never triggers)',
              'Calendar rule', 'PPO policy']
    colors = [C_NONE, C_THRESH, C_CAL, C_PPO]
    tots = [d[(p, a)][0] for a in arms]
    ref = tots[0]
    ppo_seeds = [d[(p, f'ppo_s{s}')][0] for s in (0, 1, 2)]
    ppo_lo, ppo_hi = min(ppo_seeds), max(ppo_seeds)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.6),
                                   gridspec_kw={'width_ratios': [1, 1.3]})

    # ---- (a) cumulative totals ----
    x = np.arange(len(arms))
    bars = ax1.bar(x, tots, color=colors, width=0.62)
    bars[1].set_hatch('//'); bars[1].set_edgecolor('white')
    for i, t in enumerate(tots):
        lab = f'{t:.0f}\n({100 * t / ref:.1f}%)'
        ax1.text(i, t + ref * 0.045, lab,
                 ha='center', va='bottom', fontsize=8.4, color=INK)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.4)
    ax1.set_ylabel('Cumulative transmission potential, 2022\n(relative index)')
    ax1.set_ylim(0, ref * 1.30)
    ax1.set_title('(a) Cumulative transmission potential by arm', fontsize=9.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.25, linewidth=0.6)

    # ---- (b) weekly curves ----
    series = [('none', 'No intervention / threshold', C_NONE, 1.6),
              ('calendar', 'Calendar rule', C_CAL, 1.4),
              ('ppo_s0', 'PPO policy (seed 0)', C_PPO, 1.9)]
    for arm, lab, c, lw in series:
        wk = d[(p, arm)][1]
        ax2.plot(np.arange(1, len(wk) + 1), wk, label=lab, color=c, lw=lw)
    ax2.set_xlabel('Week of the 2022 evaluation year')
    ax2.set_ylabel('Weekly transmission potential\n(relative index)')
    ax2.set_ylim(0, 46)
    ax2.set_title('(b) Weekly transmission potential, main survival scenario',
                  fontsize=9.5)
    ax2.legend(loc='upper right', fontsize=8.2, frameon=True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(alpha=0.25, linewidth=0.6)

    fig.suptitle('Fig. 9  Chikungunya transmission potential by control arm',
                 x=0.01, y=0.985, ha='left', va='top', fontsize=13,
                 fontweight='bold', color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    for ext in ('.png', '.pdf'):
        fig.savefig(os.path.join(OUT, 'Figure9' + ext), dpi=300,
                    facecolor='white')
    plt.close(fig)
    print('saved Figure9.png / Figure9.pdf ->', OUT)


if __name__ == '__main__':
    main()
