#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Publication figures (English labels, 300 dpi PNG + PDF).

Reads results/*.csv + results/monitor/ and writes Figure1..Figure8 into ../figures/.
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(BASE, '..', 'results'))
FIGS = os.path.abspath(os.path.join(BASE, '..', 'figures'))
os.makedirs(FIGS, exist_ok=True)

INK = '#2c3e50'
C_NONE = '#95a5a6'      # no intervention          - gray
C_THRESH = '#5d6d7e'    # threshold rule (inert)   - dark slate, dashed/open
C_CAL = '#e67e22'       # preventive calendar      - muted orange
C_PPO = '#c0392b'       # PPO policy               - muted red
C_SR = '#2e86c1'        # source reduction         - blue
C_SP = '#c0392b'        # space spraying           - red
CALIB_SCALE = 0.3476    # surveillance calibration (see mosquito_env.py)

ARM_STYLE = {'none': ('No intervention', C_NONE, '-'),
             'threshold': ('Threshold rule (never triggers)', C_THRESH, '--'),
             'calendar': ('Preventive calendar rule', C_CAL, '-')}
PPO_SEED_COLORS = ['#aed6f1', '#5dade2', '#2e86c1']


def set_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.25, linewidth=0.6)


def finish(fig, name, title, rect):
    fig.suptitle(title, x=0.01, y=0.985, ha='left', va='top',
                 fontsize=13, fontweight='bold', color=INK)
    fig.tight_layout(rect=rect)
    for ext in ('.png', '.pdf'):
        fig.savefig(os.path.join(FIGS, name + ext), dpi=300, facecolor='white')
    plt.close(fig)
    print('  saved', name, flush=True)


def fig1_framework():
    fig, ax = plt.subplots(figsize=(11.4, 6.0))
    ax.set_xlim(0, 126); ax.set_ylim(0, 51); ax.axis('off')
    W, H, GAP, M = 27.0, 17.0, 4.0, 3.0
    xs = [M + W / 2 + i * (W + GAP) for i in range(4)]
    CY = 40.0

    stages = [
        (xs[0], 'ERA5 reanalysis\nweather', '2015 to 2022,\nten Guangzhou grids', C_NONE, '#f4f6f7'),
        (xs[1], 'PFI vector dynamics', 'weekly baseline BI,\ncalibrated to Guangzhou\nsurveillance (mean 3.7)', C_SR, '#eaf2f8'),
        (xs[2], 'MosquitoEnv (MDP)', 'weekly state; actions:\nnone / source\nreduction / spraying', C_CAL, '#fdf2e9'),
        (xs[3], 'PPO policy', 'selects one action\nper grid, every week', C_PPO, '#fdedec'),
    ]
    for x, head, body, ec, fc in stages:
        ax.add_patch(FancyBboxPatch((x - W / 2, CY - H / 2), W, H,
                                    boxstyle='round,pad=0.02,rounding_size=1.2',
                                    lw=1.5, edgecolor=ec, facecolor=fc))
        ax.text(x, CY + H / 2 - 3.4, head, ha='center', va='center',
                fontsize=12.5, fontweight='bold', color=INK)
        ax.text(x, CY - 2.6, body, ha='center', va='center',
                fontsize=9.2, color=INK, linespacing=1.4)
    for i in range(3):
        ax.add_patch(FancyArrowPatch((xs[i] + W / 2 + 0.4, CY),
                                     (xs[i + 1] - W / 2 - 0.4, CY),
                                     arrowstyle='-|>', mutation_scale=18,
                                     lw=2.0, color=C_SR))

    RB_W, RB_H, RB_CY = 48.0, 10.0, 12.0
    RB_CX = (xs[0] + xs[3]) / 2
    ax.add_patch(FancyBboxPatch((RB_CX - RB_W / 2, RB_CY - RB_H / 2), RB_W, RB_H,
                                boxstyle='round,pad=0.02,rounding_size=1.2',
                                lw=1.4, edgecolor=C_NONE, facecolor='#f4f6f7'))
    ax.text(RB_CX, RB_CY + 1.6, 'Reward', ha='center', va='center',
            fontsize=11, fontweight='bold', color=INK)
    ax.text(RB_CX, RB_CY - 2.2, 'threshold-excess BI (squared)\nminus scaled intervention cost',
            ha='center', va='center', fontsize=9.2, color=INK, linespacing=1.4)
    st_bot = CY - H / 2
    mid_x = (xs[1] + xs[2]) / 2
    ax.add_patch(FancyArrowPatch((mid_x - 1.4, st_bot - 0.4),
                                 (mid_x - 1.4, RB_CY + RB_H / 2 + 0.4),
                                 arrowstyle='-|>', mutation_scale=15,
                                 lw=1.6, color=C_NONE))
    ax.add_patch(FancyArrowPatch((mid_x + 1.4, RB_CY + RB_H / 2 + 0.4),
                                 (mid_x + 1.4, st_bot - 0.4),
                                 arrowstyle='-|>', mutation_scale=15,
                                 lw=1.6, color=C_PPO))
    ax.text(mid_x - 2.2, (st_bot + RB_CY + RB_H / 2) / 2, 'state',
            ha='right', va='center', fontsize=9.0, color=C_NONE)
    ax.text(mid_x + 2.2, (st_bot + RB_CY + RB_H / 2) / 2, 'reward',
            ha='left', va='center', fontsize=9.0, color=C_PPO)
    finish(fig, 'Figure1', 'Fig. 1  Decision framework', (0, 0, 1, 0.952))


def fig2_learning_curves():
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    for i, (f, lab) in enumerate([
            ('ppo_a0.1_s0_acts012.monitor.csv', 'seed 0'),
            ('ppo_a0.1_s1_acts012.monitor.csv', 'seed 1'),
            ('ppo_a0.1_s2_acts012.monitor.csv', 'seed 2')]):
        p = os.path.join(RES, 'monitor', f)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p, skiprows=1)
        r = d['r'].rolling(200, min_periods=20).mean()
        ax.plot(np.arange(len(r)), r, label=f'PPO (seed {lab.split()[-1]})',
                color=PPO_SEED_COLORS[i], lw=1.5)
    ax.set_xlabel('Training episode (one episode = one simulated year)')
    ax.set_ylabel('Episode reward (200-episode moving average)')
    ax.legend(loc='lower right', fontsize=9)
    set_axis(ax)
    finish(fig, 'Figure2', 'Fig. 2  PPO training convergence (three seeds)',
           (0, 0, 1, 0.945))


def fig3_trajectories():
    ppo = sorted(glob.glob(os.path.join(RES, 'trajectory_ppo_a0.1_s0*.csv')))
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    handles = []
    for arm in ['none', 'threshold', 'calendar']:
        p = os.path.join(RES, f'trajectory_{arm}.csv')
        d = pd.read_csv(p)
        bi_cols = [c for c in d.columns if c.startswith('BI_')]
        lab, c, ls = ARM_STYLE[arm]
        h, = ax.plot(d['week_in_year'], d[bi_cols].mean(axis=1), color=c,
                     ls=ls, lw=2.4 if arm == 'none' else 1.5,
                     label=lab + (' (identical to no intervention)' if arm == 'threshold' else ''))
        handles.append(h)
    if ppo:
        d = pd.read_csv(ppo[0])
        bi_cols = [c for c in d.columns if c.startswith('BI_')]
        h, = ax.plot(d['week_in_year'], d[bi_cols].mean(axis=1), color=C_PPO,
                     lw=2.2, label='PPO dynamic policy')
        handles.append(h)
        ax2 = ax.twinx()
        hb = ax2.bar(d['week_in_year'], d['n_sr'], color=C_SR, alpha=0.30,
                     width=0.75, label='grids under source reduction (right axis)')
        ax2.set_ylabel('Grids under source reduction per week', color=C_SR)
        ax2.tick_params(axis='y', colors=C_SR)
        ax2.set_ylim(0, 40)
        ax2.spines['top'].set_visible(False)
        handles.append(hb)
    ax.axhline(5, color=INK, ls=':', lw=1.0)
    ax.text(1.0, 5.25, 'BI = 5, transmission-risk threshold', fontsize=8.2, color=INK)
    ax.set_xlabel('Week of year, 2022 (held out from training)')
    ax.set_ylabel('Mean BI across 10 grids')
    ax.set_ylim(0, 8.6)
    set_axis(ax)
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.02))
    finish(fig, 'Figure3', 'Fig. 3  Counterfactual 2022 trajectories under the four control arms',
           (0, 0.16, 1, 0.945))


def fig4_cost_bi():
    s = pd.read_csv(os.path.join(RES, 'counterfactual_summary.csv'))
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    r = s[s['arm'] == 'none'].iloc[0]
    ax.errorbar(r['cost'], r['mean_bi'], xerr=r['cost_sd'], yerr=r['mean_bi_sd'],
                fmt='o', color=C_NONE, markersize=10, capsize=4, lw=1.4, zorder=3)
    ax.plot(0, 3.547, 'o', mfc='white', mec=C_THRESH, mew=1.6, ms=11, zorder=4)
    ax.annotate('No intervention = threshold rule\n(BI 15 trigger never fires)',
                (0, 3.547), xytext=(14, 0), textcoords='offset points',
                fontsize=8.6, color=INK, va='center')
    r = s[s['arm'] == 'calendar'].iloc[0]
    ax.errorbar(r['cost'], r['mean_bi'], xerr=r['cost_sd'], yerr=r['mean_bi_sd'],
                fmt='o', color=C_CAL, markersize=10, capsize=4, lw=1.4, zorder=3)
    ax.annotate('Preventive calendar rule', (24, 3.0), xytext=(14, 0),
                textcoords='offset points', fontsize=8.6, color=INK, va='center')
    pp = s[s['arm'].str.startswith('ppo')]
    m_cost, m_bi = pp['cost'].mean(), pp['mean_bi'].mean()
    ax.errorbar(m_cost, m_bi, xerr=pp['cost'].std(), yerr=pp['mean_bi'].std(),
                fmt='D', color=C_PPO, markerfacecolor=C_PPO, markersize=10,
                capsize=4, lw=1.4, zorder=4)
    ax.scatter(pp['cost'], pp['mean_bi'], s=46, facecolor='white',
               edgecolor=C_PPO, linewidth=1.4, zorder=3)
    ax.annotate('PPO policy (diamond = seed mean;\ncircles = individual seeds)',
                (m_cost, m_bi), xytext=(40, 0), textcoords='offset points',
                fontsize=8.6, color=INK, va='center')
    ax.set_xlabel('Total control cost in 2022 (relative units, mean ± sd)')
    ax.set_ylabel('Mean BI in 2022 (mean ± sd)')
    ax.set_xlim(-12, 128)
    ax.set_ylim(1.55, 3.78)
    set_axis(ax)
    finish(fig, 'Figure4', 'Fig. 4  Cost-health trade-off by control arm (paired noise replicates)',
           (0, 0, 1, 0.945))


def fig5_alpha():
    d = pd.read_csv(os.path.join(RES, 'alpha_sweep.csv')).sort_values('alpha')
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    l1, = ax.plot(d['alpha'], d['mean_bi'], 'o-', color=C_PPO, lw=1.8,
                  label='Mean BI (left axis)')
    ax.set_xlabel('Cost weight α')
    ax.set_ylabel('Mean BI in 2022 (held-out year)', color=C_PPO)
    ax.tick_params(axis='y', labelcolor=C_PPO)
    ax2 = ax.twinx()
    l2, = ax2.plot(d['alpha'], d['total_cost'], 's--', color=C_SR, lw=1.8,
                   label='Total cost (right axis)')
    ax2.set_ylabel('Total cost (relative units)', color=C_SR)
    ax2.tick_params(axis='y', labelcolor=C_SR)
    ax2.spines['top'].set_visible(False)
    set_axis(ax)
    fig.legend(handles=[l1, l2], loc='lower center', ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.005))
    finish(fig, 'Figure5', 'Fig. 5  Policy response to the cost weight α (retrained per value)',
           (0, 0.09, 1, 0.945))


def fig6_robustness():
    d = pd.read_csv(os.path.join(RES, 'robustness_perturbation.csv'))
    nom = d[d['scenario'] == 'nominal']['mean_bi'].iloc[0]
    d = d[d['scenario'] != 'nominal'].copy()
    names = {'sr_efficacy-30%': 'SR effic. −30%',
             'sr_efficacy+30%': 'SR effic. +30%',
             'sp_efficacy-30%': 'Spray effic. −30%',
             'sp_efficacy+30%': 'Spray effic. +30%',
             'sr_decay_fast': 'SR decay fast',
             'sp_decay_slow': 'Spray decay slow',
             'both_efficacy-30%': 'Both −30%'}
    d['label'] = d['scenario'].map(names)
    d = d.sort_values('mean_bi')
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    x = np.arange(len(d))
    colors = ['#e07b54' if v > nom else '#7fb3d5' for v in d['mean_bi']]
    ax.bar(x, d['mean_bi'], color=colors, width=0.62, zorder=2)
    for xi, v in zip(x, d['mean_bi']):
        ax.text(xi, v / 2, f'{v:.2f}', ha='center', va='center', fontsize=9.4,
                color='white', fontweight='bold', zorder=4)
    ax.axhline(nom, color=INK, ls='--', lw=1.2, zorder=3)
    ax.text(0.12, nom + 0.07, f'nominal = {nom:.2f} (ungated policy)',
            fontsize=8.4, color=INK, ha='left', va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', pad=1.2))
    ax.set_xticks(x)
    ax.set_xticklabels(d['label'], fontsize=8.4, rotation=25,
                       ha='right', rotation_mode='anchor')
    ax.set_ylim(0, d['mean_bi'].max() * 1.32)
    ax.set_ylabel('Mean BI in 2022, fixed trained policy (no retraining)')
    set_axis(ax)
    finish(fig, 'Figure6', 'Fig. 6  Robustness to perturbed intervention dynamics',
           (0, 0, 1, 0.945))


def fig7_ablation():
    d = pd.read_csv(os.path.join(RES, 'action_ablation.csv'))
    labels = {'(0, 1, 2)': ('Both tools', '#2e86c1'),
              '(0, 1)': ('Source reduction only', '#7fb3d5'),
              '(0, 2)': ('Spraying only', '#b3c6e7')}
    d['label'] = d['action_set'].map(lambda a: labels[a][0])
    d['color'] = d['action_set'].map(lambda a: labels[a][1])
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    bars = ax.bar(d['label'], d['mean_bi'], color=d['color'], width=0.55)
    for b, (v, c) in zip(bars, zip(d['mean_bi'], d['total_cost'])):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.06,
                f'BI {v:.2f}\ncost {c:.0f}', ha='center', fontsize=9, color=INK)
    ax.set_ylim(0, d['mean_bi'].max() * 1.38)
    ax.set_ylabel('Mean BI in 2022 (held-out year)')
    set_axis(ax)
    finish(fig, 'Figure7', 'Fig. 7  Action-set ablation', (0, 0, 1, 0.945))


def fig8_policy():
    tp = os.path.join(RES, 'trajectory_ppo_a0.1_s0_acts012.csv')
    pp = os.path.join(RES, 'panel_data.csv')
    d = pd.read_csv(tp)
    panel = pd.read_csv(pp)
    p22 = panel[panel['year'] == 2022].copy()
    p22['mu'] = p22['PFI'] * p22['pop_density'] * CALIB_SCALE   # calibrated scale
    baseline = p22.groupby('week_in_year')['mu'].mean()
    j = d.set_index('week_in_year').join(baseline.rename('baseline'))
    bins = [0, 2, 4, 6, 8, 12]
    labels = ['0–2', '2–4', '4–6', '6–8', '8–12']
    j['bin'] = pd.cut(j['baseline'], bins=bins, labels=labels)
    g = j.groupby('bin', observed=True)[['n_sr', 'n_sp']].mean()
    counts = j.groupby('bin', observed=True).size()
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    x = np.arange(len(g))
    ax.bar(x - 0.19, g['n_sr'], width=0.38, color=C_SR, label='source reduction')
    ax.bar(x + 0.19, g['n_sp'], width=0.38, color=C_SP,
           label='space spraying (never selected)\nat the main cost weight α = 0.1')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{lab}\n(n={int(counts.get(lab, 0))})'
                        for lab in g.index], fontsize=8.4)
    ax.set_xlabel('Baseline mean BI of the week (calibrated, no-intervention level)')
    ax.set_ylabel('Mean number of grids treated')
    ax.set_ylim(0, 4.0)
    ax.legend(loc='upper right', fontsize=9)
    set_axis(ax)
    finish(fig, 'Figure8', 'Fig. 8  Learned policy behaviour: intensity vs. baseline risk',
           (0, 0, 1, 0.945))


def main():
    print('generating figures ...', flush=True)
    fig1_framework()
    fig2_learning_curves()
    fig3_trajectories()
    fig4_cost_bi()
    fig5_alpha()
    fig6_robustness()
    fig7_ablation()
    fig8_policy()
    print('figures ->', FIGS, flush=True)


if __name__ == '__main__':
    main()
