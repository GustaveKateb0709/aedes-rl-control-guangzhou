#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Publication figures for the manuscript (13-figure scheme).

Reads only pre-computed artefacts in ../results/ and writes, for every
figure, a 600-dpi LZW-compressed TIFF and a vector PDF into ../figures_v2/.

Design constraints (Array official + author rules):
  * NO title / caption text inside any figure (no suptitle, no ax.set_title).
  * Multi-panel figures: the (a)(b)(c) letters live in the *x-axis title* of
    each panel, below the panel, never inside the axes.
  * Restrained, low-saturation palette.
  * Axis units spelled out; legends outside the data region.
  * No PNG deliverables (a low-res PNG is written to /tmp for QA only).

Final mapping (filename -> content):
  Figure1  decision-framework schematic            (with weekly ceiling layer)
  Figure2  PPO training convergence, NO cap, 10 seeds
  Figure3  2022 return vs C_max by arm             (standalone panel)
  Figure4  PPO fraction of weeks at the ceiling     (standalone panel)
  Figure5  (a) return vs action-space size K ; (b) adjacency-feature ablation
  Figure6  leave-one-year-out cross-validation, C_max = 2
  Figure7  counterfactual 2022 weekly trajectories
  Figure8  cost-health plane (mean BI vs total cost, by C_max)
  Figure9  cost-weight alpha sweep
  Figure10 robustness to perturbed intervention dynamics
  Figure11 action-set ablation
  Figure12 learned policy behaviour vs baseline risk
  Figure13 chikungunya transmission potential (a) cumulative (b) weekly
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
from PIL import Image
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(BASE, '..', 'results'))
MON = os.path.join(RES, 'monitor')
OUT = os.path.abspath(os.path.join(BASE, '..', 'figures_v2'))
QA = '/tmp/figqa_v2'
os.makedirs(OUT, exist_ok=True)
os.makedirs(QA, exist_ok=True)

# --------------------------------------------------------------------------
# style / palette (muted)
# --------------------------------------------------------------------------
INK = '#22303c'
GRID = '#d7dde2'
C_NONE = '#9aa3ab'
C_THRESH = '#6e7b87'
C_CAL = '#d7a25c'
C_MPC = '#7fa39c'
C_PPO = '#3f6d8c'
C_A2C = '#8c9a5c'
C_DQN = '#a9825b'
C_SR = '#4a7fa5'
C_SP = '#b98a6e'
CALIB_SCALE = 0.3476

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.labelsize': 9.5,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.2,
    'axes.linewidth': 0.9,
    'axes.edgecolor': INK,
    'text.color': INK,
    'axes.labelcolor': INK,
    'xtick.color': INK,
    'ytick.color': INK,
    'figure.dpi': 120,
    'savefig.facecolor': 'white',
})


def tidy(ax, grid='y'):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, alpha=0.85, linewidth=0.6)
        ax.set_axisbelow(True)


def save(fig, name):
    # vector PDF
    fig.savefig(os.path.join(OUT, name + '.pdf'))
    # 600-dpi LZW TIFF, forced to RGB (Array: no alpha channel)
    fig.set_dpi(600)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    Image.fromarray(buf[..., :3]).save(
        os.path.join(OUT, name + '.tif'), format='TIFF',
        compression='tiff_lzw', dpi=(600, 600))
    fig.savefig(os.path.join(QA, name + '.png'), dpi=115)
    plt.close(fig)
    print('  saved', name, flush=True)


def load(p):
    return pd.read_csv(os.path.join(RES, p))


def ckey(v):
    return 'inf' if np.isinf(float(v)) else str(int(float(v)))


# ==========================================================================
# Figure 1 - decision-framework schematic
# ==========================================================================
def fig1_framework():
    fig, ax = plt.subplots(figsize=(11.6, 6.0))
    ax.set_xlim(0, 130); ax.set_ylim(0, 56); ax.axis('off')
    W, H, CY = 22.5, 16.0, 45.0
    xs = np.linspace(15.0, 116.0, 5)
    stages = [
        (xs[0], 'ERA5 reanalysis\nweather',
         '2015 to 2022,\nGuangzhou\n10-grid domain', C_NONE, '#f2f4f5'),
        (xs[1], 'PFI vector\ndynamics model',
         'weekly baseline BI,\ncalibrated to\nsurveillance\n(mean 3.7)', C_MPC, '#eef4f3'),
        (xs[2], 'Grid-week MDP\n(MosquitoEnv)',
         'state: PFI, density,\nintervention effects;\nactions 0 / 1 / 2\nper grid', C_CAL, '#faf3e8'),
        (xs[3], 'Weekly budget\nfilter',
         'sum_g cost_g(t) <= C_max\n-> feasible joint-action\nsubset  (e.g. K = 56\nat C_max = 2)', C_THRESH, '#eef0f2'),
        (xs[4], 'PPO agent',
         'chooses one action\nper grid\n\n\n', C_PPO, '#ecf2f6'),
    ]
    for x, head, body, ec, fc in stages:
        ax.add_patch(FancyBboxPatch((x - W / 2, CY - H / 2), W, H,
                                    boxstyle='round,pad=0.02,rounding_size=1.1',
                                    lw=1.4, edgecolor=ec, facecolor=fc))
        ax.text(x, CY + H / 2 - 3.2, head, ha='center', va='center',
                fontsize=9.4, fontweight='bold', color=INK, linespacing=1.25)
        ax.text(x, CY - 3.4, body, ha='center', va='center',
                fontsize=7.1, color=INK, linespacing=1.35)
    for i in range(4):
        ax.add_patch(FancyArrowPatch((xs[i] + W / 2 + 0.5, CY),
                                     (xs[i + 1] - W / 2 - 0.5, CY),
                                     arrowstyle='-|>', mutation_scale=15,
                                     lw=1.8, color=C_PPO))
    RB_W, RB_H, RB_CY, RB_CX = 86.0, 13.0, 13.5, 62.0
    ax.add_patch(FancyBboxPatch((RB_CX - RB_W / 2, RB_CY - RB_H / 2), RB_W, RB_H,
                                boxstyle='round,pad=0.02,rounding_size=1.1',
                                lw=1.4, edgecolor=C_NONE, facecolor='#f2f4f5'))
    ax.text(RB_CX, RB_CY + 3.0, 'Reward', ha='center', va='center',
            fontsize=9.6, fontweight='bold', color=INK)
    ax.text(RB_CX, RB_CY - 2.2,
            r'$R_t \;=\; [\,BI_t - BI_{thr}\,]_{+}^{2} \;-\; \alpha \sum_g cost_g(t)$'
            '\nthreshold-excess morbidity  minus  scaled intervention cost',
            ha='center', va='center', fontsize=8.0, color=INK, linespacing=1.5)
    ax.add_patch(FancyArrowPatch((xs[2] - 6.0, CY - H / 2 - 0.3),
                                 (RB_CX - 12.0, RB_CY + RB_H / 2 + 0.3),
                                 arrowstyle='-|>', mutation_scale=13, lw=1.4, color=C_NONE))
    ax.text(xs[2] - 8.6, (CY - H / 2 + RB_CY + RB_H / 2) / 2, 'state',
            ha='right', va='center', fontsize=8.0, color=C_NONE)
    ax.add_patch(FancyArrowPatch((RB_CX + 24.0, RB_CY + RB_H / 2 + 0.3),
                                 (xs[4] - 3.0, CY - H / 2 - 0.3),
                                 arrowstyle='-|>', mutation_scale=13, lw=1.4, color=C_PPO,
                                 connectionstyle='arc3,rad=-0.12'))
    ax.text(RB_CX + 30.0, (RB_CY + RB_H / 2 + CY - H / 2) / 2, 'reward',
            ha='left', va='center', fontsize=8.0, color=C_PPO)
    save(fig, 'Figure1')


# ==========================================================================
# Figure 2 - PPO training convergence, NO weekly ceiling, ten seeds
# ==========================================================================
def fig2_learning_curves():
    files = [f for f in glob.glob(os.path.join(MON, 'ppo_cinf_s*.monitor.csv'))
             if '_adj' not in os.path.basename(f)]
    files = sorted(files, key=lambda f: int(os.path.basename(f).split('_s')[1].split('.')[0]))
    grid = np.arange(0, 400001, 500.0)
    stack = []
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for f in files:
        d = pd.read_csv(f, skiprows=1)
        steps = np.cumsum(d['l'].to_numpy(dtype=float))
        r = d['r'].rolling(200, min_periods=40).mean().to_numpy(dtype=float)
        m = ~np.isnan(r)
        ax.plot(steps[m], r[m], color='#a9c2d4', lw=0.7, alpha=0.85, zorder=2)
        stack.append(np.interp(grid, steps[m], r[m]))
    mean = np.mean(np.vstack(stack), axis=0)
    ax.plot(grid, mean, color=C_PPO, lw=2.4, zorder=4, label='seed mean (n = %d)' % len(files))
    ax.plot([], [], color='#a9c2d4', lw=1.0, label='individual seeds')
    ax.set_xlim(0, 400000)
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Episode reward\n(200-episode moving average)')
    ax.legend(loc='lower right', frameon=False)
    tidy(ax)
    save(fig, 'Figure2')


# ==========================================================================
# Figure 3 - 2022 episode return vs C_max, by arm (standalone)
# ==========================================================================
def fig3_cap_return():
    s = load('capacity_summary.csv')
    s['Cmax_num'] = s['C_max'].apply(lambda c: np.inf if str(c) == 'inf' else float(c))
    xmap = {2.0: 2, 4.0: 4, 6.0: 6, 8.0: 8, 12.0: 12, np.inf: 14.5}
    arm_style = [('ppo', 'PPO', C_PPO, 'o'), ('a2c', 'A2C', C_A2C, 's'),
                 ('dqn', 'DQN', C_DQN, '^'), ('mpc1', 'MPC-1 (one-step)', C_MPC, 'D'),
                 ('calendar', 'Calendar rule', C_CAL, 'p'),
                 ('threshold', 'Threshold rule', C_THRESH, 'v'),
                 ('none', 'No intervention', C_NONE, '*')]
    dodge = {a[0]: off for a, off in zip(arm_style, np.linspace(-0.42, 0.42, len(arm_style)))}
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    handles = []
    for key, lab, col, mk in arm_style:
        sub = s[s['arm'] == key].sort_values('Cmax_num')
        if sub.empty:
            continue
        xx = sub['Cmax_num'].map(xmap).to_numpy(dtype=float) + dodge[key]
        h = ax.errorbar(xx, sub['ret_mean'], yerr=sub['ret_sd'], fmt=mk, color=col,
                        ms=6.5, lw=1.0, capsize=2.3, elinewidth=1.0, zorder=3, label=lab)
        handles.append(h)
    ax.set_xticks([2, 4, 6, 8, 12, 14.5])
    ax.set_xticklabels(['2', '4', '6', '8', '12', 'no cap'])
    ax.axvline(13.25, color=GRID, lw=1.0)
    ax.set_xlim(0.9, 16.1)
    ax.set_xlabel('Weekly ceiling  C$_{max}$  (relative cost units; SR = 1, SP = 3)')
    ax.set_ylabel('2022 episode return (mean $\\pm$ sd)')
    tidy(ax)
    fig.legend(handles=handles, labels=[a[1] for a in arm_style],
               loc='lower center', ncol=4, frameon=False, fontsize=8.0,
               bbox_to_anchor=(0.5, -0.005), columnspacing=1.4, handletextpad=0.5)
    fig.subplots_adjust(bottom=0.22)
    save(fig, 'Figure3')


# ==========================================================================
# Figure 4 - share of weeks the ceiling binds (PPO), standalone
# ==========================================================================
def fig4_cap_bind():
    bu = load('capacity_budget_use_summary.csv')
    xmap = {2.0: 2, 4.0: 4, 6.0: 6, 8.0: 8, 12.0: 12}
    b = bu[bu['arm'] == 'ppo'].copy()
    b['Cmax_num'] = b['C_max'].apply(lambda c: np.inf if str(c) == 'inf' else float(c))
    b = b[b['Cmax_num'] != np.inf].sort_values('Cmax_num')
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    xx = b['Cmax_num'].map(xmap)
    ax.plot(xx, b['frac_weeks_at_cap'], 'o-', color=C_PPO, lw=1.9, ms=6.5,
            label='Weeks exactly at the ceiling')
    ax.plot(xx, b['frac_weeks_zero'], 's--', color=C_MPC, lw=1.7, ms=6.0,
            label='Weeks with no intervention')
    for x, yv in zip(xx, b['frac_weeks_at_cap']):
        ax.annotate('%.3f' % yv, (x, yv), textcoords='offset points',
                    xytext=(0, 7), ha='center', fontsize=7.6, color=INK)
    ax.set_xticks([2, 4, 6, 8, 12]); ax.set_xticklabels(['2', '4', '6', '8', '12'])
    ax.set_xlim(1, 13); ax.set_ylim(-0.03, 0.60)
    ax.set_xlabel('Weekly ceiling  C$_{max}$  (no-cap value undefined: constraint inactive)')
    ax.set_ylabel('Fraction of weeks (PPO policy)')
    ax.legend(loc='upper right', frameon=False, fontsize=8.2)
    tidy(ax)
    fig.subplots_adjust(bottom=0.20)
    save(fig, 'Figure4')


# ==========================================================================
# Figure 5 - (a) return vs action-space size K ; (b) adjacency ablation
# ==========================================================================
def fig5_action_space():
    ps = load('kconfound_summary.csv')
    sp = load('spatial_adjacency_summary.csv')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.7),
                                   gridspec_kw={'width_ratios': [1.15, 1]})

    # (a) return vs flattened action-space size K
    K = {'nest56': (56, 2.0), 'nest486': (486, 4.0), 'nest6498': (6498, 8.0)}
    cmap = {2.0: '#2f6f8f', 4.0: '#5a9b8e', 8.0: '#d1a04a'}
    ks, rs, es, cs = [], [], [], []
    for arm, (k, cm) in K.items():
        r = ps[ps['arm'] == arm]
        if r.empty:
            continue
        ks.append(k); rs.append(r['ret_mean'].iloc[0]); es.append(r['ret_sd'].iloc[0]); cs.append(cmap[cm])
    for k, r, e, c in zip(ks, rs, es, cs):
        ax1.errorbar(k, r, yerr=e, fmt='o', color=c, ms=7.5, capsize=3.5,
                     elinewidth=1.2, zorder=4)
    # connecting guide: the flattened nested sequence 56 -> 486 -> 6498 -> 59049
    XF_L, XF_R = 59049 * 0.975, 59049 * 1.025   # +-2.5% jitter, both hug K=59049
    gx, gy = [], []
    for arm, k in [('nest56', 56), ('nest486', 486), ('nest6498', 6498)]:
        rr = ps[ps['arm'] == arm]
        if not rr.empty:
            gx.append(k); gy.append(rr['ret_mean'].iloc[0])
    nfull = ps[ps['arm'] == 'nest59049']
    if not nfull.empty:
        gx.append(XF_L); gy.append(float(nfull['ret_mean'].iloc[0]))
    ax1.plot(gx, gy, color='#8a97a1', lw=1.1, alpha=0.7, zorder=2)
    # The two arms share K = 59049: they are placed at the same K (only a
    # 2.5% jitter to keep the markers from coinciding).  They are already
    # 2.1 return units apart vertically, so they never overlap.
    if not nfull.empty:
        ax1.errorbar(XF_L, nfull['ret_mean'].iloc[0], yerr=nfull['ret_sd'].iloc[0],
                     fmt='v', color='#a0522d', ms=8, capsize=3.5, elinewidth=1.2, zorder=5)
        ax1.annotate('flattened full space\n(n = 3)', (XF_L, nfull['ret_mean'].iloc[0]),
                     xytext=(-9, -3), textcoords='offset points', ha='right', va='top',
                     fontsize=7.0, color='#a0522d')
    c2 = ps[ps['arm'] == 'c2proj']
    if not c2.empty:
        ax1.errorbar(XF_R, c2['ret_mean'].iloc[0], yerr=c2['ret_sd'].iloc[0],
                     fmt='s', color='#5f7f6a', ms=7, capsize=3.5, elinewidth=1.2, zorder=5)
        ax1.annotate('flattened full space,\nC$_{max}$ = 2 + projection\n(n = 3)',
                     (XF_R, c2['ret_mean'].iloc[0]),
                     xytext=(2, -9), textcoords='offset points', ha='center', va='top',
                     fontsize=7.0, color='#5f7f6a')
    ax1.set_xscale('log')
    ax1.set_xticks([56, 486, 6498, 59049])
    ax1.set_xticklabels(['56', '486', '6,498', '59,049'])
    ax1.set_xlim(45, 125000)
    lows = [r - e for r, e in zip(rs, es)]
    if not nfull.empty:
        lows.append(float(nfull['ret_mean'].iloc[0] - nfull['ret_sd'].iloc[0]))
    if not c2.empty:
        lows.append(float(c2['ret_mean'].iloc[0] - c2['ret_sd'].iloc[0]))
    rnd0 = ps[ps['arm'] == 'rnd56']
    if not rnd0.empty:
        lows.append(float(rnd0['ret_mean'].iloc[0] - rnd0['ret_sd'].iloc[0]))
    ax1.set_ylim(min(lows) - 0.20, max(rs) + 0.30)
    ylo, yhi = ax1.get_ylim()
    # MultiDiscrete reference (factorized per-zone head): horizontal line, not a K point
    md = ps[ps['arm'] == 'md']
    if not md.empty:
        mv = float(md['ret_mean'].iloc[0])
        ax1.axhline(mv, color=C_PPO, ls='--', lw=1.2, zorder=3)
        ax1.text(1500, mv + 0.012, 'MultiDiscrete status quo\n(%.3f, n = 10)' % mv,
                 fontsize=7.4, color=C_PPO, va='bottom', ha='left')
    for k, cm in [(56, 'C$_{max}$ = 2, n = 10'), (486, 'C$_{max}$ = 4, n = 10'),
                  (6498, 'C$_{max}$ = 8, n = 10')]:
        ax1.annotate(cm, (k, ylo + 0.03), ha='center', fontsize=6.9, color=INK)
    rnd = ps[ps['arm'] == 'rnd56']
    if not rnd.empty:
        ax1.errorbar(56, rnd['ret_mean'].iloc[0], yerr=rnd['ret_sd'].iloc[0],
                     fmt='X', color='#b0533f', ms=8, capsize=3.0, elinewidth=1.1, zorder=5)
        ax1.annotate('random projection, K = 56\n(n = 3, negative control)',
                     (56, rnd['ret_mean'].iloc[0]), xytext=(6, 0),
                     textcoords='offset points', ha='left', va='center',
                     fontsize=6.9, color='#b0533f')
    ax1.set_xlabel('(a) Feasible joint-action count K (log scale;\n'
                   'flattened Discrete(K) action head).\n'
                   'The two rightmost markers share K = 59,049.')
    ax1.set_ylabel('2022 episode return (mean $\\pm$ sd)')
    tidy(ax1)

    # (b) spatial adjacency feature ablation
    sp = sp.copy(); sp['__ck'] = sp['C_max'].map(ckey)
    order = [('2', 'C$_{max}$ = 2'), ('inf', 'no cap')]
    x = np.arange(2); w = 0.36; allret = []
    for i, (cm, _) in enumerate(order):
        sub = sp[sp['__ck'] == cm]
        b = sub[sub['features'] == 'baseline']; a = sub[sub['features'] == 'adjacency']
        vb, va = float(b['ret'].iloc[0]), float(a['ret'].iloc[0])
        sdb, sda = float(b['ret_sd'].iloc[0]), float(a['ret_sd'].iloc[0])
        allret += [vb, va]
        ax2.bar(i - w / 2, vb, width=w, yerr=sdb, capsize=3.5, color='#9db2c4',
                edgecolor=INK, linewidth=0.6, zorder=3)
        ax2.bar(i + w / 2, va, width=w, yerr=sda, capsize=3.5, color='#7fa39c',
                edgecolor=INK, linewidth=0.6, zorder=3)
        for xo, v, sd, tag in [(i - w / 2, vb, sdb, 'baseline'),
                               (i + w / 2, va, sda, '+ adjacency')]:
            ax2.text(xo, v + sd + 0.012, '%.3f' % v, ha='center', va='bottom',
                     fontsize=7.0, color=INK)
            ax2.text(xo, -0.055, tag, ha='center', va='top', fontsize=6.6,
                     color='white', rotation=90, zorder=5)
        ax2.text(i, 0.035, 'n = %d' % int(b['n'].iloc[0]), ha='center', fontsize=7.8, color=INK)
    ax2.set_xticks(x); ax2.set_xticklabels([l for _, l in order])
    ax2.axhline(0, color=INK, lw=0.8)
    ax2.set_ylim(min(allret) - 0.10, 0.09)
    ax2.set_xlabel('(b) Observation features $\\times$ weekly ceiling')
    ax2.set_ylabel('2022 episode return (mean $\\pm$ sd; more negative = worse)')
    tidy(ax2)

    fig.subplots_adjust(wspace=0.30, bottom=0.22, left=0.07)
    save(fig, 'Figure5')


# ==========================================================================
# Figure 6 - leave-one-year-out cross-validation (C_max = 2)
# ==========================================================================
def fig6_loyo():
    d = load('loyo_fold_summary.csv')
    d = d[d['C_max'] == 2]
    years = sorted(d['fold_year'].unique())
    style = [('none', 'No intervention / threshold', C_NONE, 2.0, 'o'),
             ('calendar', 'Calendar rule', C_CAL, 1.5, 'p'),
             ('mpc1', 'MPC-1 (one-step)', C_MPC, 1.6, 'D'),
             ('ppo', 'PPO policy', C_PPO, 2.2, 'o')]
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for arm, lab, col, lw, mk in style:
        sub = d[d['arm'] == arm].set_index('fold_year').reindex(years)
        ax.errorbar(sub.index, sub['ret_mean'], yerr=sub['ret_sd'], fmt=mk + '-',
                    color=col, lw=lw, ms=6.2, capsize=2.6, elinewidth=1.0,
                    label=lab, zorder=3)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], fontsize=8.2)
    ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
    ax.set_xlabel('Held-out evaluation year (leave-one-year-out, C$_{max}$ = 2)')
    ax.set_ylabel('Episode return (mean $\\pm$ sd; higher = better)')
    ax.text(0.985, 0.055, 'PPO: n = 100 per fold;  rule arms: n = 10',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=7.4, color=INK)
    ax.legend(loc='lower left', frameon=False, fontsize=8.2)
    tidy(ax)
    fig.subplots_adjust(bottom=0.15, left=0.10)
    save(fig, 'Figure6')


# ==========================================================================
# Figure 7 - counterfactual 2022 weekly trajectories (legacy no-ceiling files)
# ==========================================================================
def fig7_trajectories():
    def series(fn):
        d = load(fn)
        bi = [c for c in d.columns if c.startswith('BI_')]
        return d, d[bi].mean(axis=1).to_numpy()
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    handles = []
    d, y = series('trajectory_none.csv')
    handles.append(ax.plot(d['week_in_year'], y, color=C_NONE, lw=2.3,
                           label='No intervention')[0])
    dt, yt = series('trajectory_threshold.csv')
    handles.append(ax.plot(dt['week_in_year'], yt, color=C_THRESH, lw=1.3, ls='--',
                           label='Threshold rule (never triggers)')[0])
    dc, yc = series('trajectory_calendar.csv')
    handles.append(ax.plot(dc['week_in_year'], yc, color=C_CAL, lw=1.6,
                           label='Preventive calendar rule')[0])
    dp, yp = series('trajectory_ppo_a0.1_s0_acts012.csv')
    handles.append(ax.plot(dp['week_in_year'], yp, color=C_PPO, lw=2.2,
                           label='PPO policy (seed 0)')[0])
    ax2 = ax.twinx()
    handles.append(ax2.bar(dp['week_in_year'], dp['n_sr'], color=C_SR, alpha=0.28,
                           width=0.78, label='PPO source-reduction actions (right axis)'))
    ax2.set_ylabel('Grids under source reduction per week', color=C_SR)
    ax2.tick_params(axis='y', colors=C_SR)
    ax2.set_ylim(0, 42); ax2.spines['top'].set_visible(False)
    ax.axhline(5, color=INK, ls=':', lw=1.0)
    ax.text(2.0, 5.15, 'BI = 5 transmission-risk threshold', fontsize=7.8, color=INK, va='bottom')
    ax.set_xlabel('Week of 2022 (held-out evaluation year)')
    ax.set_ylabel('Mean BI across the ten grids')
    ax.set_ylim(0, 9.0); ax.set_xlim(1, 52)
    tidy(ax)
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
               fontsize=8.0, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.28)
    save(fig, 'Figure7')


# ==========================================================================
# Figure 8 - cost-health plane (standalone)
# ==========================================================================
def fig8_cost_health():
    s = load('capacity_summary.csv')
    s['Cmax_num'] = s['C_max'].apply(lambda c: np.inf if str(c) == 'inf' else float(c))
    cmap = {2.0: '#2f6f8f', 4.0: '#5a9b8e', 6.0: '#8fb06a', 8.0: '#d1a04a',
            12.0: '#c07f5a', np.inf: '#6e6e7a'}
    lbl = {2.0: '2', 4.0: '4', 6.0: '6', 8.0: '8', 12.0: '12', np.inf: 'no cap'}
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for cm, g in s.groupby('Cmax_num'):
        ax.scatter(g['cost'], g['bi_mean'], s=46, color=cmap[cm],
                   edgecolor='white', linewidth=0.6, zorder=3, label=lbl[cm])
    for _, r in s.iterrows():
        ax.annotate(lbl[r['Cmax_num']], (r['cost'], r['bi_mean']),
                    textcoords='offset points', xytext=(5, 3),
                    fontsize=6.4, color=cmap[r['Cmax_num']])
    ax.set_xlabel('Total 2022 control cost (relative units)')
    ax.set_ylabel('Mean BI in 2022')
    ax.set_xlim(-12, 245)
    ax.legend(title='C$_{max}$', loc='upper right', frameon=False,
              fontsize=7.4, title_fontsize=7.6, ncol=2)
    tidy(ax)
    fig.subplots_adjust(bottom=0.13, left=0.11, right=0.97)
    save(fig, 'Figure8')


# ==========================================================================
# Figure 9 - cost-weight alpha sweep
# ==========================================================================
def fig9_alpha():
    d = load('alpha_sweep.csv').sort_values('alpha')
    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    l1, = ax.plot(d['alpha'], d['mean_bi'], 'o-', color=C_PPO, lw=1.9, ms=6.5,
                  label='Mean BI (left axis)')
    ax.set_xlabel('Cost weight  $\\alpha$')
    ax.set_ylabel('Mean BI in 2022 (held-out year)', color=C_PPO)
    ax.tick_params(axis='y', labelcolor=C_PPO)
    ax.set_xlim(-0.02, 0.52); ax.set_ylim(-0.05, 3.0)
    tidy(ax)
    ax2 = ax.twinx()
    l2, = ax2.plot(d['alpha'], d['total_cost'], 's--', color=C_CAL, lw=1.9, ms=6.5,
                   label='Total cost (right axis)')
    ax2.set_ylabel('Total 2022 control cost (relative units)', color=C_CAL)
    ax2.tick_params(axis='y', labelcolor=C_CAL)
    ax2.spines['top'].set_visible(False)
    fig.legend(handles=[l1, l2], loc='lower center', ncol=2, frameon=False,
               fontsize=8.4, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(bottom=0.22)
    save(fig, 'Figure9')


# ==========================================================================
# Figure 10 - robustness to perturbed intervention dynamics
# ==========================================================================
def fig10_robustness():
    d = load('robustness_perturbation.csv')
    nom = float(d.loc[d['scenario'] == 'nominal', 'mean_bi'].iloc[0])
    d = d[d['scenario'] != 'nominal'].copy()
    names = {'sr_efficacy-30%': 'Source red. efficacy  \u2013 30 %',
             'sr_efficacy+30%': 'Source red. efficacy  + 30 %',
             'sp_efficacy-30%': 'Spray efficacy  \u2013 30 %',
             'sp_efficacy+30%': 'Spray efficacy  + 30 %',
             'sr_decay_fast': 'Faster source-red. decay',
             'sp_decay_slow': 'Slower spray decay',
             'both_efficacy-30%': 'Both efficacies  \u2013 30 %'}
    d['label'] = d['scenario'].map(names)
    d = d.sort_values('mean_bi')
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    y = np.arange(len(d))
    colors = ['#c98a6b' if v > nom + 1e-9 else ('#7fa6c9' if v < nom - 1e-9 else '#a9a9b2')
              for v in d['mean_bi']]
    ax.barh(y, d['mean_bi'], color=colors, height=0.62, zorder=2)
    for yi, v in zip(y, d['mean_bi']):
        ax.text(v + 0.02, yi, '%.2f' % v, va='center', ha='left', fontsize=8.4, color=INK, zorder=4)
    ax.axvline(nom, color=INK, ls='--', lw=1.2, zorder=3)
    ax.text(nom + 0.015, len(d) - 0.35, 'nominal = %.2f\n(no retraining)' % nom,
            fontsize=7.8, color=INK, va='top', ha='left')
    ax.set_yticks(y); ax.set_yticklabels(d['label'], fontsize=8.3)
    ax.set_xlim(0, d['mean_bi'].max() * 1.12); ax.set_ylim(-0.6, len(d) - 0.2)
    ax.set_xlabel('Mean BI in 2022, fixed trained policy (no retraining)')
    tidy(ax, grid='x')
    fig.subplots_adjust(left=0.34, right=0.97)
    save(fig, 'Figure10')


# ==========================================================================
# Figure 11 - action-set ablation
# ==========================================================================
def fig11_ablation():
    d = load('action_ablation.csv')
    order = ['(0, 1, 2)', '(0, 1)', '(0, 2)']
    lab = {'(0, 1, 2)': 'Both tools', '(0, 1)': 'Source reduction only', '(0, 2)': 'Spraying only'}
    col = {'(0, 1, 2)': C_PPO, '(0, 1)': '#7fa6c9', '(0, 2)': C_SP}
    d = d.set_index('action_set').loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    x = np.arange(len(d))
    ax.bar(x, d['mean_bi'], color=[col[a] for a in d['action_set']], width=0.56, zorder=2)
    for xi, v, c in zip(x, d['mean_bi'], d['total_cost']):
        ax.text(xi, v + 0.05, '%.2f\ncost %.1f' % (v, c), ha='center', va='bottom',
                fontsize=8.4, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([lab[a] for a in d['action_set']])
    ax.set_ylim(0, d['mean_bi'].max() * 1.32)
    ax.set_ylabel('Mean BI in 2022 (held-out year)')
    tidy(ax)
    fig.subplots_adjust(bottom=0.14)
    save(fig, 'Figure11')


# ==========================================================================
# Figure 12 - learned policy behaviour vs baseline risk
# ==========================================================================
def fig12_policy():
    d = load('trajectory_ppo_a0.1_s0_acts012.csv')
    panel = load('panel_data.csv')
    p22 = panel[panel['year'] == 2022].copy()
    p22['mu'] = p22['PFI'] * p22['pop_density'] * CALIB_SCALE
    baseline = p22.groupby('week_in_year')['mu'].mean()
    j = d.set_index('week_in_year').join(baseline.rename('baseline'))
    bins = [0, 2, 4, 6, 8, 12]
    labels = ['0\u20132', '2\u20134', '4\u20136', '6\u20138', '8\u201312']
    j['bin'] = pd.cut(j['baseline'], bins=bins, labels=labels)
    g = j.groupby('bin', observed=True)[['n_sr', 'n_sp']].mean()
    counts = j.groupby('bin', observed=True).size()
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    x = np.arange(len(g))
    ax.bar(x - 0.19, g['n_sr'], width=0.38, color=C_SR, label='Source reduction', zorder=2)
    ax.bar(x + 0.19, g['n_sp'], width=0.38, color=C_SP,
           label='Space spraying (never selected, $\\alpha$ = 0.1)', zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(['%s\n(n = %d)' % (l, int(counts.get(l, 0))) for l in g.index], fontsize=8.2)
    ax.set_xlabel('Baseline mean BI of the week (calibrated, no-intervention level)')
    ax.set_ylabel('Mean number of grids treated per week')
    ax.set_ylim(0, 4.2)
    ax.legend(loc='upper left', frameon=False, fontsize=8.2)
    tidy(ax)
    fig.subplots_adjust(bottom=0.20, left=0.11)
    save(fig, 'Figure12')


# ==========================================================================
# Figure 13 - chikungunya transmission potential (two panels)
# ==========================================================================
def fig13_transmission():
    raw = np.load(os.path.join(RES, 'transmission_weekly.npy'), allow_pickle=True).item()
    d = {(float(p), arm): (float(tot), np.asarray(wk, dtype=float))
         for (p, arm), (tot, wk) in raw.items()}
    p = 0.90
    arms = ['none', 'threshold', 'calendar', 'ppo_s0']
    labels = ['No\nintervention', 'Threshold\n(never fires)', 'Calendar\nrule', 'PPO\npolicy']
    colors = [C_NONE, C_THRESH, C_CAL, C_PPO]
    tots = [d[(p, a)][0] for a in arms]
    ref = tots[0]
    seeds = [d[(p, 'ppo_s%d' % s)][0] for s in (0, 1, 2)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.7),
                                   gridspec_kw={'width_ratios': [1, 1.25]})
    x = np.arange(len(arms))
    bars = ax1.bar(x, tots, color=colors, width=0.62, zorder=2)
    bars[1].set_hatch('//'); bars[1].set_edgecolor('white')
    ax1.plot([3, 3], [tots[3], max(seeds)], color=INK, lw=1.2, zorder=5)
    ax1.plot([2.9, 3.1], [tots[3], tots[3]], color=INK, lw=1.2, zorder=5)
    ax1.plot([2.9, 3.1], [max(seeds), max(seeds)], color=INK, lw=1.2, zorder=5)
    for i, t in enumerate(tots):
        ytop = max(seeds) if i == 3 else t
        ax1.text(i, ytop + ref * 0.03, '%.0f\n(%.1f%%)' % (t, 100 * t / ref),
                 ha='center', va='bottom', fontsize=7.8, color=INK)
    ax1.text(3.48, ref * 1.33, 'PPO bar = seed 0;\nline = range, seeds 0\u20132',
             ha='right', va='top', fontsize=6.9, color=INK)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8.2)
    ax1.set_ylim(0, ref * 1.38)
    ax1.set_xlabel('(a) Control arm')
    ax1.set_ylabel('Cumulative transmission potential, 2022\n(relative index)')
    tidy(ax1)
    for arm, lab, c, lw in [('none', 'No intervention / threshold', C_NONE, 1.6),
                            ('calendar', 'Calendar rule', C_CAL, 1.5),
                            ('ppo_s0', 'PPO policy (seed 0)', C_PPO, 1.9)]:
        wk = d[(p, arm)][1]
        ax2.plot(np.arange(1, len(wk) + 1), wk, color=c, lw=lw, label=lab)
    ax2.set_xlabel('(b) Week of the 2022 evaluation year')
    ax2.set_ylabel('Weekly transmission potential (relative index)')
    ax2.set_ylim(0, 46)
    ax2.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=8.0)
    tidy(ax2)
    fig.subplots_adjust(wspace=0.30, bottom=0.22, left=0.09)
    save(fig, 'Figure13')


def main():
    print('generating final 13-figure set ->', OUT, flush=True)
    fig1_framework()
    fig2_learning_curves()
    fig3_cap_return()
    fig4_cap_bind()
    fig5_action_space()
    fig6_loyo()
    fig7_trajectories()
    fig8_cost_health()
    fig9_alpha()
    fig10_robustness()
    fig11_ablation()
    fig12_policy()
    fig13_transmission()
    print('done.', flush=True)


if __name__ == '__main__':
    main()
