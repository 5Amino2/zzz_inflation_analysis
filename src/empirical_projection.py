#!/usr/bin/env python3
"""
Chapter 6: Empirical Prediction Computations
基于线性膨胀模型的分数衰减预测 — 数值计算与预测图表

Supports two player types:
  - Type A (满分/竞速玩家): parameterized by clear time T_full or T_base
  - Type B (奖励线玩家): parameterized by current score S at the current period/node

Outputs:
  1. Prediction tables for Chapter 6 of zzz_inflation_analysis.md
  2. Prediction charts (fig9-fig12) to charts/prediction/
"""

import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import sys, io, os
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Chart output directory
PREDICTION_CHART_DIR = os.path.join(os.path.dirname(__file__), '..', 'charts', 'prediction')

# Matplotlib Chinese font setup
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. DATA: Regression parameters from appendix
# ============================================================

# 危局强袭战: per-period average HP regression (41 periods)
# HP(n) = k * n + b (million HP)
HZ_k = 3.944
HZ_b = 47.233
HZ_HP1 = HZ_k + HZ_b  # 51.177 million at period 1

# 式舆防卫战: per-node average HP regression (13 nodes)
# HP(n) = k * n + b (million HP)
SD_k = 1.089
SD_b = 24.550
SD_HP1 = SD_k + SD_b  # 25.639 million at node 1

# Current baseline
HZ_CURRENT_PERIOD = 41
SD_CURRENT_NODE = 13

def alpha_hz(period):
    """Equivalent inflation coefficient for 危局 at given period."""
    hp_n = HZ_k * period + HZ_b
    return hp_n / HZ_HP1 - 1.0

def alpha_sd(node):
    """Equivalent inflation coefficient for 式舆 at given node."""
    hp_n = SD_k * node + SD_b
    return hp_n / SD_HP1 - 1.0

print("=== Current Inflation Levels ===")
print(f"危局 第{HZ_CURRENT_PERIOD}期: alpha = {alpha_hz(HZ_CURRENT_PERIOD)*100:.1f}%")
print(f"式舆 节点{SD_CURRENT_NODE}: alpha = {alpha_sd(SD_CURRENT_NODE)*100:.1f}%")
print()

# ============================================================
# 2. SCORE FUNCTIONS
# ============================================================

# --- 危局强袭战: piecewise linear ---
hz_segments = [
    (1.2, 4, 1000),   # bars 29-26
    (1.7, 4, 1200),   # 25-22
    (2.2, 4, 1800),   # 21-18
    (2.5, 4, 2400),   # 17-14
    (3.0, 4, 2600),   # 13-10
    (5.0, 3, 2600),   # 9-7
    (5.0, 6, 2700),   # 6-1
]

hz_cum_hp = [0.0]
hz_cum_score = [0.0]
for mult, bars, score_per_bar in hz_segments:
    for _ in range(bars):
        hz_cum_hp.append(hz_cum_hp[-1] + mult)
        hz_cum_score.append(hz_cum_score[-1] + score_per_bar)

HZ_TOTAL_HP = hz_cum_hp[-1]       # 87.4
HZ_TOTAL_DMG_SCORE = hz_cum_score[-1]  # 60000
HZ_TIME_LIMIT = 180.0
HZ_TECH_SCORE = 5000.0

def score_hz(alpha, T_full):
    """危局 score for given inflation and base full-clear time."""
    frac = HZ_TIME_LIMIT / ((1 + alpha) * T_full)
    if frac >= 1.0:
        return HZ_TOTAL_DMG_SCORE + HZ_TECH_SCORE
    hp_cleared = frac * HZ_TOTAL_HP
    for i in range(len(hz_cum_hp) - 1):
        if hz_cum_hp[i] <= hp_cleared < hz_cum_hp[i+1]:
            bar_frac = (hp_cleared - hz_cum_hp[i]) / (hz_cum_hp[i+1] - hz_cum_hp[i])
            dmg_score = hz_cum_score[i] + bar_frac * (hz_cum_score[i+1] - hz_cum_score[i])
            return dmg_score + HZ_TECH_SCORE
    return HZ_TOTAL_DMG_SCORE + HZ_TECH_SCORE

def effective_Tfull_from_score(target_score, alpha):
    """Find T_full such that score_hz(alpha, T_full) = target_score.
    Uses binary search since score_hz is monotonic in T_full."""
    lo, hi = 0.5, 3000.0
    # Ensure root is bracketed
    if score_hz(alpha, lo) < target_score or score_hz(alpha, hi) > target_score:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        s = score_hz(alpha, mid)
        if s > target_score:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

# --- 式舆防卫战: time-weighted integral ---
time_boundaries = [0, 60, 70, 80, 90, 105, 120, 135, 150, 300]
time_rates = [5.0, 4.2, 3.5, 3.0, 2.5, 2.0, 1.6, 1.3, 1.0]
SD_TIME_LIMIT = 300.0

def avg_multiplier(t_start, t_end):
    """Average multiplier over [t_start, t_end]."""
    if t_start >= t_end:
        return 0.0
    total = 0.0
    for i in range(len(time_boundaries) - 1):
        a = max(t_start, time_boundaries[i])
        b = min(t_end, time_boundaries[i+1])
        if a < b:
            total += time_rates[i] * (b - a)
    return total / (t_end - t_start)

def score_sd(alpha, T_base):
    """式舆 score for given inflation and base clear time."""
    total_time = (1 + alpha) * T_base
    if total_time > SD_TIME_LIMIT:
        # Incomplete: scale boss portion
        boss_start = total_time * 0.25
        boss_end = min(total_time, SD_TIME_LIMIT)
        boss_duration = boss_end - boss_start
        boss_original = total_time * 0.75
        boss_frac = boss_duration / boss_original if boss_original > 0 else 0.0
        m_bar = avg_multiplier(boss_start, boss_end)
        return 10000.0 + 8000.0 * m_bar * boss_frac
    t_start = total_time * 0.25
    t_end = total_time
    m_bar = avg_multiplier(t_start, t_end)
    return 10000.0 + 8000.0 * m_bar

def effective_Tbase_from_score(target_score, alpha):
    """Find T_base such that score_sd(alpha, T_base) = target_score.
    Uses binary search since score_sd is monotonic in T_base."""
    lo, hi = 0.5, 3000.0
    if score_sd(alpha, lo) < target_score or score_sd(alpha, hi) > target_score:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        s = score_sd(alpha, mid)
        if s > target_score:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

# Verify score functions
assert abs(score_hz(0, 180) - 65000) < 1, f"score_hz(0,180)={score_hz(0,180)}"
assert abs(score_sd(0, 60) - 50000) < 1, f"score_sd(0,60)={score_sd(0,60)}"
print("Score function verification: PASSED\n")


# ============================================================
# 3. TYPE A (满分/竞速玩家) — 预测工具：按通关时间查询
# ============================================================

print("=" * 80)
print("TYPE A: 满分/竞速玩家 (预测工具 — 评判标准 = 通关时间)")
print("=" * 80)

# --- 危局 Type A ---
hz_typeA_T_values = list(range(15, 181, 15))  # 15, 30, 45, ..., 180
hz_alpha_curr = alpha_hz(HZ_CURRENT_PERIOD)

print(f"\n危局强袭战 Type A (当前第{HZ_CURRENT_PERIOD}期, alpha={hz_alpha_curr*100:.1f}%):")
print(f"{'T_full':>8s} {'当前通关时间':>12s} {'当前状态':>10s} {'满分掉分期':>10s} {'距当前':>10s}")
print("-" * 55)

for Tf in hz_typeA_T_values:
    t_curr = (1 + hz_alpha_curr) * Tf
    if t_curr <= HZ_TIME_LIMIT:
        status = "满分★"
    else:
        status = "已掉✗"

    # Find full-score drop period
    full_drop = None
    for p in range(1, 301):
        a = alpha_hz(p)
        if (1 + a) * Tf > HZ_TIME_LIMIT:
            full_drop = p
            break

    drop_str = f"第{full_drop}期" if full_drop else "永不"
    remain = full_drop - HZ_CURRENT_PERIOD if full_drop else "—"

    print(f"{Tf:>6d}s  {t_curr:>10.0f}s  {status:>10s}  {drop_str:>10s}  {str(remain):>10s}")

# --- 式舆 Type A ---
sd_typeA_T_values = list(range(5, 61, 5))  # 5, 10, 15, ..., 60
sd_alpha_curr = alpha_sd(SD_CURRENT_NODE)
SD_FULL_TIME = 60.0

print(f"\n式舆防卫战 Type A (当前节点{SD_CURRENT_NODE}, alpha={sd_alpha_curr*100:.1f}%):")
print(f"{'T_base':>8s} {'当前通关时间':>12s} {'当前状态':>10s} {'满分掉节点':>10s} {'距当前':>10s}")
print("-" * 55)

for Tb in sd_typeA_T_values:
    t_curr = (1 + sd_alpha_curr) * Tb
    if t_curr <= SD_FULL_TIME:
        status = "满分★"
    else:
        status = "已掉✗"

    # Find full-score drop node
    full_drop = None
    for n in range(1, 101):
        a = alpha_sd(n)
        if (1 + a) * Tb > SD_FULL_TIME:
            full_drop = n
            break

    drop_str = f"节点{full_drop}" if full_drop else "永不"
    remain = full_drop - SD_CURRENT_NODE if full_drop else "—"

    print(f"{Tb:>6d}s  {t_curr:>10.0f}s  {status:>10s}  {drop_str:>10s}  {str(remain):>10s}")


# ============================================================
# 4. TYPE B (奖励线玩家) — 预测工具：按当前分数查询
# ============================================================

print()
print("=" * 80)
print("TYPE B: 奖励线玩家 (预测工具 — 评判标准 = 当前分数)")
print("=" * 80)

# --- 危局 Type B ---
hz_typeB_scores = [60000, 55000, 50000, 45000, 40000, 35000, 30000, 25000, 22000]
HZ_REWARD = 20000

print(f"\n危局强袭战 Type B (当前第{HZ_CURRENT_PERIOD}期, alpha={hz_alpha_curr*100:.1f}%):")
print(f"  (奖励线 = {HZ_REWARD}分)")
print(f"{'当前分数':>10s} {'等效T_full':>12s} {'奖励线掉分期':>12s} {'距当前':>10s}")
print("-" * 50)

hz_typeB_results = []
for S_curr in hz_typeB_scores:
    T_eff = effective_Tfull_from_score(S_curr, hz_alpha_curr)
    if T_eff is None:
        print(f"  {S_curr:>8.0f}  {'无法求解':>12s}")
        continue

    # Find reward-line drop period
    reward_drop = None
    for p in range(HZ_CURRENT_PERIOD, 501):
        a = alpha_hz(p)
        s = score_hz(a, T_eff)
        if s < HZ_REWARD:
            reward_drop = p
            break

    drop_str = f"第{reward_drop}期" if reward_drop else "永不"
    remain = reward_drop - HZ_CURRENT_PERIOD if reward_drop else "—"

    print(f"{S_curr:>8.0f}  {T_eff:>10.1f}s  {drop_str:>12s}  {str(remain):>10s}")

    hz_typeB_results.append({
        'S_curr': S_curr, 'T_eff': T_eff, 'reward_drop': reward_drop, 'remain': remain
    })

# --- 式舆 Type B ---
sd_typeB_scores = [49000, 47000, 45000, 42000, 39000, 36000, 33000, 30000, 27000]
SD_REWARD = 25000

print(f"\n式舆防卫战 Type B (当前节点{SD_CURRENT_NODE}, alpha={sd_alpha_curr*100:.1f}%):")
print(f"  (奖励线 = {SD_REWARD}分)")
print(f"{'当前分数':>10s} {'等效T_base':>12s} {'奖励线掉节点':>12s} {'距当前':>10s}")
print("-" * 50)

sd_typeB_results = []
for S_curr in sd_typeB_scores:
    T_eff = effective_Tbase_from_score(S_curr, sd_alpha_curr)
    if T_eff is None:
        print(f"  {S_curr:>8.0f}  {'无法求解':>12s}")
        continue

    # Find reward-line drop node
    reward_drop = None
    for n in range(SD_CURRENT_NODE, 201):
        a = alpha_sd(n)
        s = score_sd(a, T_eff)
        if s < SD_REWARD:
            reward_drop = n
            break

    drop_str = f"节点{reward_drop}" if reward_drop else "永不"
    remain = reward_drop - SD_CURRENT_NODE if reward_drop else "—"

    print(f"{S_curr:>8.0f}  {T_eff:>10.1f}s  {drop_str:>12s}  {str(remain):>10s}")

    sd_typeB_results.append({
        'S_curr': S_curr, 'T_eff': T_eff, 'reward_drop': reward_drop, 'remain': remain
    })


# ============================================================
# 5. OVERVIEW TABLES: 分数预测演化总览
# ============================================================

print()
print("=" * 80)
print("预测总览表1: 危局分数随期数演化预测")
print("=" * 80)

hz_overview_T = [30, 45, 60, 75, 90, 120, 150, 180]
hz_overview_periods = [1, 5, 10, 20, 30, 40, 50, 60]

header = f"{'T_full':>8s}"
for p in hz_overview_periods:
    header += f" {'P'+str(p):>8s}"
print(header)
print("-" * len(header))

for Tf in hz_overview_T:
    row = f"{Tf:>6d}s  "
    for p in hz_overview_periods:
        a = alpha_hz(p)
        s = score_hz(a, Tf)
        if s >= 64999:
            row += f" {'65000★':>8s}"
        elif s >= 20000:
            row += f" {s:>8.0f}"
        else:
            row += f" {s:>8.0f}"
    print(row)

print("\n★ = 满分(>=65000), 其他 >=20000(奖励线以上)")

print()
print("=" * 80)
print("预测总览表2: 式舆分数随节点演化预测")
print("=" * 80)

sd_overview_T = list(range(20, 130, 10))  # 20, 30, 40, ..., 120
sd_overview_nodes = [1] + list(range(5, 45, 5))  # 1, 5, 10, 15, 20, 25, 30, 35, 40

header = f"{'T_base':>8s}"
for n in sd_overview_nodes:
    header += f" {'N'+str(n):>8s}"
print(header)
print("-" * len(header))

for Tb in sd_overview_T:
    row = f"{Tb:>6d}s  "
    for n in sd_overview_nodes:
        a = alpha_sd(n)
        s = score_sd(a, Tb)
        if s >= 49999:
            row += f" {'50000★':>8s}"
        elif s >= 25000:
            row += f" {s:>8.0f}"
        else:
            row += f" {s:>8.0f}"
    print(row)

print("\n★ = 满分(>=50000), 其他 >=25000(奖励线以上)")


# ============================================================
# 6. MARKDOWN TABLES FOR CHAPTER 6
# ============================================================

print()
print("=" * 80)
print("MARKDOWN TABLES FOR CHAPTER 6")
print("=" * 80)

# --- Table 1: 危局 Overview ---
print("\n### 表1: 危局强袭战 -- 分数随期数演化")
print()
header = "| $T_{\\text{full}}$ | " + " | ".join([f"第{p}期" for p in hz_overview_periods]) + " |"
print(header)
sep = "|:---:|" + ":---:|" * len(hz_overview_periods)
print(sep)
for Tf in hz_overview_T:
    cells = [f"{Tf}s"]
    for p in hz_overview_periods:
        a = alpha_hz(p)
        s = score_hz(a, Tf)
        if s >= 64999:
            cells.append("65000 ★")
        elif s >= 20000:
            cells.append(f"{s:.0f} ○")
        else:
            cells.append(f"**{s:.0f}**")
    print("| " + " | ".join(cells) + " |")
print("\n★ = 满分(>=65000), ○ = 拿满奖励(>=20000), **粗体** = 掉出奖励线(<20000)")

# --- Table 2: 式舆 Overview ---
print("\n### 表2: 式舆防卫战 -- 分数随节点演化")
print()
header = "| $T_{\\text{base}}$ | " + " | ".join([f"节点{n}" for n in sd_overview_nodes]) + " |"
print(header)
sep = "|:---:|" + ":---:|" * len(sd_overview_nodes)
print(sep)
for Tb in sd_overview_T:
    cells = [f"{Tb}s"]
    for n in sd_overview_nodes:
        a = alpha_sd(n)
        s = score_sd(a, Tb)
        if s >= 49999:
            cells.append("50000 ★")
        elif s >= 25000:
            cells.append(f"{s:.0f} ○")
        else:
            cells.append(f"**{s:.0f}**")
    print("| " + " | ".join(cells) + " |")
print("\n★ = 满分(>=50000), ○ = 拿满奖励(>=25000), **粗体** = 掉出奖励线(<25000)")

# --- Table 3: 危局 Type A ---
print("\n### 表3: 危局强袭战 Type A -- 满分掉分期预测")
print()
print(f"(当前: 第{HZ_CURRENT_PERIOD}期, alpha = {hz_alpha_curr*100:.1f}%)")
print()
print("| $T_{\\text{full}}$ | 当前通关时间 | 当前状态 | 满分掉分期 | 距当前剩余 |")
print("|:---:|:---:|:---:|:---:|:---:|")
for Tf in hz_typeA_T_values:
    t_curr = (1 + hz_alpha_curr) * Tf
    if t_curr <= HZ_TIME_LIMIT:
        status = "满分 ★"
    else:
        status = "已掉出 ✗"

    full_drop = None
    for p in range(1, 301):
        if (1 + alpha_hz(p)) * Tf > HZ_TIME_LIMIT:
            full_drop = p
            break

    drop_str = f"第{full_drop}期" if full_drop else "永不"
    remain = f"{full_drop - HZ_CURRENT_PERIOD}期" if full_drop else "—"
    print(f"| {Tf}s | {t_curr:.0f}s | {status} | {drop_str} | {remain} |")

# --- Table 4: 危局 Type B ---
print("\n### 表4: 危局强袭战 Type B -- 奖励线掉分期预测")
print()
print(f"(当前: 第{HZ_CURRENT_PERIOD}期, alpha = {hz_alpha_curr*100:.1f}%, 奖励线 = {HZ_REWARD}分)")
print()
print("| 当前分数 $S$ | 等效 $T_{\\text{full}}$ | 奖励线掉分期 | 距当前剩余 |")
print("|:---:|:---:|:---:|:---:|")
for r in hz_typeB_results:
    drop_str = f"第{r['reward_drop']}期" if r['reward_drop'] else "永不"
    remain = f"{r['remain']}期" if r['remain'] != '—' else "—"
    print(f"| {r['S_curr']:.0f} | {r['T_eff']:.0f}s | {drop_str} | {remain} |")

# --- Table 5: 式舆 Type A ---
print("\n### 表5: 式舆防卫战 Type A -- 满分掉分节点预测")
print()
print(f"(当前: 节点{SD_CURRENT_NODE}, alpha = {sd_alpha_curr*100:.1f}%)")
print()
print("| $T_{\\text{base}}$ | 当前通关时间 | 当前状态 | 满分掉分节点 | 距当前剩余 |")
print("|:---:|:---:|:---:|:---:|:---:|")
for Tb in sd_typeA_T_values:
    t_curr = (1 + sd_alpha_curr) * Tb
    if t_curr <= SD_FULL_TIME:
        status = "满分 ★"
    else:
        status = "已掉出 ✗"

    full_drop = None
    for n in range(1, 101):
        if (1 + alpha_sd(n)) * Tb > SD_FULL_TIME:
            full_drop = n
            break

    drop_str = f"节点{full_drop}" if full_drop else "永不"
    remain = f"{full_drop - SD_CURRENT_NODE}节点" if full_drop else "—"
    print(f"| {Tb}s | {t_curr:.0f}s | {status} | {drop_str} | {remain} |")

# --- Table 6: 式舆 Type B ---
print("\n### 表6: 式舆防卫战 Type B -- 奖励线掉分节点预测")
print()
print(f"(当前: 节点{SD_CURRENT_NODE}, alpha = {sd_alpha_curr*100:.1f}%, 奖励线 = {SD_REWARD}分)")
print()
print("| 当前分数 $S$ | 等效 $T_{\\text{base}}$ | 奖励线掉分节点 | 距当前剩余 |")
print("|:---:|:---:|:---:|:---:|")
for r in sd_typeB_results:
    drop_str = f"节点{r['reward_drop']}" if r['reward_drop'] else "永不"
    remain = f"{r['remain']}节点" if r['remain'] != '—' else "—"
    print(f"| {r['S_curr']:.0f} | {r['T_eff']:.0f}s | {drop_str} | {remain} |")

print()
print("=" * 80)
print("PREDICTION CHARTS (fig9-fig12)")
print("=" * 80)

# ============================================================
# Fig9: 危局 score evolution overview
# ============================================================
def plot_fig9_weiju_score_evolution():
    """危局分数演化预测总览 — 多条T_full曲线 vs 期数"""
    periods = np.arange(1, 61)
    T_full_values = [30, 45, 60, 75, 90, 120, 150, 180]
    colors = plt.cm.viridis(np.linspace(0, 1, len(T_full_values)))

    fig, ax = plt.subplots(figsize=(12, 7))
    for Tf, c in zip(T_full_values, colors):
        scores = [score_hz(alpha_hz(p), Tf) for p in periods]
        ax.plot(periods, scores, color=c, linewidth=1.8, label=f'T_full={Tf}s')

    ax.axhline(y=65000, color='green', linestyle='--', alpha=0.6, linewidth=1.2, label='满分 (65000)')
    ax.axhline(y=20000, color='red', linestyle='--', alpha=0.6, linewidth=1.2, label='奖励线 (20000)')
    ax.axvline(x=HZ_CURRENT_PERIOD, color='gray', linestyle=':', alpha=0.7, linewidth=1.5,
               label=f'当前第{HZ_CURRENT_PERIOD}期')

    ax.set_xlabel('期数', fontsize=12)
    ax.set_ylabel('预测分数', fontsize=12)
    ax.set_title('危局强袭战：不同强度玩家的分数预测轨迹', fontsize=14, fontweight='bold')
    ax.set_xlim(1, 60)
    ax.set_ylim(15000, 70000)
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fpath = os.path.join(PREDICTION_CHART_DIR, 'fig9_weiju_score_evolution.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fpath}")

# ============================================================
# Fig10: 式舆 score evolution overview
# ============================================================
def plot_fig10_shiyu_score_evolution():
    """式舆分数演化预测总览 — 多条T_base曲线 vs 节点"""
    nodes = np.arange(1, 36)
    T_base_values = [20, 30, 40, 50, 60, 70, 80, 100, 120]
    colors = plt.cm.plasma(np.linspace(0, 1, len(T_base_values)))

    fig, ax = plt.subplots(figsize=(12, 7))
    for Tb, c in zip(T_base_values, colors):
        scores = [score_sd(alpha_sd(n), Tb) for n in nodes]
        ax.plot(nodes, scores, color=c, linewidth=1.8, label=f'T_base={Tb}s')

    ax.axhline(y=50000, color='green', linestyle='--', alpha=0.6, linewidth=1.2, label='满分 (50000)')
    ax.axhline(y=25000, color='red', linestyle='--', alpha=0.6, linewidth=1.2, label='奖励线 (25000)')
    ax.axvline(x=SD_CURRENT_NODE, color='gray', linestyle=':', alpha=0.7, linewidth=1.5,
               label=f'当前节点{SD_CURRENT_NODE}')

    ax.set_xlabel('节点序号', fontsize=12)
    ax.set_ylabel('预测分数', fontsize=12)
    ax.set_title('式舆防卫战：不同强度玩家的分数预测轨迹', fontsize=14, fontweight='bold')
    ax.set_xlim(1, 35)
    ax.set_ylim(20000, 55000)
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fpath = os.path.join(PREDICTION_CHART_DIR, 'fig10_shiyu_score_evolution.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fpath}")

# ============================================================
# Fig11: 危局 Type B prediction roadmap
# ============================================================
def plot_fig11_weiju_typeB_roadmap():
    """危局 Type B 预测路线图 — 各分数档玩家从当前期数起的预测轨迹"""
    hz_typeB_scores = [60000, 55000, 50000, 45000, 40000, 35000, 30000, 25000, 22000]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(hz_typeB_scores)))

    fig, ax = plt.subplots(figsize=(12, 7))

    for S_curr, c in zip(hz_typeB_scores, colors):
        T_eff = effective_Tfull_from_score(S_curr, alpha_hz(HZ_CURRENT_PERIOD))
        if T_eff is None:
            continue

        max_p = HZ_CURRENT_PERIOD + 200
        periods = np.arange(HZ_CURRENT_PERIOD, max_p + 1)
        scores = [score_hz(alpha_hz(p), T_eff) for p in periods]

        # Find where score drops below reward line
        drop_p = None
        for p in periods:
            if score_hz(alpha_hz(p), T_eff) < HZ_REWARD:
                drop_p = p
                break

        # Plot only up to where it drops below reward line (or full range)
        end_idx = None
        for i, (p, s) in enumerate(zip(periods, scores)):
            if s < HZ_REWARD:
                end_idx = i + 5  # show a few more points after crossing
                break
        if end_idx is None or end_idx > len(periods):
            end_idx = len(periods)

        ax.plot(periods[:end_idx], scores[:end_idx], color=c, linewidth=1.8,
                label=f'S={S_curr}分')

        # Mark drop point
        if drop_p:
            s_drop = score_hz(alpha_hz(drop_p), T_eff)
            ax.scatter([drop_p], [s_drop], color=c, s=60, zorder=5, edgecolors='black', linewidth=0.5)
            remain = drop_p - HZ_CURRENT_PERIOD
            ax.annotate(f'{remain}期后', xy=(drop_p, s_drop),
                        xytext=(drop_p + 5, s_drop - 800),
                        fontsize=7, color=c,
                        arrowprops=dict(arrowstyle='->', color=c, alpha=0.7))

    ax.axhline(y=HZ_REWARD, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='奖励线 (20000分)')
    ax.axvline(x=HZ_CURRENT_PERIOD, color='gray', linestyle=':', alpha=0.5, linewidth=1,
               label=f'当前第{HZ_CURRENT_PERIOD}期')

    ax.set_xlabel('期数', fontsize=12)
    ax.set_ylabel('预测分数', fontsize=12)
    ax.set_title(f'危局强袭战 Type B 预测路线图：当前第{HZ_CURRENT_PERIOD}期各分数档玩家的未来分数轨迹', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fpath = os.path.join(PREDICTION_CHART_DIR, 'fig11_weiju_typeB_roadmap.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fpath}")

# ============================================================
# Fig12: 式舆 Type B prediction roadmap
# ============================================================
def plot_fig12_shiyu_typeB_roadmap():
    """式舆 Type B 预测路线图 — 各分数档玩家从当前节点起的预测轨迹"""
    sd_typeB_scores = [49000, 47000, 45000, 42000, 39000, 36000, 33000, 30000, 27000]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(sd_typeB_scores)))

    fig, ax = plt.subplots(figsize=(12, 7))

    for S_curr, c in zip(sd_typeB_scores, colors):
        T_eff = effective_Tbase_from_score(S_curr, alpha_sd(SD_CURRENT_NODE))
        if T_eff is None:
            continue

        max_n = SD_CURRENT_NODE + 100
        nodes = np.arange(SD_CURRENT_NODE, max_n + 1)
        scores = [score_sd(alpha_sd(n), T_eff) for n in nodes]

        # Find where score drops below reward line
        drop_n = None
        for n in nodes:
            if score_sd(alpha_sd(n), T_eff) < SD_REWARD:
                drop_n = n
                break

        # Plot range
        end_idx = None
        for i, (n, s) in enumerate(zip(nodes, scores)):
            if s < SD_REWARD:
                end_idx = i + 5
                break
        if end_idx is None or end_idx > len(nodes):
            end_idx = len(nodes)

        ax.plot(nodes[:end_idx], scores[:end_idx], color=c, linewidth=1.8,
                label=f'S={S_curr}分')

        # Mark drop point
        if drop_n:
            s_drop = score_sd(alpha_sd(drop_n), T_eff)
            ax.scatter([drop_n], [s_drop], color=c, s=60, zorder=5, edgecolors='black', linewidth=0.5)
            remain = drop_n - SD_CURRENT_NODE
            ax.annotate(f'{remain}节点后', xy=(drop_n, s_drop),
                        xytext=(drop_n + 3, s_drop - 400),
                        fontsize=7, color=c,
                        arrowprops=dict(arrowstyle='->', color=c, alpha=0.7))

    ax.axhline(y=SD_REWARD, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='奖励线 (25000分)')
    ax.axvline(x=SD_CURRENT_NODE, color='gray', linestyle=':', alpha=0.5, linewidth=1,
               label=f'当前节点{SD_CURRENT_NODE}')

    ax.set_xlabel('节点序号', fontsize=12)
    ax.set_ylabel('预测分数', fontsize=12)
    ax.set_title(f'式舆防卫战 Type B 预测路线图：当前节点{SD_CURRENT_NODE}各分数档玩家的未来分数轨迹', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fpath = os.path.join(PREDICTION_CHART_DIR, 'fig12_shiyu_typeB_roadmap.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fpath}")

# Generate all prediction charts
plot_fig9_weiju_score_evolution()
plot_fig10_shiyu_score_evolution()
plot_fig11_weiju_typeB_roadmap()
plot_fig12_shiyu_typeB_roadmap()

print()
print("=" * 80)
print("ALL DONE!")
print("=" * 80)
