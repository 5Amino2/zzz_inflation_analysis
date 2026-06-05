import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ==================== 全局设置 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
OUTPUT_DIR = "../charts/theory/"

# ==================== 危局强袭战数据 ====================
hp_multipliers = [1.2]*4 + [1.7]*4 + [2.2]*4 + [2.5]*4 + [3.0]*4 + [5.0]*3 + [5.0]*6
score_per_segment = [1000]*4 + [1200]*4 + [1800]*4 + [2400]*4 + [2600]*4 + [2600]*3 + [2700]*6

cumulative_hp = np.cumsum([0] + hp_multipliers)
cumulative_score = np.cumsum([0] + score_per_segment)
total_hp = cumulative_hp[-1]        # 87.4
total_damage_score = cumulative_score[-1]  # 60000
tech_score = 5000
TIME_LIMIT_WEIJU = 180

# 危局：给定输出量q，计算伤害分
def score_weiju_damage(q):
    if q >= total_hp:
        return total_damage_score
    if q <= 0:
        return 0.0
    for k in range(29):
        if cumulative_hp[k] <= q < cumulative_hp[k+1]:
            eff = score_per_segment[k] / hp_multipliers[k]
            return cumulative_score[k] + (q - cumulative_hp[k]) * eff
    return 0.0

# 危局：给定膨胀率alpha和原始打满时间T_full，计算总分
def score_weiju(alpha, T_full=180):
    Q = total_hp * TIME_LIMIT_WEIJU / T_full
    q = Q / (1 + alpha)
    return score_weiju_damage(q) + tech_score

# 危局：反解——给定目标伤害分，求所需输出量q
def inverse_score_weiju(target_damage_score):
    if target_damage_score >= total_damage_score:
        return total_hp
    if target_damage_score <= 0:
        return 0.0
    for k in range(29):
        if cumulative_score[k] <= target_damage_score <= cumulative_score[k+1]:
            eff = score_per_segment[k] / hp_multipliers[k]
            return cumulative_hp[k] + (target_damage_score - cumulative_score[k]) / eff
    return 0.0

# 危局：给定膨胀率alpha，求刚好能拿到奖励线(20000分)的玩家的原始分数S0
# 原理：膨胀后输出q = Q/(1+alpha)，要求score_weiju_damage(q) = 15000
# 反解得q，则原始输出Q0 = q*(1+alpha)，原始分数S0 = score_weiju_damage(Q0) + 5000
def reward_threshold_weiju(alpha):
    q = inverse_score_weiju(15000)  # 膨胀后刚好15000伤害分所需的输出量
    Q0 = q * (1 + alpha)            # 原始输出量
    S0 = score_weiju_damage(Q0) + tech_score
    return S0

# ==================== 式舆防卫战数据 ====================
time_boundaries = [0, 60, 70, 80, 90, 105, 120, 135, 150, 300]
time_rates = [5.0, 4.2, 3.5, 3.0, 2.5, 2.0, 1.6, 1.3, 1.0]

def M_integral(T1, T2):
    total = 0.0
    for i in range(len(time_boundaries)-1):
        start = time_boundaries[i]
        end = time_boundaries[i+1]
        rate = time_rates[i]
        intersect_start = max(T1, start)
        intersect_end = min(T2, end)
        if intersect_start < intersect_end:
            total += rate * (intersect_end - intersect_start)
    return total

def score_shiyu(alpha, T_base):
    T_elite = (1 + alpha) * T_base / 4
    T_total = (1 + alpha) * T_base
    T_boss = T_total - T_elite
    if T_boss > 0:
        avg_rate = M_integral(T_elite, T_total) / T_boss
    else:
        avg_rate = 5.0
    return 10000 + 8000 * avg_rate

# 式舆：给定膨胀率alpha，求刚好能拿到奖励线(25000分)的玩家的原始通关时间T_base
# 原理：score_shiyu(alpha, T_base) = 25000，反解T_base，再计算原始分数S0 = score_shiyu(0, T_base)
def reward_threshold_shiyu(alpha):
    # 定义目标函数：给定T_base，计算膨胀后的分数与25000的差
    def f(T_base):
        return score_shiyu(alpha, T_base) - 25000

    # 寻找根的范围：T_base太小则分数太高，T_base太大则分数太低（趋近18000）
    # 当alpha=0时，T_base约在100~300之间；随着alpha增大，临界T_base减小
    T_low, T_high = 1.0, 500.0
    # 确保区间内有根
    if f(T_low) < 0:  # 即使T_base=1也拿不到25000，说明alpha过大，无玩家能拿奖励线
        return None
    if f(T_high) > 0:  # 即使T_base=500也能拿25000，扩大搜索范围
        T_high = 1000.0

    try:
        T_base_crit = brentq(f, T_low, T_high)
        S0 = score_shiyu(0, T_base_crit)
        return S0
    except ValueError:
        return None

# ==================== 新增：S0基准奖励线临界函数 ====================
# 奖励线玩家不以T_full/T_base衡量自身强度，而是以无膨胀时的原始分数S0衡量。
# 以下函数计算：原始分数为S0的玩家，在膨胀率达到多少时会掉出奖励线。

def critical_alpha_weiju(S0):
    """
    给定危局玩家在alpha=0时的原始分数S0，求膨胀率alpha使其分数刚好降至20000分（奖励线）。
    解析解：利用分段线性映射可逆。原始输出Q0 = inverse_score(S0-5000)，
    奖励线所需输出q_reward = inverse_score(15000)，alpha = Q0/q_reward - 1。
    """
    damage_score_0 = S0 - tech_score
    if damage_score_0 <= 15000:
        return 0.0  # 已在奖励线或之下，无安全边际
    Q0 = inverse_score_weiju(damage_score_0)  # 原始输出量
    q_reward = inverse_score_weiju(15000)     # 奖励线所需输出量
    return Q0 / q_reward - 1.0


def critical_alpha_shiyu(S0):
    """
    给定式舆玩家在alpha=0时的原始分数S0，求膨胀率alpha使其分数刚好降至25000分（奖励线）。
    数值解：先由S0反解T_base，再求使score_shiyu(alpha, T_base)=25000的alpha。
    """
    if S0 <= 25000:
        return 0.0
    # 满分玩家（S0 >= 50000）：取T_base = 60s为规范值（满分边界）
    if S0 >= 50000:
        T_base = 60.0
    else:
        # 二分搜索：score_shiyu(0, T_base) = S0（T_base > 60时分数严格递减）
        lo, hi = 60.0, 2000.0
        for _ in range(60):
            mid = (lo + hi) / 2
            s = score_shiyu(0, mid)
            if s > S0:
                lo = mid
            else:
                hi = mid
        T_base = (lo + hi) / 2

    # 求alpha：score_shiyu(alpha, T_base) = 25000
    def f(alpha):
        return score_shiyu(alpha, T_base) - 25000

    if f(0) <= 0:
        return 0.0
    alpha_lo, alpha_hi = 0.0, 1.0
    while f(alpha_hi) > 0:
        alpha_hi *= 2
        if alpha_hi > 1000.0:
            return float('inf')
    try:
        alpha_crit = brentq(f, alpha_lo, alpha_hi)
        return alpha_crit
    except ValueError:
        return float('inf')

# ==================== 图1: 危局强袭战——刚好满分玩家分数随膨胀衰减 ====================
# 场景：玩家无膨胀时刚好180秒打满65000分（T_full=180s），随着怪物血量膨胀，
# 该玩家限时内能打出的输出减少，分数平滑下降。曲线为分段线性，斜率在血条效率切换处变化。
# 标注奖励线临界：当膨胀率达到355.7%时，分数刚好降至20000分（奖励线）。
def plot_fig1_weiju_score_decay():
    alphas = np.linspace(0, 4.0, 500)
    scores = [score_weiju(a, 180) for a in alphas]
    alpha_reward = 3.557  # 355.7%

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(alphas*100, scores, 'b-', linewidth=2.5, label='T_full=180s (刚好满分)')
    ax.axhline(y=20000, color='r', linestyle='--', alpha=0.7, label='奖励线 (20000分)')
    ax.axhline(y=65000, color='g', linestyle='--', alpha=0.7, label='满分 (65000分)')
    ax.axvline(x=alpha_reward*100, color='r', linestyle=':', alpha=0.5)
    ax.scatter([alpha_reward*100], [20000], color='red', s=100, zorder=5)
    ax.annotate(f'奖励线临界: α={alpha_reward*100:.1f}%\n分数=20000', 
                xy=(alpha_reward*100, 20000), 
                xytext=(alpha_reward*100+30, 28000),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    ax.set_xlabel('膨胀率 α (%)', fontsize=12)
    ax.set_ylabel('最终得分', fontsize=12)
    ax.set_title('危局强袭战：刚好满分玩家的分数随膨胀衰减', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 70000)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'fig1_weiju_score_decay.png', dpi=150, bbox_inches='tight')
    plt.show()

# ==================== 图2: 式舆防卫战——刚好满分玩家分数随膨胀衰减 ====================
# 场景：玩家无膨胀时刚好60秒通关（T_base=60s），首领战斗完全落在5.0倍率区间，满分50000分。
# 随着膨胀，通关时间延长，首领时段跨越倍率边界，加权平均倍率下降，分数平滑递减。
# 标注奖励线临界：当膨胀率达到270.3%时，分数刚好降至25000分（奖励线）。
def plot_fig2_shiyu_score_decay():
    alphas = np.linspace(0, 3.5, 500)
    scores = [score_shiyu(a, 60) for a in alphas]
    alpha_reward = 2.703  # 270.3%

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(alphas*100, scores, 'purple', linewidth=2.5, label='T_base=60s (刚好满分)')
    ax.axhline(y=25000, color='r', linestyle='--', alpha=0.7, label='奖励线 (25000分)')
    ax.axhline(y=50000, color='g', linestyle='--', alpha=0.7, label='满分 (50000分)')
    ax.axvline(x=alpha_reward*100, color='r', linestyle=':', alpha=0.5)
    ax.scatter([alpha_reward*100], [25000], color='red', s=100, zorder=5)
    ax.annotate(f'奖励线临界: α={alpha_reward*100:.1f}%\n分数=25000', 
                xy=(alpha_reward*100, 25000), 
                xytext=(alpha_reward*100+20, 32000),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    ax.set_xlabel('膨胀率 α (%)', fontsize=12)
    ax.set_ylabel('最终得分', fontsize=12)
    ax.set_title('式舆防卫战：刚好满分玩家的分数随膨胀衰减', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 350)
    ax.set_ylim(0, 55000)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'fig2_shiyu_score_decay.png', dpi=150, bbox_inches='tight')
    plt.show()

# ==================== 图3: 两模式归一化分数对比 ====================
# 场景：将危局分数除以65000、式舆分数除以50000，统一为"相对于满分的比例"，
# 在同一坐标系下对比两模式在膨胀下的掉分速率。危局在初期掉分更快（分段线性映射的陡峭段），
# 但奖励线安全边际更高（355.7% vs 270.3%）。两曲线在约α=135%处相交。
def plot_fig3_normalized_comparison():
    alphas = np.linspace(0, 5.0, 500)
    scores_weiju_norm = [score_weiju(a, 180)/65000 for a in alphas]
    scores_shiyu_norm = [score_shiyu(a, 60)/50000 for a in alphas]
    alpha_weiju_reward = 3.557
    alpha_shiyu_reward = 2.703

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(alphas*100, scores_weiju_norm, 'b-', linewidth=2.5, label='危局强袭战 (归一化: S/65000)')
    ax.plot(alphas*100, scores_shiyu_norm, 'purple', linewidth=2.5, label='式舆防卫战 (归一化: S/50000)')
    ax.axhline(y=20000/65000, color='b', linestyle='--', alpha=0.5)
    ax.axhline(y=25000/50000, color='purple', linestyle='--', alpha=0.5)
    ax.scatter([alpha_weiju_reward*100], [20000/65000], color='blue', s=100, zorder=5)
    ax.scatter([alpha_shiyu_reward*100], [25000/50000], color='purple', s=100, zorder=5)
    ax.annotate(f'危局奖励线\nα={alpha_weiju_reward*100:.1f}%', 
                xy=(alpha_weiju_reward*100, 20000/65000), 
                xytext=(alpha_weiju_reward*100+30, 0.55),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, color='blue')
    ax.annotate(f'式舆奖励线\nα={alpha_shiyu_reward*100:.1f}%', 
                xy=(alpha_shiyu_reward*100, 25000/50000), 
                xytext=(alpha_shiyu_reward*100+20, 0.75),
                arrowprops=dict(arrowstyle='->', color='purple'),
                fontsize=10, color='purple')
    ax.set_xlabel('膨胀率 α (%)', fontsize=12)
    ax.set_ylabel('归一化分数 (相对于满分)', fontsize=12)
    ax.set_title('两模式归一化分数对比：膨胀对满分玩家的影响', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'fig3_normalized_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# ==================== 图4: 危局强袭战——绰绰有余玩家打满分数时间随膨胀增加 ====================
# 场景：选取5档不同强度玩家（T_full=150,120,90,60,30s，对应提前30~150秒打完），
# 展示膨胀后打满65000分所需时间如何线性增加。每条曲线与180秒限时线的交点即为该玩家掉出满分的临界膨胀率。
# 核心规律：提前x%时间 → 可承受x%膨胀（如提前50%对应α_full=100%）。
def plot_fig4_weiju_time_increase():
    T_fulls = [150, 120, 90, 60, 30]
    colors = plt.cm.viridis(np.linspace(0, 1, len(T_fulls)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, T_full in enumerate(T_fulls):
        alpha_max = TIME_LIMIT_WEIJU / T_full - 1
        alphas = np.linspace(0, alpha_max * 1.2, 200)
        times = T_full * (1 + alphas)
        ax.plot(alphas*100, times, color=colors[i], linewidth=2, 
                label=f'T_full={T_full}s (提前{180-T_full}s)')
        ax.scatter([alpha_max*100], [TIME_LIMIT_WEIJU], color=colors[i], s=80, zorder=5)
        ax.annotate(f'α={alpha_max*100:.1f}%', 
                    xy=(alpha_max*100, TIME_LIMIT_WEIJU), 
                    xytext=(alpha_max*100+5, TIME_LIMIT_WEIJU+10),
                    fontsize=9, color=colors[i])

    ax.axhline(y=TIME_LIMIT_WEIJU, color='red', linestyle='--', alpha=0.7, label='限时180s (满分边界)')
    ax.fill_between([0, 550], [TIME_LIMIT_WEIJU, TIME_LIMIT_WEIJU], [250, 250], 
                    alpha=0.1, color='red', label='超时区域 (无法满分)')
    ax.set_xlabel('膨胀率 α (%)', fontsize=12)
    ax.set_ylabel('打满分数所需时间 (秒)', fontsize=12)
    ax.set_title('危局强袭战：绰绰有余玩家打满分数的时间随膨胀增加', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 550)
    ax.set_ylim(0, 250)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'fig4_weiju_time_increase.png', dpi=150, bbox_inches='tight')
    plt.show()

# ==================== 图5: 式舆防卫战——绰绰有余玩家通关时间随膨胀增加 ====================
# 场景：选取5档不同强度玩家（T_base=50,40,30,20,10s，对应提前10~50秒通关），
# 展示膨胀后通关时间如何线性增加。每条曲线与60秒满分边界线的交点即为该玩家掉出满分的临界膨胀率。
# 与危局完全对称：提前x% → 承受x%膨胀。
def plot_fig5_shiyu_time_increase():
    T_bases = [50, 40, 30, 20, 10]
    TIME_LIMIT_SHIYU = 60
    colors = plt.cm.plasma(np.linspace(0, 1, len(T_bases)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, T_base in enumerate(T_bases):
        alpha_max = TIME_LIMIT_SHIYU / T_base - 1
        alphas = np.linspace(0, alpha_max * 1.2, 200)
        times = T_base * (1 + alphas)
        ax.plot(alphas*100, times, color=colors[i], linewidth=2, 
                label=f'T_base={T_base}s (提前{60-T_base}s)')
        ax.scatter([alpha_max*100], [TIME_LIMIT_SHIYU], color=colors[i], s=80, zorder=5)
        ax.annotate(f'α={alpha_max*100:.1f}%', 
                    xy=(alpha_max*100, TIME_LIMIT_SHIYU), 
                    xytext=(alpha_max*100+5, TIME_LIMIT_SHIYU+3),
                    fontsize=9, color=colors[i])

    ax.axhline(y=TIME_LIMIT_SHIYU, color='red', linestyle='--', alpha=0.7, label='满分边界60s')
    ax.fill_between([0, 550], [TIME_LIMIT_SHIYU, TIME_LIMIT_SHIYU], [80, 80], 
                    alpha=0.1, color='red', label='超边界区域 (无法满分)')
    ax.set_xlabel('膨胀率 α (%)', fontsize=12)
    ax.set_ylabel('通关所需时间 (秒)', fontsize=12)
    ax.set_title('式舆防卫战：绰绰有余玩家通关时间随膨胀增加', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 550)
    ax.set_ylim(0, 80)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'fig5_shiyu_time_increase.png', dpi=150, bbox_inches='tight')
    plt.show()

# ==================== 图6: 危局强袭战——奖励线门槛曲线（S0基准） ====================
# 场景：奖励线玩家不以T_full衡量自身强度，而是以无膨胀时的原始分数S0来衡量。
# 横坐标为原始分数S0（alpha=0时的得分），纵坐标为能承受的临界膨胀率alpha。
# 曲线单调递增：S0越高，可承受的膨胀越大。
# S0=20000时alpha=0%（已在奖励线上，无缓冲）；S0=65000时alpha达到最大值355.7%。
# 该曲线直接回答：以你当前的无膨胀实力，能承受多少怪物血量膨胀还保住奖励线。
def plot_fig6_weiju_reward_threshold():
    S0_vals = np.linspace(20000, 65000, 500)
    alphas_crit = [critical_alpha_weiju(s) * 100 for s in S0_vals]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(S0_vals, alphas_crit, 'b-', linewidth=2.5, label='临界膨胀率 α')
    ax.axvline(x=20000, color='r', linestyle='--', alpha=0.7, label='奖励线 (20000分)')
    ax.axvline(x=65000, color='g', linestyle='--', alpha=0.7, label='满分 (65000分)')
    # 标注关键点
    for S0_mark in [25000, 30000, 40000, 50000, 60000, 65000]:
        alpha_mark = critical_alpha_weiju(S0_mark) * 100
        ax.scatter([S0_mark], [alpha_mark], color='blue', s=50, zorder=5)
        offset = 20 if S0_mark < 50000 else -80
        ax.annotate(f'S0={S0_mark/1000:.1f}万\nα={alpha_mark:.1f}%',
                    xy=(S0_mark, alpha_mark),
                    xytext=(S0_mark + offset, alpha_mark + 30),
                    fontsize=8, color='blue',
                    arrowprops=dict(arrowstyle='->', color='blue', alpha=0.6))
    ax.fill_between(S0_vals, 0, alphas_crit,
                    alpha=0.15, color='blue', label='安全区（膨胀率低于曲线即可保奖励）')
    ax.set_xlabel('原始分数 S0（无膨胀时得分）', fontsize=12)
    ax.set_ylabel('临界膨胀率 α (%)', fontsize=12)
    ax.set_title('危局强袭战：不同原始实力的玩家能承受多少膨胀才掉出奖励线', fontsize=14, fontweight='bold')
    ax.set_xlim(18000, 67000)
    ax.set_ylim(-20, 420)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'fig6_weiju_reward_threshold.png', dpi=150, bbox_inches='tight')
    plt.show()

# ==================== 图7: 式舆防卫战——奖励线门槛曲线（S0基准） ====================
# 场景：与危局图6对应，横坐标为原始分数S0，纵坐标为临界膨胀率alpha。
# S0=25000时alpha=0%（已在奖励线上）；S0=50000时alpha达到270.3%。
# 曲线形态反映积分模型的加权平均缓冲效应。
def plot_fig7_shiyu_reward_threshold():
    S0_vals = np.linspace(25000, 50000, 500)
    alphas_crit = []
    for s in S0_vals:
        a = critical_alpha_shiyu(s)
        alphas_crit.append(a * 100 if a != float('inf') else np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(S0_vals, alphas_crit, 'purple', linewidth=2.5, label='临界膨胀率 α')
    ax.axvline(x=25000, color='r', linestyle='--', alpha=0.7, label='奖励线 (25000分)')
    ax.axvline(x=50000, color='g', linestyle='--', alpha=0.7, label='满分 (50000分)')
    # 标注关键点
    for S0_mark in [28000, 33000, 38000, 42000, 47000, 50000]:
        alpha_mark = critical_alpha_shiyu(S0_mark) * 100
        ax.scatter([S0_mark], [alpha_mark], color='purple', s=50, zorder=5)
        offset = 30 if S0_mark < 40000 else -80
        ax.annotate(f'S0={S0_mark/1000:.2f}万\nα={alpha_mark:.1f}%',
                    xy=(S0_mark, alpha_mark),
                    xytext=(S0_mark + offset, alpha_mark + 15),
                    fontsize=8, color='purple',
                    arrowprops=dict(arrowstyle='->', color='purple', alpha=0.6))
    ax.fill_between(S0_vals, 0, alphas_crit,
                    alpha=0.15, color='purple', label='安全区（膨胀率低于曲线即可保奖励）')
    ax.set_xlabel('原始分数 S0（无膨胀时得分）', fontsize=12)
    ax.set_ylabel('临界膨胀率 α (%)', fontsize=12)
    ax.set_title('式舆防卫战：不同原始实力的玩家能承受多少膨胀才掉出奖励线', fontsize=14, fontweight='bold')
    ax.set_xlim(24000, 51000)
    ax.set_ylim(-10, 300)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'fig7_shiyu_reward_threshold.png', dpi=150, bbox_inches='tight')
    plt.show()

# ==================== 图8: 两模式归一化奖励线门槛对比 ====================
# 场景：将两模式的S0归一化到[0,1]区间：
# 危局：x = (S0-20000)/(65000-20000)，式舆：x = (S0-25000)/(50000-25000)。
# 横坐标为"从奖励线到满分的相对位置"，纵坐标为临界膨胀率。
# 直观展示：在相同的相对实力位置，哪个模式对玩家"保奖励"更宽容。
# 危局曲线在上方（超比例缓冲），式舆曲线在下方（线性缓冲）。
def plot_fig8_normalized_reward_threshold():
    # 危局：S0 from 20000 to 65000
    S0_weiju = np.linspace(20000, 65000, 400)
    x_weiju = (S0_weiju - 20000) / (65000 - 20000)  # 归一化到[0,1]
    y_weiju = [critical_alpha_weiju(s) * 100 for s in S0_weiju]

    # 式舆：S0 from 25000 to 50000
    S0_shiyu = np.linspace(25000, 50000, 400)
    x_shiyu = (S0_shiyu - 25000) / (50000 - 25000)  # 归一化到[0,1]
    y_shiyu = []
    for s in S0_shiyu:
        a = critical_alpha_shiyu(s)
        y_shiyu.append(a * 100 if a != float('inf') else np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_weiju, y_weiju, 'b-', linewidth=2.5, label='危局强袭战')
    ax.plot(x_shiyu, y_shiyu, 'purple', linewidth=2.5, label='式舆防卫战')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='奖励线 (临界α=0)')
    ax.fill_between(x_weiju, 0, y_weiju, alpha=0.08, color='blue')
    ax.fill_between(x_shiyu, 0, y_shiyu, alpha=0.08, color='purple')
    ax.set_xlabel('归一化相对实力 (0=刚好奖励线, 1=满分)', fontsize=12)
    ax.set_ylabel('临界膨胀率 α (%)', fontsize=12)
    ax.set_title('两模式归一化对比：相同相对实力下，哪个模式更能承受膨胀', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-10, 400)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'fig8_normalized_reward_threshold.png', dpi=150, bbox_inches='tight')
    plt.show()

# ==================== 图A: 危局强袭战——29条血条分数效率 η 对比 ====================
# 展示每条血条的分数效率 η = score/hp，按7组着色。
# 效率并非恒定——中间血条（17-14）效率最高（960.0），外圈血条（9-7）效率最低（520.0），差异接近2倍。
# 水平虚线标注恒定效率参考值（60000/87.4 ≈ 686.5）。
def plot_fig_weiju_hpbar_efficiency():
    efficiencies = [score_per_segment[i] / hp_multipliers[i] for i in range(29)]
    bar_indices = np.arange(1, 30)

    # 7组颜色
    group_colors = (['#4472C4']*4 + ['#ED7D31']*4 + ['#70AD47']*4 +
                    ['#FF0000']*4 + ['#9B59B6']*4 + ['#A52A2A']*3 + ['#E91E63']*6)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(bar_indices, efficiencies, color=group_colors, edgecolor='white', linewidth=0.5)

    # 恒定效率参考线
    const_eff = total_damage_score / total_hp  # ~686.5
    ax.axhline(y=const_eff, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'恒定效率参考 ({const_eff:.1f} 分/血量)')

    # 标注各组（组中点位置）
    group_centers = [2.5, 6.5, 10.5, 14.5, 18.5, 22.0, 26.5]
    group_labels = ['833.33 (外圈)', '705.88 (低谷)', '818.18', '960.00 (峰值)', '866.67', '520.00 (最低)', '540.00']
    group_etas = [833.33, 705.88, 818.18, 960.00, 866.67, 520.00, 540.00]
    for cx, label, eta in zip(group_centers, group_labels, group_etas):
        ax.annotate(label, xy=(cx, eta), xytext=(cx, eta + 35),
                    fontsize=8, ha='center', color='black',
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6))

    ax.set_xlabel('血条编号（29=最外圈, 1=最内圈）', fontsize=12)
    ax.set_ylabel('分数效率 η（分 / 血量单位）', fontsize=12)
    ax.set_title('危局强袭战：29条血条的分数效率 η 对比', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1100)
    ax.set_xticks([1, 4, 8, 12, 16, 20, 23, 29])
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'fig_weiju_hpbar_efficiency.png', dpi=150, bbox_inches='tight')
    plt.show()


# ==================== 图B: 危局强袭战——累计伤害-累计分数映射 ====================
# 展示分段线性映射：累计伤害（x轴）→ 累计伤害分（y轴）。
# 红色虚线为从(0,0)到(87.4,60000)的线性参考，凸显实际映射的非线性——曲线在中段"弓起"。
# 标注0伤害点和满伤害点。
def plot_fig_weiju_cumulative_damage_score():
    fig, ax = plt.subplots(figsize=(10, 7))

    # 分段线性实际曲线
    ax.plot(cumulative_hp, cumulative_score, 'b-', linewidth=2.5, label='实际分段线性映射')

    # 线性参考虚线（恒定效率）
    ax.plot([0, total_hp], [0, total_damage_score], 'r--', linewidth=1.8, alpha=0.8,
            label=f'线性参考（恒定效率 {total_damage_score/total_hp:.0f} 分/血量）')

    # 标注端点（0伤害和满伤害）
    ax.scatter([0], [0], color='blue', s=100, zorder=5)
    ax.scatter([total_hp], [total_damage_score], color='blue', s=100, zorder=5)
    ax.annotate(f'(0, 0)\n零伤害起点', xy=(0, 0), xytext=(5, 5000),
                fontsize=10, color='blue', arrowprops=dict(arrowstyle='->', color='blue', alpha=0.6))
    ax.annotate(f'({total_hp:.1f}, {total_damage_score:.0f})\n满分点',
                xy=(total_hp, total_damage_score), xytext=(total_hp-25, total_damage_score-8000),
                fontsize=10, color='blue', arrowprops=dict(arrowstyle='->', color='blue', alpha=0.6))

    # 标注组边界点
    group_boundaries = [4, 8, 12, 16, 20, 23]
    for b in group_boundaries:
        ax.scatter([cumulative_hp[b]], [cumulative_score[b]], color='green', s=40, zorder=4)

    const_eff_ref = total_damage_score / total_hp

    # 标注最大偏差区域
    # 在中段（bar 17-14, cumulative_hp 约35-45）填充偏差
    mid_start = cumulative_hp[12]  # bar 17开始 (0-indexed: bar 17 = index 12)
    mid_end = cumulative_hp[16]    # bar 14结束 (0-indexed: bar 14 = index 16)
    ax.fill_between([mid_start, mid_end],
                    [mid_start * const_eff_ref, mid_end * const_eff_ref],
                    [cumulative_score[12], cumulative_score[16]],
                    alpha=0.15, color='green', label='超线性增益区（峰值效率段）')

    ax.set_xlabel('累计伤害输出 q（血量单位）', fontsize=12)
    ax.set_ylabel('累计伤害分', fontsize=12)
    ax.set_title('危局强袭战：累计伤害-累计分数映射（非线性）', fontsize=14, fontweight='bold')
    ax.set_xlim(-2, total_hp + 5)
    ax.set_ylim(-2000, total_damage_score + 5000)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'fig_weiju_cumulative_damage_score.png', dpi=150, bbox_inches='tight')
    plt.show()


# ==================== 图C: 式舆防卫战——时间倍率阶梯函数 ====================
# 展示游戏时间（0-300s）与得分倍率的阶梯函数关系。
# 倍率从5.0×逐级降至1.0×，共9个平台。垂直线标注60秒满分边界。
def plot_fig_shiyu_time_multiplier():
    fig, ax = plt.subplots(figsize=(11, 6))

    # 阶梯图 (step plot with 'post')
    # time_boundaries有10个点，time_rates有9个值；step需要等长，在rates末尾补最后一个值
    rates_for_step = time_rates + [time_rates[-1]]
    ax.step(time_boundaries, rates_for_step, where='post', linewidth=2.5, color='#8E44AD',
            label='时间倍率')

    # 填充区域
    ax.fill_between(time_boundaries, 0, rates_for_step, step='post', alpha=0.12, color='#8E44AD')

    # 标注每个平台
    tier_labels = ['5.0×', '4.2×', '3.5×', '3.0×', '2.5×', '2.0×', '1.6×', '1.3×', '1.0×']
    for i in range(len(time_boundaries)-1):
        mid = (time_boundaries[i] + time_boundaries[i+1]) / 2
        rate = time_rates[i]
        ax.text(mid, rate + 0.12, tier_labels[i], ha='center', fontsize=9, color='#8E44AD', fontweight='bold')

    # 60秒满分边界
    ax.axvline(x=60, color='green', linestyle='--', linewidth=1.5, alpha=0.8, label='满分边界（60秒）')
    ax.annotate('满分边界\n5.0×区间', xy=(60, 5.0), xytext=(75, 5.3),
                fontsize=9, color='green', arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))

    # 时间区间标注
    interval_labels = ['0-60s', '61-70s', '71-80s', '81-90s', '91-105s', '106-120s', '121-135s', '136-150s', '151-300s']
    for i in range(len(time_boundaries)-1):
        mid = (time_boundaries[i] + time_boundaries[i+1]) / 2
        ax.text(mid, 0.15, interval_labels[i], ha='center', fontsize=7, color='gray')

    ax.set_xlabel('游戏时间（秒）', fontsize=12)
    ax.set_ylabel('得分倍率', fontsize=12)
    ax.set_title('式舆防卫战：时间倍率阶梯函数', fontsize=14, fontweight='bold')
    ax.set_xlim(-5, 310)
    ax.set_ylim(0, 5.8)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'fig_shiyu_time_multiplier.png', dpi=150, bbox_inches='tight')
    plt.show()


# ==================== 主程序：依次生成所有图表 ====================
if __name__ == "__main__":
    plot_fig_weiju_hpbar_efficiency()
    plot_fig_weiju_cumulative_damage_score()
    plot_fig_shiyu_time_multiplier()
    plot_fig1_weiju_score_decay()
    plot_fig2_shiyu_score_decay()
    plot_fig3_normalized_comparison()
    plot_fig4_weiju_time_increase()
    plot_fig5_shiyu_time_increase()
    plot_fig6_weiju_reward_threshold()
    plot_fig7_shiyu_reward_threshold()
    plot_fig8_normalized_reward_threshold()
    print("所有11幅图表已生成完毕！")
