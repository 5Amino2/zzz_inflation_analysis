#!/usr/bin/env python3
"""
危局强袭战 - 怪物血量膨胀分析
分析怪物HP随赛季（版本更新）的变化关系：
  1. 数据透视表（赛季×怪物）
  2. 折线图（线性插值补全缺失值）
  3. 线性拟合：拟合函数、R²，分析膨胀线性度和膨胀速度
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.linear_model import LinearRegression
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
INPUT_FILE = "危局强袭战怪物数据_69001-69041.xlsx"
OUTPUT_DIR = "../../charts/weiju"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 尝试设置中文字体
CHINESE_FONT_CANDIDATES = [
    'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC',
    'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'PingFang SC',
    'Hiragino Sans GB', 'STHeiti', 'sans-serif'
]

def setup_chinese_font():
    """检测并设置中文字体"""
    import matplotlib.font_manager as fm
    available = [f.name for f in fm.fontManager.ttflist]
    for font in CHINESE_FONT_CANDIDATES:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"[字体] 使用: {font}")
            return font
    # Fallback
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("[警告] 未找到中文字体，图表中文可能显示异常")
    return None

# ==================== 数据加载 ====================

def load_data():
    """加载Excel数据"""
    df = pd.read_excel(INPUT_FILE, engine='openpyxl')
    # 列名可能乱码，按位置重命名
    if len(df.columns) == 6:
        df.columns = ['season_id', 'phase', 'phase_name', 'monster_id', 'monster_name', 'hp']
    else:
        df.columns = ['season_id', 'phase', 'phase_name', 'monster_id', 'monster_name', 'hp_low', 'hp_high']
        # 使用高等级血量
        df['hp'] = df['hp_high']
        df = df.drop(columns=['hp_low', 'hp_high'])

    # 清理HP值：移除逗号，转换为整数
    df['hp'] = df['hp'].astype(str).str.replace(',', '').str.replace('，', '')
    df['hp'] = pd.to_numeric(df['hp'], errors='coerce')

    # 删除无效行
    df = df.dropna(subset=['hp', 'monster_name'])
    df['hp'] = df['hp'].astype(int)
    df['season_id'] = df['season_id'].astype(int)

    print(f"[数据] 加载 {len(df)} 条记录")
    print(f"[数据] 赛季范围: {df['season_id'].min()} - {df['season_id'].max()}")
    print(f"[数据] 怪物种类: {df['monster_name'].nunique()}")

    return df

# ==================== 数据透视表 ====================

def create_pivot_table(df):
    """创建赛季×怪物的透视表（同一赛季同一怪物取均值）"""
    # 聚合（同一赛季同一怪物可能出现在不同阶段，取平均HP）
    pivot = df.pivot_table(
        values='hp',
        index='monster_name',
        columns='season_id',
        aggfunc='mean'
    )
    # 按赛季ID排序
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    # 按怪物首次出现排序
    first_season = pivot.apply(lambda row: row.first_valid_index(), axis=1)
    pivot = pivot.loc[first_season.sort_values().index]

    print(f"[透视表] 形状: {pivot.shape[0]} 怪物 × {pivot.shape[1]} 赛季")

    return pivot

def save_pivot_table(pivot):
    """保存透视表到Excel"""
    # 格式化：数值加上千位分隔符
    formatted = pivot.copy()
    for col in formatted.columns:
        formatted[col] = formatted[col].apply(
            lambda x: f"{int(x):,}" if pd.notna(x) else ""
        )

    output_path = os.path.join(OUTPUT_DIR, "table1_monster_by_season.xlsx")
    formatted.to_excel(output_path, sheet_name="怪物血量×赛季")
    print(f"[表格] 已保存: {output_path}")

    # 也保存CSV
    csv_path = os.path.join(OUTPUT_DIR, "table1_monster_by_season.csv")
    formatted.to_csv(csv_path, encoding='utf-8-sig')
    print(f"[表格] 已保存: {csv_path}")

    return output_path

# ==================== 折线图 ====================

def plot_line_charts(pivot, chinese_font):
    """为每种怪物画折线图（含线性插值）+ 综合总览图"""

    # --- 单图：每种怪物一张 ---
    monster_names = pivot.index.tolist()
    n_monsters = len(monster_names)
    seasons_all = pivot.columns.tolist()

    print(f"\n[折线图] 绘制 {n_monsters} 种怪物的折线图...")

    for i, monster in enumerate(monster_names):
        series = pivot.loc[monster]
        valid = series.dropna()

        if len(valid) < 2:
            continue  # 少于2个数据点不画

        fig, ax = plt.subplots(figsize=(14, 6))

        # 所有赛季的x坐标
        x_all = np.array(seasons_all)
        x_valid = np.array(valid.index.tolist())
        y_valid = np.array(valid.values.tolist())

        # 原始数据点
        ax.scatter(x_valid, y_valid / 1e6, c='#2196F3', s=60, zorder=5,
                   label=f'原始数据 ({len(valid)} 个赛季)')

        # 线性插值：只在数据范围内的缺失点插值
        x_min, x_max = x_valid.min(), x_valid.max()
        interp_mask = (x_all >= x_min) & (x_all <= x_max)
        x_interp = x_all[interp_mask]

        # 对所有在范围内的赛季插值
        y_interp = np.interp(x_interp, x_valid, y_valid)

        # 画插值线
        ax.plot(x_interp, y_interp / 1e6, '-', c='#FF5722', linewidth=2,
                alpha=0.8, label='线性插值')

        # 标注插值点（非原始数据点）
        interp_only_x = [x for x in x_interp if x not in x_valid]
        interp_only_y = [y_interp[list(x_interp).index(x)] for x in interp_only_x]
        if interp_only_x:
            ax.scatter(interp_only_x, np.array(interp_only_y) / 1e6,
                      c='#FF5722', s=30, zorder=4, marker='s', alpha=0.6,
                      label=f'插值点 ({len(interp_only_x)} 个)')

        # 标注
        ax.set_xlabel('赛季ID', fontsize=12)
        ax.set_ylabel('生命值 (百万)', fontsize=12)
        ax.set_title(f'怪物: {monster}\nHP变化趋势 (赛季 {x_min} - {x_max})',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)

        # x轴刻度
        xticks = np.arange(seasons_all[0], seasons_all[-1] + 1, 2)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks], rotation=45, fontsize=8)

        plt.tight_layout()

        # 文件名：替换非法字符
        safe_name = monster.replace('/', '_').replace('\\', '_').replace(':', '_')
        fig.savefig(os.path.join(OUTPUT_DIR, f"line_{i:02d}_{safe_name}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"[折线图] 单怪物图已保存到 {OUTPUT_DIR}/")

    # --- 综合总览图：所有怪物在同一张图上 ---
    fig, ax = plt.subplots(figsize=(20, 12))

    colors = plt.cm.tab20(np.linspace(0, 1, n_monsters))

    for i, monster in enumerate(monster_names):
        series = pivot.loc[monster]
        valid = series.dropna()

        if len(valid) < 2:
            continue

        x_valid = np.array(valid.index.tolist())
        y_valid = np.array(valid.values.tolist())

        # 插值
        x_min, x_max = x_valid.min(), x_valid.max()
        x_interp = np.linspace(x_min, x_max, 100)
        y_interp = np.interp(x_interp, x_valid, y_valid)

        color = colors[i % len(colors)]
        ax.plot(x_interp, y_interp / 1e6, '-', c=color, linewidth=1.5, alpha=0.8)
        ax.scatter(x_valid, y_valid / 1e6, c=[color], s=20, alpha=0.6)

        # 在线的末端标注怪物名
        ax.annotate(monster, xy=(x_max, y_interp[-1] / 1e6),
                   xytext=(5, 0), textcoords='offset points',
                   fontsize=6, alpha=0.8, va='center')

    ax.set_xlabel('赛季ID', fontsize=14)
    ax.set_ylabel('生命值 (百万)', fontsize=14)
    ax.set_title(f'所有怪物HP变化总览 ({pivot.shape[1]} 个赛季)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    xticks = np.arange(seasons_all[0], seasons_all[-1] + 1, 4)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks], rotation=45, fontsize=9)
    ax.legend().set_visible(False)

    plt.tight_layout()
    overview_path = os.path.join(OUTPUT_DIR, "line_overview_all_monsters.png")
    fig.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[折线图] 总览图: {overview_path}")

# ==================== 线性拟合 ====================

def linear_fit_analysis(pivot):
    """对每种怪物做线性拟合，给出统计列表"""

    monster_names = pivot.index.tolist()
    seasons_all = pivot.columns.tolist()

    results = []

    print(f"\n[线性拟合] 分析 {len(monster_names)} 种怪物...")

    for monster in monster_names:
        series = pivot.loc[monster]
        valid = series.dropna()

        if len(valid) < 3:
            continue  # 少于3个数据点不做线性拟合

        x_valid = np.array(valid.index.tolist()).reshape(-1, 1)
        y_valid = np.array(valid.values.tolist())

        # 线性回归
        model = LinearRegression()
        model.fit(x_valid, y_valid)

        y_pred = model.predict(x_valid)
        slope = model.coef_[0]  # 每个赛季HP增长量
        intercept = model.intercept_

        # R²
        r2 = model.score(x_valid, y_valid)

        # Pearson r
        r, p_value = stats.pearsonr(x_valid.flatten(), y_valid)

        # 均方根误差
        rmse = np.sqrt(np.mean((y_valid - y_pred) ** 2))

        # 变异系数 (CV of residuals)
        cv_residual = np.std(y_valid - y_pred) / np.mean(y_valid) * 100

        # 增长率（每赛季百分比）
        mean_hp = np.mean(y_valid)
        growth_rate_per_season = (slope / mean_hp) * 100

        # 从第一个数据点到最后一个数据点的总增长
        first_hp = y_valid[0]
        last_hp = y_valid[-1]
        total_growth = (last_hp - first_hp) / first_hp * 100
        n_intervals = len(y_valid) - 1
        cagr = ((last_hp / first_hp) ** (1 / n_intervals) - 1) * 100 if n_intervals > 0 else 0

        results.append({
            'monster_name': monster,
            'n_seasons': len(valid),
            'first_season': valid.index.min(),
            'last_season': valid.index.max(),
            'first_hp': first_hp,
            'last_hp': last_hp,
            'mean_hp': mean_hp,
            'slope': slope,
            'intercept': intercept,
            'r2': r2,
            'pearson_r': r,
            'p_value': p_value,
            'rmse': rmse,
            'cv_residual_pct': cv_residual,
            'growth_rate_pct_per_season': growth_rate_per_season,
            'total_growth_pct': total_growth,
            'cagr_pct': cagr,
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('r2', ascending=False)

    # 保存统计结果
    output_path = os.path.join(OUTPUT_DIR, "table2_linear_fit_stats.xlsx")

    # 准备输出
    output_cols = {
        'monster_name': '怪物名称',
        'n_seasons': '出现赛季数',
        'first_season': '首次赛季',
        'last_season': '末次赛季',
        'first_hp': '首次HP',
        'last_hp': '末次HP',
        'mean_hp': '平均HP',
        'slope': '斜率(HP/赛季)',
        'intercept': '截距',
        'r2': 'R²',
        'pearson_r': 'Pearson r',
        'p_value': 'p值',
        'rmse': 'RMSE',
        'cv_residual_pct': '残差CV(%)',
        'growth_rate_pct_per_season': '每赛季增长率(%)',
        'total_growth_pct': '总增长率(%)',
        'cagr_pct': 'CAGR(%)',
    }

    df_out = df_results[list(output_cols.keys())].copy()
    df_out.columns = list(output_cols.values())

    # 格式化数值列
    df_out.to_excel(output_path, index=False, sheet_name='线性拟合统计')
    print(f"[线性拟合] 统计表: {output_path}")

    # CSV
    csv_path = os.path.join(OUTPUT_DIR, "table2_linear_fit_stats.csv")
    df_out.to_csv(csv_path, encoding='utf-8-sig', index=False)
    print(f"[线性拟合] CSV: {csv_path}")

    return df_results

# ==================== 膨胀分析图表 ====================

def plot_inflation_analysis(df_results, pivot):
    """画出膨胀分析汇总图"""

    df = df_results.copy()

    # --- 图1: R² 排名图 (线性度) ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # R² 排名
    df_sorted_r2 = df.sort_values('r2', ascending=True)
    colors_r2 = ['#4CAF50' if v >= 0.8 else '#FF9800' if v >= 0.6 else '#F44336'
                 for v in df_sorted_r2['r2']]

    ax = axes[0]
    bars = ax.barh(range(len(df_sorted_r2)), df_sorted_r2['r2'], color=colors_r2)
    ax.set_yticks(range(len(df_sorted_r2)))
    ax.set_yticklabels(df_sorted_r2['monster_name'], fontsize=8)
    ax.set_xlabel('R$^2$ (拟合优度)', fontsize=12)
    ax.set_title('怪物HP膨胀线性度排名 (R$^2$越接近1越线性)', fontsize=14, fontweight='bold')
    ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='R$^2$=0.8 (高线性)')
    ax.axvline(x=0.6, color='orange', linestyle='--', alpha=0.5, label='R$^2$=0.6 (中等线性)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')

    # 增长率排名
    df_sorted_growth = df.sort_values('growth_rate_pct_per_season', ascending=True)
    colors_growth = ['#F44336' if v >= 3 else '#FF9800' if v >= 2 else '#4CAF50'
                     for v in df_sorted_growth['growth_rate_pct_per_season']]

    ax = axes[1]
    bars = ax.barh(range(len(df_sorted_growth)), df_sorted_growth['growth_rate_pct_per_season'],
                   color=colors_growth)
    ax.set_yticks(range(len(df_sorted_growth)))
    ax.set_yticklabels(df_sorted_growth['monster_name'], fontsize=8)
    ax.set_xlabel('每赛季HP增长率 (%)', fontsize=12)
    ax.set_title('怪物HP膨胀速度排名 (每赛季增长率%)', fontsize=14, fontweight='bold')
    ax.axvline(x=3, color='red', linestyle='--', alpha=0.5, label='3%/赛季 (快速膨胀)')
    ax.axvline(x=2, color='orange', linestyle='--', alpha=0.5, label='2%/赛季 (中等膨胀)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "chart_inflation_rankings.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- 图2: R² vs 增长率 散点图 ---
    fig, ax = plt.subplots(figsize=(14, 8))

    scatter = ax.scatter(df['r2'], df['growth_rate_pct_per_season'],
                        s=df['n_seasons'] * 15,  # 点大小表示出现赛季数
                        c=df['total_growth_pct'],  # 颜色表示总增长
                        cmap='YlOrRd', alpha=0.8, edgecolors='black', linewidth=0.5)

    # 标注每个点
    for _, row in df.iterrows():
        ax.annotate(row['monster_name'],
                   xy=(row['r2'], row['growth_rate_pct_per_season']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=6, alpha=0.8)

    # 分区标注
    ax.axhline(y=df['growth_rate_pct_per_season'].median(), color='blue',
              linestyle='--', alpha=0.5, label=f'增长率中位数: {df["growth_rate_pct_per_season"].median():.2f}%')
    ax.axvline(x=df['r2'].median(), color='green',
              linestyle='--', alpha=0.5, label=f'R$^2$中位数: {df["r2"].median():.3f}')

    # 四象限标注
    mid_r2 = df['r2'].median()
    mid_growth = df['growth_rate_pct_per_season'].median()
    ax.text(0.98, 0.98, '高线性 + 快膨胀\n(最值得关注)', transform=ax.transAxes,
           fontsize=8, ha='right', va='top', alpha=0.6)
    ax.text(0.02, 0.98, '低线性 + 快膨胀\n(波动大但涨得快)', transform=ax.transAxes,
           fontsize=8, ha='left', va='top', alpha=0.6)
    ax.text(0.98, 0.02, '高线性 + 慢膨胀\n(稳定缓慢增长)', transform=ax.transAxes,
           fontsize=8, ha='right', va='bottom', alpha=0.6)
    ax.text(0.02, 0.02, '低线性 + 慢膨胀\n(波动大且涨得慢)', transform=ax.transAxes,
           fontsize=8, ha='left', va='bottom', alpha=0.6)

    ax.set_xlabel('R$^2$ (线性拟合优度)', fontsize=12)
    ax.set_ylabel('每赛季HP增长率 (%)', fontsize=12)
    ax.set_title('怪物HP膨胀：线性度 vs 膨胀速度\n(点大小=出现赛季数，颜色=总增长率%)',
                fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('总增长率 (%)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "chart_r2_vs_growth.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- 图2b: R² vs 增长率 散点图（排除低线性异常点） ---
    # 排除死路屠夫 (R²=0.1554) 和「霸主侵蚀体·庞培」 (R²=0.8034)
    EXCLUDED_MONSTERS = ['死路屠夫', '「霸主侵蚀体·庞培」']
    df_filtered = df[~df['monster_name'].isin(EXCLUDED_MONSTERS)].copy()

    if len(df_filtered) > 0:
        fig, ax = plt.subplots(figsize=(14, 8))

        scatter = ax.scatter(df_filtered['r2'], df_filtered['growth_rate_pct_per_season'],
                            s=df_filtered['n_seasons'] * 15,
                            c=df_filtered['total_growth_pct'],
                            cmap='YlOrRd', alpha=0.8, edgecolors='black', linewidth=0.5)

        for _, row in df_filtered.iterrows():
            ax.annotate(row['monster_name'],
                       xy=(row['r2'], row['growth_rate_pct_per_season']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=7, alpha=0.8)

        ax.axhline(y=df_filtered['growth_rate_pct_per_season'].median(), color='blue',
                  linestyle='--', alpha=0.5,
                  label=f'增长率中位数: {df_filtered["growth_rate_pct_per_season"].median():.2f}%')
        ax.axvline(x=df_filtered['r2'].median(), color='green',
                  linestyle='--', alpha=0.5,
                  label=f'R$^2$中位数: {df_filtered["r2"].median():.3f}')

        ax.text(0.98, 0.98, '高线性 + 快膨胀\n(最值得关注)', transform=ax.transAxes,
               fontsize=8, ha='right', va='top', alpha=0.6)
        ax.text(0.02, 0.98, '低线性 + 快膨胀\n(波动大但涨得快)', transform=ax.transAxes,
               fontsize=8, ha='left', va='top', alpha=0.6)
        ax.text(0.98, 0.02, '高线性 + 慢膨胀\n(稳定缓慢增长)', transform=ax.transAxes,
               fontsize=8, ha='right', va='bottom', alpha=0.6)
        ax.text(0.02, 0.02, '低线性 + 慢膨胀\n(波动大且涨得慢)', transform=ax.transAxes,
               fontsize=8, ha='left', va='bottom', alpha=0.6)

        ax.set_xlabel('R$^2$ (线性拟合优度)', fontsize=12)
        ax.set_ylabel('每赛季HP增长率 (%)', fontsize=12)
        ax.set_title('怪物HP膨胀：线性度 vs 膨胀速度 (排除低线性异常点)\n'
                     '已排除: 死路屠夫(R$^2$=0.16)、霸主侵蚀体·庞培(R$^2$=0.80)\n'
                     '(点大小=出现赛季数，颜色=总增长率%)',
                    fontsize=14, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('总增长率 (%)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "chart_r2_vs_growth_filtered.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[膨胀分析图表] 已生成排除异常点版本: chart_r2_vs_growth_filtered.png")
        print(f"  排除怪物: {', '.join(EXCLUDED_MONSTERS)}")
        print(f"  过滤后: {len(df_filtered)} 种怪物, R2均值={df_filtered['r2'].mean():.4f}, "
              f"R2中位数={df_filtered['r2'].median():.4f}")
        print(f"  斜率均值={df_filtered['slope'].mean():.0f} HP/期, "
              f"斜率中位数={df_filtered['slope'].median():.0f} HP/期")

    # --- 图3: 每种怪物的拟合线图 (分面) ---
    plot_facet_fits(df_results, pivot)

    print(f"[膨胀分析图表] 已保存到 {OUTPUT_DIR}/")

def plot_facet_fits(df_results, pivot):
    """为每种怪物画实际数据+拟合线的分面图"""
    n = len(df_results)
    if n == 0:
        return

    # 确定网格
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, (_, row) in enumerate(df_results.iterrows()):
        ax = axes[idx]
        monster = row['monster_name']
        series = pivot.loc[monster]
        valid = series.dropna()

        x_valid = np.array(valid.index.tolist())
        y_valid = np.array(valid.values.tolist())

        # 原始数据
        ax.scatter(x_valid, y_valid / 1e6, c='#2196F3', s=30, zorder=5)

        # 拟合线
        x_fit = np.linspace(x_valid.min(), x_valid.max(), 100)
        y_fit = row['slope'] * x_fit + row['intercept']
        ax.plot(x_fit, y_fit / 1e6, '-', c='#F44336', linewidth=2, alpha=0.7, label='线性拟合')

        # 标注统计量
        ax.text(0.02, 0.98,
               f"斜率: {row['slope']:.0f}/赛季\n"
               f"R$^2$: {row['r2']:.3f}\n"
               f"增长率: {row['growth_rate_pct_per_season']:.2f}%/赛季",
               transform=ax.transAxes, fontsize=7, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_title(monster, fontsize=9, fontweight='bold')
        ax.set_xlabel('赛季ID', fontsize=7)
        ax.set_ylabel('HP (百万)', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('怪物HP线性拟合分面图', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "chart_facet_linear_fits.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

# ==================== 分析报告 ====================

def generate_report(df_results):
    """生成文字分析报告"""
    df = df_results.copy()

    lines = []
    lines.append("=" * 70)
    lines.append("危局强袭战 - 怪物HP膨胀分析报告")
    lines.append("=" * 70)
    lines.append("")

    # 基本统计
    lines.append(f"总计分析怪物数: {len(df)} 种")
    lines.append(f"平均R²: {df['r2'].mean():.4f} (中位数: {df['r2'].median():.4f})")
    lines.append(f"平均每赛季增长率: {df['growth_rate_pct_per_season'].mean():.2f}%")
    lines.append(f"平均总增长率: {df['total_growth_pct'].mean():.1f}%")
    lines.append("")

    # --- 膨胀最线性的怪物 ---
    lines.append("-" * 70)
    lines.append("【1】膨胀最线性的怪物 (R² >= 0.8, 高线性)")
    lines.append("-" * 70)
    high_linear = df[df['r2'] >= 0.8].sort_values('r2', ascending=False)
    if len(high_linear) > 0:
        for _, r in high_linear.iterrows():
            lines.append(f"  ● {r['monster_name']}")
            lines.append(f"    R²={r['r2']:.4f}, 斜率={r['slope']:.0f} HP/赛季, "
                         f"每赛季增长={r['growth_rate_pct_per_season']:.2f}%, "
                         f"总增长={r['total_growth_pct']:.1f}%")
            lines.append(f"    出现于赛季 {r['first_season']}~{r['last_season']} "
                         f"(共{r['n_seasons']}次)")
    else:
        lines.append("  (无)")
    lines.append("")

    # --- 膨胀最快的怪物 ---
    lines.append("-" * 70)
    lines.append("【2】膨胀最快的怪物 (每赛季增长率 >= 3%)")
    lines.append("-" * 70)
    fast_growth = df[df['growth_rate_pct_per_season'] >= 3].sort_values('growth_rate_pct_per_season', ascending=False)
    if len(fast_growth) > 0:
        for _, r in fast_growth.iterrows():
            lines.append(f"  ● {r['monster_name']}")
            lines.append(f"    每赛季增长={r['growth_rate_pct_per_season']:.2f}%, "
                         f"总增长={r['total_growth_pct']:.1f}%, "
                         f"R²={r['r2']:.4f}, "
                         f"斜率={r['slope']:.0f} HP/赛季")
    else:
        lines.append("  (无)")
    lines.append("")

    # --- 又线性又快 ---
    lines.append("-" * 70)
    lines.append("【3】既线性又快速的膨胀 (R² >= 0.8 且 增长率 >= 2%)")
    lines.append("    这类怪物是最值得关注的——HP稳定且快速增长")
    lines.append("-" * 70)
    both = df[(df['r2'] >= 0.8) & (df['growth_rate_pct_per_season'] >= 2)].sort_values('growth_rate_pct_per_season', ascending=False)
    if len(both) > 0:
        for _, r in both.iterrows():
            lines.append(f"  ★ {r['monster_name']}")
            lines.append(f"    R²={r['r2']:.4f}, 每赛季增长={r['growth_rate_pct_per_season']:.2f}%, "
                         f"总增长={r['total_growth_pct']:.1f}%")
    else:
        lines.append("  (无)")
    lines.append("")

    # --- 低线性怪物 ---
    lines.append("-" * 70)
    lines.append("【4】膨胀线性度较差的怪物 (R² < 0.5)")
    lines.append("    这些怪物的HP变化不太符合线性规律，可能存在跳跃式调整")
    lines.append("-" * 70)
    low_linear = df[df['r2'] < 0.5].sort_values('r2', ascending=True)
    if len(low_linear) > 0:
        for _, r in low_linear.iterrows():
            lines.append(f"  ○ {r['monster_name']}")
            lines.append(f"    R²={r['r2']:.4f}, 每赛季增长={r['growth_rate_pct_per_season']:.2f}%, "
                         f"总增长={r['total_growth_pct']:.1f}%")
    else:
        lines.append("  (无)")
    lines.append("")

    # --- 综合排序表 ---
    lines.append("-" * 70)
    lines.append("【5】综合排序（按R²降序）")
    lines.append("-" * 70)
    lines.append(f"{'排名':<4} {'怪物名称':<20} {'R²':>8} {'斜率/赛季':>12} {'增长率%/季':>10} {'总增长%':>10} {'出现次数':>8}")
    lines.append("-" * 70)
    for i, (_, r) in enumerate(df.sort_values('r2', ascending=False).iterrows()):
        lines.append(f"{i+1:<4} {r['monster_name']:<20} {r['r2']:>8.4f} {r['slope']:>12.0f} "
                     f"{r['growth_rate_pct_per_season']:>10.2f} {r['total_growth_pct']:>10.1f} {r['n_seasons']:>8}")
    lines.append("")

    lines.append("=" * 70)
    lines.append("报告结束")
    lines.append("=" * 70)

    # 保存
    report_text = "\n".join(lines)
    report_path = os.path.join(OUTPUT_DIR, "report_analysis.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"[报告] 已保存: {report_path}")

    return report_text

# ==================== 主函数 ====================

def main():
    print("=" * 60)
    print("危局强袭战 - 怪物HP膨胀分析")
    print("=" * 60)

    # 0. 设置字体
    chinese_font = setup_chinese_font()

    # 1. 加载数据
    df = load_data()

    # 2. 创建透视表
    pivot = create_pivot_table(df)
    save_pivot_table(pivot)

    # 3. 画折线图
    plot_line_charts(pivot, chinese_font)

    # 4. 线性拟合分析
    df_results = linear_fit_analysis(pivot)

    # 5. 膨胀分析图表
    plot_inflation_analysis(df_results, pivot)

    # 6. 生成报告
    report = generate_report(df_results)
    # 尝试打印报告（Windows GBK终端可能报错，跳过）
    try:
        print(report)
    except UnicodeEncodeError:
        print("[提示] 报告已保存到文件中（终端编码不支持中文打印）")

    print(f"\n[完成] 所有结果已保存到: {os.path.abspath(OUTPUT_DIR)}/")

if __name__ == "__main__":
    main()
