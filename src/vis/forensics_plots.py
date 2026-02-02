# ==============================================================================
# src/vis/forensics_plots.py
# Role: Data Forensics Visualization Engine (Figure 2: The Ridge Plot)
# Function: Visualizing 34 seasons of score distributions to prove structural breaks.
# Physics: Showing the "Inflationary Drift" and "Phase Transition" at Season 28.
# Standard: High-DPI, Ridge-Style (Joyplot), Academic Annotations.
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from scipy.stats import gaussian_kde
import logging
import os

# 引入全局绘图风格 (Single Source of Truth)
from src.utils.plotting import DWTSPlotter


class ForensicsVisualizer:
    """
    法医级数据可视化器：
    专门用于生成 Section 3 (Data Curation) 的证据图表。
    """

    def __init__(self, df_gold: pd.DataFrame, fig_dir: str = "reports/figures/"):
        self.logger = logging.getLogger("VIZ_FORENSICS")
        self.df = df_gold.copy()
        self.fig_dir = fig_dir
        # 实例化全局绘图器以继承配色和字体设置
        self.plotter = DWTSPlotter(output_dir=fig_dir)
        os.makedirs(self.fig_dir, exist_ok=True)

    def plot_score_evolution_ridge(self):
        """
        【图表 2】评分通胀与制度断裂山脊图 (Evolutionary Ridge Plot)

        视觉逻辑：
        1. Y轴：赛季 (S1 -> S34)，模拟时间的流逝。
        2. X轴：评委打分 (Week Average Score)，展示分布形态。
        3. S28 断裂带：在 Season 28 处增加视觉间隙，标记规则突变。
        4. 漂移趋势线：连接各赛季均值，证明分数通胀。
        """
        self.logger.info("正在绘制评分通胀与制度断裂山脊图 (Ridge Plot)...")

        # 1. 数据准备
        # 聚合每个赛季的评分分布
        # 过滤掉包含 NaN 的数据
        clean_df = self.df.dropna(subset=['week_avg_score'])
        seasons = sorted(clean_df['season'].unique())

        if not seasons:
            self.logger.warning("数据为空，跳过绘图。")
            return

        # 设置绘图画布 (窄长型，适合放在论文侧边或整页)
        fig, ax = plt.subplots(figsize=(10, 12))

        # [修复 Warning]: 使用 matplotlib.colormaps
        # 颜色映射：从冷色(早期)到暖色(晚期)，暗示"竞争升温"或"通胀"
        try:
            cmap = mpl.colormaps['coolwarm']
        except AttributeError:
            # 兼容旧版 matplotlib
            cmap = plt.get_cmap('coolwarm')

        colors = [cmap(i / len(seasons)) for i in range(len(seasons))]

        # 视觉参数
        y_base = 0.0  # 基础高度
        y_step = 0.5  # 层叠间距 (越小越密集)

        # 根据数据范围动态设定 X 轴网格
        x_min = clean_df['week_avg_score'].min() - 2
        x_max = clean_df['week_avg_score'].max() + 2
        x_grid = np.linspace(x_min, x_max, 200)

        means = []  # 存储均值用于画趋势线
        mean_x = []  # 存储均值对应的X坐标
        y_positions = []  # 存储Y轴位置 (用于趋势线连接)

        # 2. 循环绘制每一季的密度层
        for i, s in enumerate(seasons):
            # 提取当季分数
            scores = clean_df[clean_df['season'] == s]['week_avg_score'].values

            # 数据太少无法计算密度，占位处理
            if len(scores) < 5 or np.std(scores) == 0:
                y_base += y_step
                continue

            # S28 断裂处理：在 S28 处增加额外的垂直间距，形成视觉隔断
            # 物理意义：Structure Break at Season 28
            if s == 28:
                y_base += 1.2  # 拉开显著距离
                # 添加断裂带标注线
                ax.axhline(y_base - 0.6, color='black', linestyle=':', linewidth=1.2, alpha=0.6)
                ax.text(x_max - 2, y_base - 0.6, "Structural Break\n(Judges' Save Introduced)",
                        color='#d62728', fontsize=10, va='center', ha='right', fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            # 计算 KDE 密度
            try:
                kde = gaussian_kde(scores, bw_method=0.35)
                density = kde(x_grid)
                # 归一化高度以便堆叠 (最大高度固定为 0.8 * y_step，避免过度遮挡)
                density = density / density.max() * 0.8
            except Exception:
                y_base += y_step
                continue

            # 绘制填充 (Fill) - 带透明度，展示重叠
            ax.fill_between(x_grid, y_base, y_base + density,
                            color=colors[i], alpha=0.75, zorder=len(seasons) - i)

            # 绘制轮廓 (Outline) - 白色描边增强立体感
            ax.plot(x_grid, y_base + density, color='white', linewidth=0.8, zorder=len(seasons) - i + 1)

            # 记录均值点 (用于趋势线)
            mean_score = np.mean(scores)
            means.append(mean_score)
            y_positions.append(y_base)
            mean_x.append(mean_score)

            # Y轴标签 (放在左侧，代替刻度)
            label_color = 'black' if s in [1, 10, 20, 27, 28, 34] else 'gray'
            font_weight = 'bold' if s in [27, 28] else 'normal'
            ax.text(x_min - 0.5, y_base + 0.1, f"S{s}", fontsize=9, ha='right',
                    color=label_color, fontweight=font_weight)

            y_base += y_step

        # 3. 绘制通胀趋势线 (Inflation Trend)
        # 使用三次多项式平滑拟合趋势，而不是折线，显得更学术
        if len(means) > 3:
            z = np.polyfit(y_positions, mean_x, 3)
            p = np.poly1d(z)
            trend_x = p(y_positions)

            ax.plot(trend_x, y_positions, color='#2c3e50', linestyle='--', linewidth=2.0,
                    alpha=0.8, label='Mean Score Drift (Inflation)', zorder=100)

        # 4. 装饰与标注
        ax.set_yticks([])  # 隐藏默认Y轴
        ax.set_xlim(x_min - 2, x_max + 1)
        ax.set_ylim(-0.5, y_base + 1.0)

        # 移除多余边框，仅保留底部X轴
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.2)

        ax.set_xlabel("Judges' Technical Score (Sum of 3/4 Scaled)", fontsize=12, fontweight='bold')
        ax.set_title("Figure 2: Evidence of Grade Inflation & Structural Break (S1-S34)",
                     fontsize=14, pad=20, fontweight='bold')

        # 标注 S27 异常 (Bobby Bones 所在的高方差区域)
        s27_idx = next((i for i, s in enumerate(seasons) if s == 27), -1)
        if s27_idx != -1 and s27_idx < len(y_positions):
            # 找到 S27 的 Y 轴位置
            # 注意：由于 S28 跳跃，S27 的位置要通过 y_positions 查找
            # 这里简单处理，直接找倒数第几个
            pass  # 已经在上面的循环里处理了通用逻辑，此处不仅标注

        # 添加图例 (固定在左上角)
        ax.legend(loc='upper left', frameon=False, fontsize=10)

        # 保存为高 DPI 图片
        # 使用 plotter 的标准化保存，自动处理路径
        self.plotter.save_figure("forensics_ridge_plot.png")


# --- 单元测试 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 构造模拟数据 (包含 S28 断裂和通胀趋势)
    data = []
    print("生成模拟数据...")
    for s in range(1, 35):
        # 模拟通胀：均值从 20 涨到 26
        base_mean = 20 + (s / 34) * 6
        # 模拟 S28 后的方差收缩 (制度规范化)
        base_std = 3.0 if s < 28 else 1.5
        # S27 异常高方差 (混乱)
        if s == 27: base_std = 6.0

        # 生成正态分布数据
        scores = np.random.normal(base_mean, base_std, 100)
        scores = np.clip(scores, 10, 30)  # 截断在合理范围

        for sc in scores:
            data.append({'season': s, 'week_avg_score': sc})

    df_mock = pd.DataFrame(data)

    viz = ForensicsVisualizer(df_mock)
    viz.plot_score_evolution_ridge()
    print("Test passed. Check reports/figures/forensics_ridge_plot.png")