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
from scipy.stats import gaussian_kde
import logging
import os

# 引入全局绘图风格
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
        seasons = sorted(self.df['season'].unique())
        if not seasons:
            self.logger.warning("数据为空，跳过绘图。")
            return

        # 设置绘图画布
        fig, ax = plt.subplots(figsize=(10, 12))

        # 颜色映射：从冷色(早期)到暖色(晚期)，暗示"竞争升温"或"通胀"
        colormap = cm.get_cmap('coolwarm')
        colors = [colormap(i / len(seasons)) for i in range(len(seasons))]

        # 视觉参数
        y_base = 0  # 基础高度
        y_step = 0.6  # 层叠间距
        x_grid = np.linspace(10, 30, 200)  # 假设满分30分制 (3个评委)

        means = []  # 存储均值用于画趋势线
        y_positions = []  # 存储Y轴位置

        # 2. 循环绘制每一季的密度层
        for i, s in enumerate(seasons):
            # 提取当季分数
            scores = self.df[self.df['season'] == s]['week_avg_score'].dropna().values
            if len(scores) < 10:
                means.append(np.nan)
                y_positions.append(y_base)
                y_base += y_step
                continue

            # 计算 KDE 密度
            kde = gaussian_kde(scores, bw_method=0.4)
            density = kde(x_grid)

            # 归一化高度以便堆叠
            density = density / density.max() * 0.9

            # S28 断裂处理：在 S28 处增加额外的垂直间距，形成视觉隔断
            if s == 28:
                y_base += 1.5  # 拉开距离
                # 添加断裂带标注
                ax.axhline(y_base - 0.8, color='black', linestyle=':', linewidth=1.0, alpha=0.5)
                ax.text(31, y_base - 0.8, "Structural Break\n(Judges' Save Introduced)",
                        color='#d62728', fontsize=10, va='center', fontweight='bold')

            # 绘制填充 (Fill)
            ax.fill_between(x_grid, y_base, y_base + density,
                            color=colors[i], alpha=0.7, zorder=len(seasons) - i)
            # 绘制轮廓 (Outline)
            ax.plot(x_grid, y_base + density, color='white', linewidth=0.5, zorder=len(seasons) - i + 1)

            # 记录均值点
            mean_score = np.mean(scores)
            means.append(mean_score)
            y_positions.append(y_base)

            # Y轴标签 (放在左侧)
            ax.text(9.5, y_base + 0.1, f"S{s}", fontsize=8, ha='right', color='gray')

            y_base += y_step

        # 3. 绘制通胀趋势线 (Inflation Trend)
        # 过滤 NaN 并绘制连接线
        valid_indices = [i for i, m in enumerate(means) if not np.isnan(m)]
        valid_means = [means[i] for i in valid_indices]
        valid_y = [y_positions[i] for i in valid_indices]

        ax.plot(valid_means, valid_y, color='black', linestyle='--', linewidth=1.5,
                alpha=0.6, label='Mean Score Drift (Inflation)', zorder=100)

        # 4. 装饰与标注
        ax.set_yticks([])  # 隐藏默认Y轴
        ax.set_xlim(10, 32)
        ax.set_ylim(-0.5, y_base + 1.5)

        # 移除边框，保留底部X轴
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_xlabel("Judges' Technical Score (Sum of 3)", fontsize=12)
        ax.set_title("Statistical Evidence: Grade Inflation & Structural Break (S1-S34)",
                     fontsize=14, pad=20, fontweight='bold')

        # 添加图例
        ax.legend(loc='upper left', frameon=False)

        # 关键区域高亮
        # S27 Bobby Bones 区域
        s27_idx = seasons.index(27) if 27 in seasons else -1
        if s27_idx != -1:
            ax.text(28, y_positions[s27_idx], "High Variance\n(S27 Anomaly)",
                    fontsize=8, color='#1f77b4', ha='center')

        # 保存
        save_path = os.path.join(self.fig_dir, "forensics_ridge_plot.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"山脊图已保存: {save_path}")


# --- 单元测试 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 构造模拟数据 (包含 S28 断裂和通胀趋势)
    data = []
    for s in range(1, 35):
        # 模拟通胀：均值从 20 涨到 26
        base_mean = 20 + (s / 34) * 6
        # 模拟 S28 后的方差收缩 (制度规范化)
        base_std = 3.0 if s < 28 else 2.0
        # S27 异常高方差
        if s == 27: base_std = 5.0

        scores = np.random.normal(base_mean, base_std, 100)
        scores = np.clip(scores, 0, 30)  # 截断

        for sc in scores:
            data.append({'season': s, 'week_avg_score': sc})

    df_mock = pd.DataFrame(data)

    viz = ForensicsVisualizer(df_mock)
    viz.plot_score_evolution_ridge()
    print("Test passed. Check reports/figures/forensics_ridge_plot.png")