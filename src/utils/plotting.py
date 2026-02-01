# ==============================================================================
# src/utils/plotting.py
# Role: Visual Narrative Engine (The "Nature-Style" Plotting Suite)
# Function: Standardizing aesthetics for Violin, Butterfly, and Pareto plots.
# Standard: High-DPI Publication Quality / Robust Font Fallback.
# ==============================================================================

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
import pandas as pd
import os
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class DWTSPlotter:
    """
    视觉叙事官：
    定义全局视觉风格，并提供针对 Problem C 核心任务的标准化绘图算子。
    """

    def __init__(self, output_dir: str = "reports/figures/"):
        self.logger = logging.getLogger("VISUAL_ENGINE")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # 核心配色 (Academic Muted Palette)
        self.colors = {
            'judge': '#ff7f0e',  # 橙色：代表专业评委 (Merit)
            'fan': '#1f77b4',  # 蓝色：代表观众偏好 (Populism)
            'highlight': '#d62728',  # 红色：代表异常点/最优解
            'grid': '#e0e0e0',
            'background': '#fdfdfd'
        }

        self._setup_global_style()

    def _setup_global_style(self):
        """配置符合顶刊标准的绘图风格"""
        # 1. 字体探测与自适应加载
        font_candidates = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif']
        system_fonts = {f.name for f in fm.fontManager.ttflist}
        selected_font = 'serif'
        for f in font_candidates:
            if f in system_fonts:
                selected_font = f
                break

        # 2. 全局参数注入 (RcParams)
        plt.rcParams.update({
            'font.family': selected_font,
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 300,
            'axes.facecolor': self.colors['background'],
            'grid.color': self.colors['grid'],
            'axes.axisbelow': True,
            'savefig.transparent': False,
            'axes.unicode_minus': False
        })
        sns.set_style("whitegrid")
        self.logger.info(f"绘图引擎初始化完成，采用字体: {selected_font}")

    def save_figure(self, filename: str):
        """标准化保存函数"""
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()
        self.logger.debug(f"图像已保存至: {path}")

    # --------------------------------------------------------------------------
    # Task 1: 投票分布可视化 (Violin Plot)
    # --------------------------------------------------------------------------
    def plot_posterior_vote_distribution(self, df_platinum: pd.DataFrame, season: int, week: int):
        """
        绘制后验投票分布图。
        展示模型反演出的投票点估计及不确定性区间。
        """
        week_data = df_platinum[(df_platinum['season'] == season) &
                                (df_platinum['week_num'] == week)].copy()

        if week_data.empty: return

        plt.figure(figsize=(10, 6))

        # 假设我们有采样数据或置信区间
        # 这里使用后验均值 (mu) 和标准差 (sigma) 模拟分布
        sns.barplot(data=week_data, x='celebrity_name', y='est_fan_vote_mu',
                    palette="Blues_d", alpha=0.7)

        # 叠加误差棒 (95% Credible Interval)
        # 如果没有 CI 列，则使用 sigma 估算
        if 'uq_low_95' in week_data.columns:
            yerr = [week_data['est_fan_vote_mu'] - week_data['uq_low_95'],
                    week_data['uq_high_95'] - week_data['est_fan_vote_mu']]
            plt.errorbar(x=np.arange(len(week_data)), y=week_data['est_fan_vote_mu'],
                         yerr=yerr, fmt='none', c='black', capsize=5, lw=1.5)

        plt.title(f"Inferred Fan Vote Distribution: Season {season} Week {week}", pad=15)
        plt.ylabel("Inferred Vote Share (Normalized)")
        plt.xticks(rotation=45)
        self.save_figure(f"S{season}W{week}_vote_distribution.png")

    # --------------------------------------------------------------------------
    # Task 3: 归因分析对比图 (Butterfly Plot)
    # --------------------------------------------------------------------------
    def plot_attribution_butterfly(self, features: list, judge_betas: list, fan_betas: list):
        """
        绘制“审美鸿沟”蝴蝶图。
        左侧：观众偏好系数；右侧：评委偏好系数。
        """
        df = pd.DataFrame({
            'Feature': features,
            'Judge': judge_betas,
            'Fan': fan_betas
        }).sort_values('Fan')

        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(df))

        # 左右双向条形图
        ax.barh(y_pos, df['Fan'], 0.4, label='Public Sentiment (Fans)', color=self.colors['fan'])
        ax.barh(y_pos, df['Judge'], -0.4, label='Expert Quality (Judges)', color=self.colors['judge'])

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['Feature'])
        ax.axvline(0, color='black', linewidth=0.8)

        plt.title("Evaluation Dissonance: Beta Coefficient Comparison", fontsize=15)
        plt.xlabel("Standardized Impact Factor (Beta)")
        plt.legend(loc='upper right')

        self.save_figure("causality_butterfly_contrast.png")

    # --------------------------------------------------------------------------
    # Task 4: 帕累托最优前沿 (Pareto Frontier)
    # --------------------------------------------------------------------------
    def plot_pareto_efficiency(self, df_optimizer_results: pd.DataFrame):
        """
        绘制 Equity vs. Efficiency 帕累托图。
        标注最优解点和历史基准点。
        """
        plt.figure(figsize=(10, 7))

        # 1. 绘制参数搜索散点云 (颜色深浅代表参数 t0)
        sc = plt.scatter(df_optimizer_results['efficiency'],
                         df_optimizer_results['equity'],
                         c=df_optimizer_results['t0'],
                         cmap='YlGnBu', alpha=0.6, s=40)

        plt.colorbar(sc, label='Transition Midpoint ($t_0$)')

        # 2. 绘制前沿包络线 (Pareto Frontier)
        # 寻找前沿点的简易算法
        df_sorted = df_optimizer_results.sort_values(by='efficiency')
        frontier = [df_sorted.iloc[0]]
        for _, row in df_sorted.iterrows():
            if row['equity'] > frontier[-1]['equity']:
                frontier.append(row)

        f_df = pd.DataFrame(frontier)
        plt.plot(f_df['efficiency'], f_df['equity'], color='black', linestyle='--', alpha=0.5)

        # 3. 标记乌托邦点
        plt.scatter(1.0, 1.0, color='gold', marker='*', s=200, label='Utopia Point')

        plt.title("Mechanism Optimization: The Equity-Efficiency Trade-off", fontsize=15)
        plt.xlabel("Engagement Metric (Fan Influence)")
        plt.ylabel("Fairness Metric (Spearman Correlation)")
        plt.legend()

        self.save_figure("mechanism_pareto_frontier.png")

    # --------------------------------------------------------------------------
    # 补充：SHAP 全局重要性 (Beeswarm)
    # --------------------------------------------------------------------------
    def plot_shap_summary_proxy(self, feature_names, importance_values):
        """
        SHAP 库本身绘图很丑，这里提供一个符合本系统风格的包装。
        """
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_values
        }).sort_values('Importance', ascending=True).tail(12)

        plt.figure(figsize=(10, 6))
        plt.barh(df['Feature'], df['Importance'], color='teal', alpha=0.8)
        plt.title("Latent Determinants of Fan Preference (SHAP Proxy)", loc='left')
        plt.xlabel("Mean |SHAP Value| (Impact on Vote Share)")

        self.save_figure("shap_global_summary.png")


# --- 单元测试 ---
if __name__ == "__main__":
    # 配置基础日志
    logging.basicConfig(level=logging.INFO)
    plotter = DWTSPlotter()

    # 测试蝴蝶图
    feats = ['Age', 'Athlete_Status', 'Partner_Alpha', 'Momentum']
    j_b = [-0.2, 0.5, 0.8, 0.3]
    f_b = [0.4, 0.6, 0.2, 0.9]
    plotter.plot_attribution_butterfly(feats, j_b, f_b)

    print("Visual test assets generated in reports/figures/")