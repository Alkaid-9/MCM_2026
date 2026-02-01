"""
MCM 2026 Problem C: Mechanism Forensics & Design Visualization
Role: Visualizing the "Multiverse" (Counterfactuals) and the "Pareto Frontier" (Optimization).
Key Outputs: Kaplan-Meier Survival Curves, Pareto Trade-off Plots, DAW Weight Trajectories.
Standard: High-DPI, Publication-Ready, Color-Coded by Regime.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# 引入全局绘图风格配置
from src.utils.plotting import DWTSPlotter


class MechanismVisualizer:
    """
    机制可视化引擎：
    负责将‘平行宇宙’的模拟结果和‘机制设计’的优化结果转化为视觉证据。
    """

    def __init__(self, fig_dir: str = "reports/figures/"):
        self.logger = logging.getLogger("VIZ_MECHANISM")
        self.fig_dir = fig_dir
        self.plotter = DWTSPlotter(output_dir=fig_dir)
        os.makedirs(self.fig_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # Task 2: 生存分析与反事实对比
    # --------------------------------------------------------------------------
    def plot_survival_comparison(self, survival_df: pd.DataFrame):
        """
        【图表 1】技术流选手的生存曲线 (Kaplan-Meier)
        物理意义：直观展示 Rank 机制是否比 Percent 机制更能保护‘高技术、低流量’选手。
        输入 df 需包含: 'duration', 'observed_event', 'regime' (Rank/Percent)
        """
        self.logger.info("绘制 Kaplan-Meier 生存对比图...")

        plt.figure(figsize=(10, 7))
        kmf = KaplanMeierFitter()

        regimes = survival_df['regime'].unique()
        results = {}

        for regime in regimes:
            mask = survival_df['regime'] == regime
            # 设定颜色：Rank 用蓝色（理性），Percent 用橙色（激情/流量）
            color = self.plotter.colors['fan'] if 'Rank' in regime else self.plotter.colors['judge']
            linestyle = '-' if 'Rank' in regime else '--'

            kmf.fit(survival_df.loc[mask, 'duration'],
                    survival_df.loc[mask, 'observed_event'],
                    label=regime)

            kmf.plot_survival_function(ci_show=True, color=color, linestyle=linestyle, linewidth=2.5)
            results[regime] = {
                'durations': survival_df.loc[mask, 'duration'],
                'events': survival_df.loc[mask, 'observed_event']
            }

        # 计算 Log-Rank Test P-value
        if len(regimes) == 2:
            r1, r2 = regimes[0], regimes[1]
            lr_result = logrank_test(
                results[r1]['durations'], results[r2]['durations'],
                results[r1]['events'], results[r2]['events']
            )
            p_val = lr_result.p_value
            plt.text(0.05, 0.1, f"Log-Rank Test: p={p_val:.4e}",
                     transform=plt.gca().transAxes, fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        plt.title("Survival Analysis of Meritocratic Candidates (Top 30% Tech Score)", fontsize=14, pad=15)
        plt.xlabel("Weeks Survived", fontsize=12)
        plt.ylabel("Survival Probability $S(t)$", fontsize=12)
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle=':', alpha=0.4)
        plt.legend(loc="lower left")

        self.plotter.save_figure("task2_survival_km_curve.png")

    def plot_counterfactual_flip_rate(self, sensitivity_df: pd.DataFrame):
        """
        【图表 2】鲁棒性/翻转率曲线 (Stability Curve)
        物理意义：展示机制对噪音的敏感度。Rank 机制应表现为‘低通滤波器’（曲线平缓）。
        """
        self.logger.info("绘制机制稳定性曲线...")

        plt.figure(figsize=(10, 6))

        # 绘制 Rank 曲线
        plt.plot(sensitivity_df['noise_level'], sensitivity_df['flip_rate_rank'],
                 label='Rank System (Ordinal)', color=self.plotter.colors['fan'],
                 linewidth=3, marker='o', markersize=5)

        # 绘制 Percent 曲线
        plt.plot(sensitivity_df['noise_level'], sensitivity_df['flip_rate_percent'],
                 label='Percent System (Cardinal)', color=self.plotter.colors['judge'],
                 linewidth=3, marker='s', markersize=5, linestyle='--')

        # 填充差异区域 (Robustness Gap)
        plt.fill_between(sensitivity_df['noise_level'],
                         sensitivity_df['flip_rate_rank'],
                         sensitivity_df['flip_rate_percent'],
                         color='gray', alpha=0.1, label='Robustness Gap')

        plt.title("Robustness Audit: Winner Flip Rate under Noise Injection", fontsize=14, pad=15)
        plt.xlabel(r"Noise Intensity ($\sigma$ of Fan Vote Perturbation)", fontsize=12)
        plt.ylabel("Outcome Flip Probability", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)

        self.plotter.save_figure("task2_stability_curve.png")

    # --------------------------------------------------------------------------
    # Task 4: 机制设计与帕累托优化
    # --------------------------------------------------------------------------
    def plot_daw_weight_trajectory(self, k: float, t0: float, total_weeks: int = 10):
        """
        【图表 3】DAW 动态权重移交曲线 (Power Transfer Curve)
        物理意义：展示评委权力如何随赛程推进而接管比赛。
        """
        self.logger.info(f"绘制 DAW 权力曲线 (k={k}, t0={t0})...")

        weeks = np.arange(1, total_weeks + 1)
        progress = weeks / total_weeks

        # Sigmoid 逻辑复现 (需与 daw_engine 一致)
        # w(t) = 0.3 + 0.5 * Sigmoid(k*(t-t0)*10)
        def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

        weights = 0.3 + 0.5 * sigmoid(k * (progress - t0) * 10)

        plt.figure(figsize=(10, 5))
        plt.plot(weeks, weights, color='#d62728', linewidth=4, label='DAW Judge Weight')

        # 绘制辅助线
        plt.axhline(0.5, color='gray', linestyle='--', label='Balance Line (50/50)')
        plt.axvline(t0 * total_weeks, color='green', linestyle=':', label='Transition Point')

        # 区域标注
        plt.text(1.5, 0.35, "Populism Phase\n(Traffic Driven)", color='#1f77b4', fontsize=12, fontweight='bold')
        plt.text(total_weeks - 1.5, 0.75, "Meritocracy Phase\n(Skill Driven)", color='#ff7f0e', fontsize=12,
                 fontweight='bold', ha='right')

        plt.title(r"DAW Mechanism: Dynamic Authority Transfer ($k=" + f"{k}, t_0={t0}" + r"$)", fontsize=14)
        plt.xlabel("Competition Week", fontsize=12)
        plt.ylabel("Weight of Professional Judges ($w_J$)", fontsize=12)
        plt.ylim(0.2, 0.9)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

        self.plotter.save_figure("task4_daw_trajectory.png")

    def plot_pareto_frontier(self, optimizer_results: pd.DataFrame, baseline_metrics: dict = None):
        """
        【图表 4】帕累托前沿图 (The Money Plot)
        物理意义：证明 DAW 机制位于‘公平-效率’权衡的最优边界上。
        """
        self.logger.info("绘制帕累托优化前沿...")

        plt.figure(figsize=(12, 8))

        # 1. 绘制搜索空间 (散点)
        # 颜色映射 t0 (切换时机)，大小映射 k (激进程度)
        sc = plt.scatter(optimizer_results['efficiency'], optimizer_results['equity'],
                         c=optimizer_results['t0'], cmap='viridis',
                         s=optimizer_results['k'] * 10, alpha=0.6, edgecolors='w', linewidth=0.5)
        cbar = plt.colorbar(sc)
        cbar.set_label(r'Transition Timing ($t_0$)')

        # 2. 绘制前沿包络线 (Convex Hull Approximation)
        # 简单的包络算法：按 Efficiency 排序，计算 Equity 的累计最大值
        df_sorted = optimizer_results.sort_values('efficiency')
        frontier_equity = df_sorted['equity'].cummax()
        plt.plot(df_sorted['efficiency'], frontier_equity,
                 color='black', linestyle='--', alpha=0.6, linewidth=1.5, label='Pareto Frontier')

        # 3. 标记最优解 (Best DAW)
        # 假设距离 (1,1) 最近的点
        dist = np.sqrt((1 - df_sorted['equity']) ** 2 + (1 - df_sorted['efficiency']) ** 2)
        best_idx = dist.idxmin()
        best_pt = df_sorted.loc[best_idx]
        plt.scatter(best_pt['efficiency'], best_pt['equity'],
                    color='#d62728', s=300, marker='*', label='Optimal DAW', zorder=10)

        # 4. 标记历史基准
        if baseline_metrics:
            if 'RANK' in baseline_metrics:
                b = baseline_metrics['RANK']
                plt.scatter(b['efficiency'], b['equity'], color='#1f77b4', s=150, marker='s', label='Rank System',
                            zorder=10)
            if 'PERCENT' in baseline_metrics:
                b = baseline_metrics['PERCENT']
                plt.scatter(b['efficiency'], b['equity'], color='#ff7f0e', s=150, marker='^', label='Percent System',
                            zorder=10)

        # 5. 标记乌托邦点
        plt.scatter(1.0, 1.0, color='gold', s=200, marker='P', label='Utopia Point (Theoretical Max)', zorder=5)

        plt.title("Multi-Objective Optimization: Equity vs. Efficiency Trade-off", fontsize=16, pad=15)
        plt.xlabel("Engagement Metric (Fan Vote Impact)", fontsize=12)
        plt.ylabel("Fairness Metric (Merit Correlation)", fontsize=12)
        plt.legend(loc='lower left', frameon=True, framealpha=0.9)
        plt.grid(True, linestyle=':', alpha=0.4)

        self.plotter.save_figure("task4_pareto_frontier.png")


# --- 单元测试 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 模拟数据生成
    viz = MechanismVisualizer()

    # 1. Test Pareto
    df_opt = pd.DataFrame({
        'k': np.random.uniform(2, 20, 100),
        't0': np.random.uniform(0.3, 0.8, 100),
        'equity': np.random.uniform(0.6, 0.9, 100),
        'efficiency': np.random.uniform(0.5, 0.9, 100)
    })
    baselines = {
        'RANK': {'equity': 0.85, 'efficiency': 0.6},
        'PERCENT': {'equity': 0.65, 'efficiency': 0.9}
    }
    viz.plot_pareto_frontier(df_opt, baselines)

    # 2. Test DAW Trajectory
    viz.plot_daw_weight_trajectory(k=10.0, t0=0.6)

    # 3. Test Sensitivity
    df_sens = pd.DataFrame({
        'noise_level': np.linspace(0, 0.2, 10),
        'flip_rate_rank': np.linspace(0, 0.1, 10),
        'flip_rate_percent': np.linspace(0, 0.4, 10)  # Percent 翻转更快
    })
    viz.plot_counterfactual_flip_rate(df_sens)

    print("Test visualizations generated in reports/figures/")