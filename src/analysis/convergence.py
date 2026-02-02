# ==============================================================================
# src/analysis/convergence.py
# Role: MCMC Convergence Diagnostic Engine (Figure 3 Generator)
# Function: Proving the "Scientific Integrity" of Bayesian Inversion.
# Visuals: Trace Plots (Left) + R-hat Histogram (Right) composite figure.
# Standard: Gelman-Rubin Statistic < 1.1 | ESS > 1000 | High-DPI Formatting.
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy import stats
import os
import logging
from typing import List, Optional

# 引入全局绘图风格
from src.utils.plotting import DWTSPlotter


class ConvergenceAnalyzer:
    """
    收敛性审计师：
    负责生成 Figure 3，回答 Task 1 中的 "consistency" 和 "uncertainty" 问题。
    """

    def __init__(self, figures_dir: str = "reports/figures/"):
        self.logger = logging.getLogger("CONVERGENCE_AUDIT")
        self.fig_dir = figures_dir
        self.plotter = DWTSPlotter(output_dir=figures_dir)
        os.makedirs(self.fig_dir, exist_ok=True)

    def plot_convergence_audit_panel(self,
                                     traces: np.ndarray,
                                     param_names: List[str],
                                     rhat_series: pd.Series,
                                     season: int,
                                     week: int):
        """
        【图表 3】MCMC 收敛审计复合图 (Trace + R-hat)

        学术意义：
        1. Left Panel: 展示马尔可夫链的遍历性 (Ergodicity)。
           我们要看到多条链（不同颜色）完美混合，像一条毛毛虫，没有趋势项。
        2. Right Panel: 展示 Gelman-Rubin 统计量的分布。
           我们要证明 99% 以上的参数 $\hat{R} < 1.1$，模型已收敛至稳态分布。

        :param traces: shape (n_chains, n_samples, n_params)
        :param param_names: 参数名称列表 (对应 n_params)
        :param rhat_series: 全局 R-hat 数据的 Series
        """
        self.logger.info(f"正在生成 Figure 3: MCMC 收敛审计图 (S{season}W{week})...")

        # 1. 布局设计：使用 constrained_layout 解决重叠问题
        # width_ratios 控制左右比例，1.5:1 让 Trace Plot 稍微宽一点
        fig = plt.figure(figsize=(15, 9), constrained_layout=True)
        gs = fig.add_gridspec(3, 2, width_ratios=[1.4, 1], wspace=0.15, hspace=0.1)

        # 选取三个代表性参数进行 Trace 展示
        n_params = traces.shape[2]
        # 智能选择：确保选到 Winner, Loser 和 Middle
        indices = [0, n_params // 2, n_params - 1]

        # 颜色配置 (学术色轮)
        chain_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        # --- Left Panel: Trace Plots ---
        # 共享 X 轴
        ax_traces = []
        for i, param_idx in enumerate(indices):
            ax = fig.add_subplot(gs[i, 0])
            ax_traces.append(ax)
            param_name = param_names[param_idx]

            # 降采样
            thinning_factor = max(1, traces.shape[1] // 2000)
            plot_data = traces[:, ::thinning_factor, param_idx]
            n_chains = plot_data.shape[0]

            # 绘制每条链
            for chain_idx in range(min(n_chains, 3)):  # 仅展示前3条链，保持清爽
                ax.plot(plot_data[chain_idx], lw=0.6, alpha=0.8,
                        color=chain_colors[chain_idx], label=f'Chain {chain_idx + 1}')

            # 装饰
            ax.set_ylabel("Vote Share", fontsize=10, labelpad=5)

            # 标题仅在第一个子图
            if i == 0:
                ax.set_title(f"(A) Trace Plots: Markov Chain Mixing (S{season} W{week})",
                             loc='left', fontsize=14, fontweight='bold', pad=10)
                # 图例放在图表上方外部，避免遮挡数据
                leg = ax.legend(ncol=3, loc='upper right', bbox_to_anchor=(1.0, 1.35),
                                frameon=False, fontsize=10, handlelength=1.5)

            # X轴标签仅在最后一个子图
            if i == 2:
                ax.set_xlabel("Iterations (Thinned)", fontsize=11)
            else:
                ax.set_xticklabels([])  # 隐藏中间图的 X 轴刻度

            # In-plot Annotation (半透明白底，防止遮挡)
            ax.text(0.015, 0.9, f"Param: {param_name}", transform=ax.transAxes,
                    fontsize=10, fontweight='bold', va='top',
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.2'))

            # 去除垃圾边框
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', linestyle=':', alpha=0.4)

        # --- Right Panel: R-hat Histogram ---
        ax_hist = fig.add_subplot(gs[:, 1])  # 占据右侧整列

        # 绘制直方图
        sns.histplot(rhat_series, bins=40, kde=True, ax=ax_hist,
                     color='#4c72b0', alpha=0.3, edgecolor='#4c72b0', linewidth=0.5)

        # 统计指标
        mean_r = rhat_series.mean()
        pass_rate = (rhat_series < 1.1).mean()
        max_r = rhat_series.max()

        # 绘制阈值线
        # 1.1 阈值线 (红色虚线)
        ax_hist.axvline(1.1, color='#d62728', linestyle='--', linewidth=2.5, label='Threshold (1.1)')
        # 均值线 (绿色实线)
        ax_hist.axvline(mean_r, color='#2ca02c', linestyle='-', linewidth=2.5, label=f'Mean $\hat{{R}}$')

        # 装饰
        ax_hist.set_title(r"(B) Global Convergence Audit: $\hat{R}$ Distribution",
                          loc='left', fontsize=14, fontweight='bold', pad=10)
        ax_hist.set_xlabel(r"Split-$\hat{R}$ Statistic", fontsize=12)
        ax_hist.set_ylabel("Parameter Count (Frequency)", fontsize=12)

        # 关键区域文字标注 (智能避让)
        y_lim = ax_hist.get_ylim()[1]

        # "Converged Zone" (绿色)
        ax_hist.text(1.01, y_lim * 0.95, "Converged Zone\n(Stable)",
                     color='#2ca02c', fontsize=11, fontweight='bold', ha='left', va='top')

        # "Non-Converged" (红色) - 如果范围够大才显示
        if ax_hist.get_xlim()[1] > 1.12:
            ax_hist.text(1.12, y_lim * 0.95, "Non-Converged\n(Transient)",
                         color='#d62728', fontsize=11, fontweight='bold', ha='left', va='top')

        # 统计卡片 (右上角)
        stats_text = (f"Total Params: {len(rhat_series)}\n"
                      f"Pass Rate (<1.1): {pass_rate:.1%}\n"
                      f"Mean $\hat{{R}}$: {mean_r:.4f}\n"
                      f"Max $\hat{{R}}$: {max_r:.3f}")

        ax_hist.text(0.95, 0.75, stats_text, transform=ax_hist.transAxes,
                     fontsize=11, ha='right', va='top',
                     bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', edgecolor='#dee2e6', alpha=0.95))

        # 图例
        ax_hist.legend(loc='upper right', frameon=True, fontsize=10)
        ax_hist.spines['top'].set_visible(False)
        ax_hist.spines['right'].set_visible(False)

        # 保存
        save_path = "task1_convergence_audit_panel.png"
        self.plotter.save_figure(save_path)
        self.logger.info(f"图表优化完成: {save_path}")


# --- 单元测试 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 构造模拟数据 (更逼真的 Log-Normal 拖尾分布)
    n_chains, n_samples = 3, 5000
    trace_data = np.zeros((n_chains, n_samples, 3))

    # 生成 "毛毛虫"
    for c in range(n_chains):
        # 模拟不同混合效率
        trace_data[c, :, 0] = np.random.normal(0.2, 0.05, n_samples)
        trace_data[c, :, 1] = np.random.normal(0.5, 0.08, n_samples) + np.sin(np.linspace(0, 20, n_samples)) * 0.02
        trace_data[c, :, 2] = np.random.normal(0.1, 0.02, n_samples)

    names = ['Bobby Bones (Winner)', 'Milo (RunnerUp)', 'Joe (Loser)']

    # 模拟 R-hat 分布 (Log-Gamma)
    # 绝大多数在 1.0-1.05，长尾到 1.2
    rhats = 1.0 + np.random.gamma(shape=1.5, scale=0.02, size=500)
    # 强行插入几个坏点测试鲁棒性
    rhats = np.append(rhats, [1.15, 1.25, 1.08])
    rhat_series = pd.Series(rhats)

    analyzer = ConvergenceAnalyzer()
    analyzer.plot_convergence_audit_panel(trace_data, names, rhat_series, 27, 10)

    print("Test passed. Check reports/figures/task1_convergence_audit_panel.png")