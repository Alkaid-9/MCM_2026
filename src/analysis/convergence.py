# ==============================================================================
# src/analysis/convergence.py
# Role: MCMC Convergence Diagnostic Engine (v4.6 - The "Truth" Auditor)
# Function: Proving the "Scientific Integrity" of Bayesian Inversion
# Metrics: R-hat (Gelman-Rubin), ESS (Effective Sample Size), Trace Plots
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from typing import Optional, List

# 尝试导入 ArviZ (专业的贝叶斯绘图库)，如果环境没有则降级使用原生 Matplotlib
try:
    import arviz as az

    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False

# --- 学术绘图防御性配置 ---
try:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_context("paper", font_scale=1.4)
except:
    pass


class ConvergenceAnalyzer:
    """
    收敛性分析器：
    直接回答 Task 1：‘你对估计结果有多大把握？’
    证据链：R-hat < 1.1 表示模型已锁定全局最优解；ESS > 1000 表示样本有效性高。
    """

    def __init__(self, figures_dir: str = "reports/figures/"):
        self.fig_dir = figures_dir
        os.makedirs(self.fig_dir, exist_ok=True)
        self.logger = logging.getLogger("CONVERGENCE_AUDIT")

    def run_global_audit(self, df_platinum: pd.DataFrame) -> dict:
        """
        [宏观审计]：计算全赛季的收敛率。
        """
        self.logger.info(">>> 启动全局收敛性审计 (Global Convergence Audit)...")

        # 提取周级指标 (去重，因为同一周的 r_hat 是共享的)
        audit_df = df_platinum.drop_duplicates(subset=['season', 'week_num'])

        # 1. R-hat 审计 (稳定性)
        # 阈值 1.1 是 Gelman & Rubin (1992) 推荐的工业标准
        converged_mask = audit_df['r_hat'] < 1.1
        conv_rate = converged_mask.mean()

        # 2. ESS 审计 (有效性)
        avg_ess = audit_df['ess'].mean()

        # 3. 极端异常检测
        diverged_weeks = audit_df[audit_df['r_hat'] > 1.5]

        report = {
            'total_weeks': len(audit_df),
            'converged_weeks': converged_mask.sum(),
            'convergence_rate': conv_rate,
            'mean_r_hat': audit_df['r_hat'].mean(),
            'mean_ess': avg_ess,
            'diverged_count': len(diverged_weeks)
        }

        self._print_audit_report(report)
        self._plot_rhat_distribution(audit_df['r_hat'])

        return report

    def _print_audit_report(self, report: dict):
        """打印审计日志"""
        self.logger.info("-" * 50)
        self.logger.info(" MCMC CONVERGENCE AUDIT REPORT")
        self.logger.info("-" * 50)
        self.logger.info(f"Total Competition Weeks: {report['total_weeks']}")
        self.logger.info(f"Converged Weeks (R < 1.1): {report['converged_weeks']}")
        self.logger.info(f"Global Convergence Rate:   {report['convergence_rate']:.2%}")
        self.logger.info(f"Mean Gelman-Rubin (R-hat): {report['mean_r_hat']:.4f}")
        self.logger.info(f"Mean Effective Sample Size:{report['mean_ess']:.1f}")
        self.logger.info("-" * 50)

        if report['convergence_rate'] > 0.95:
            self.logger.info("STATUS: [PLATINUM] - 极高置信度")
        elif report['convergence_rate'] > 0.8:
            self.logger.info("STATUS: [GOLD] - 科学完整性通过")
        else:
            self.logger.warning("STATUS: [RED] - 模型可能存在欠拟合，请增加 MCMC 步数")

    def _plot_rhat_distribution(self, rhat_series: pd.Series):
        """
        绘制全样本 R-hat 分布直方图。
        学术意义：这是证明‘全局收敛’最有力的可视化证据。
        """
        plt.figure(figsize=(10, 6))

        # 绘制直方图与 KDE
        sns.histplot(rhat_series, bins=30, kde=True, color='#2c3e50', alpha=0.7, edgecolor='w')

        # 标注 1.1 临界线 (统计学公认的收敛阈值)
        plt.axvline(1.1, color='#e74c3c', linestyle='--', linewidth=2.5, label='Convergence Threshold (1.1)')

        # 标注 1.01 理想线
        plt.axvline(1.01, color='#27ae60', linestyle=':', linewidth=2, label='Ideal Target (1.01)')

        plt.title("Stability Audit: Distribution of Gelman-Rubin Statistics ($\hat{R}$)", fontsize=14, pad=15)
        plt.xlabel("$\hat{R}$ Value (Lower is Better, 1.0 is Perfect)", fontsize=12)
        plt.ylabel("Frequency (Competition Weeks)", fontsize=12)
        plt.legend()
        plt.grid(axis='y', alpha=0.2)

        # 嵌入统计摘要
        stats_text = (f"Mean: {rhat_series.mean():.3f}\n"
                      f"Max:  {rhat_series.max():.3f}\n"
                      f"Pass: {(rhat_series < 1.1).mean():.1%}")
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        save_path = os.path.join(self.fig_dir, "audit_rhat_global.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"R-hat 全局审计图已存至: {save_path}")

    def plot_trace_diagnostics(self, traces: np.ndarray, param_names: List[str], season: int, week: int):
        """
        [微观审计]：绘制单周的 Trace Plot (轨迹图) 和 Posterior Density (后验密度)。
        物理意义：展示多条并行链是否像“毛毛虫”一样交织在一起（即‘混合’良好）。

        :param traces: shape (n_chains, n_samples, n_params)
        """
        self.logger.info(f"正在生成 S{season}W{week} 的微观诊断图...")

        if HAS_ARVIZ:
            # Plan A: 使用 ArviZ 专业绘图 (O奖首选)
            self._plot_with_arviz(traces, param_names, season, week)
        else:
            # Plan B: 手写 Matplotlib 降级方案
            self._plot_manual(traces, param_names, season, week)

    def _plot_with_arviz(self, traces, param_names, season, week):
        """使用 ArviZ 绘制高大上的贝叶斯诊断图"""
        # 转换为 InferenceData 对象
        dataset = az.convert_to_inference_data(
            np.swapaxes(traces, 0, 1) if traces.shape[0] > traces.shape[1] else traces
        )

        # 绘制 Trace Plot
        axes = az.plot_trace(dataset, compact=True)
        plt.suptitle(f"MCMC Trace Audit: Season {season} Week {week}", fontsize=16)

        save_path = os.path.join(self.fig_dir, f"trace_S{season}W{week}_arviz.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 绘制 Forest Plot (展示 95% HDI)
        plt.figure()
        az.plot_forest(dataset, hdi_prob=0.95, combined=True)
        plt.title(f"Posterior Credible Intervals: S{season}W{week}")
        plt.savefig(os.path.join(self.fig_dir, f"forest_S{season}W{week}.png"), dpi=300)
        plt.close()

    def _plot_manual(self, traces, param_names, season, week):
        """
        手写 Trace Plot (当 ArviZ 不可用时)。
        绘制左侧：轨迹图；右侧：KDE 密度图。
        """
        n_chains, n_samples, n_params = traces.shape

        # 限制展示参数数量，防止图表过长
        n_plot = min(n_params, 10)
        fig, axes = plt.subplots(n_plot, 2, figsize=(12, 2 * n_plot), constrained_layout=True)

        if n_plot == 1: axes = np.array([axes])

        for i in range(n_plot):
            # 左图：Trace (毛毛虫)
            for c in range(min(n_chains, 5)):  # 最多画5条链，避免混乱
                axes[i, 0].plot(traces[c, :, i], alpha=0.6, linewidth=0.8)
            axes[i, 0].set_title(f"Trace: {param_names[i]}")
            axes[i, 0].set_ylabel("Vote Share")

            # 右图：Density (后验分布)
            sns.kdeplot(traces[:, :, i].flatten(), ax=axes[i, 1], fill=True, color='purple')
            axes[i, 1].set_title(f"Posterior PDF: {param_names[i]}")
            axes[i, 1].set_ylabel("Density")

        fig.suptitle(f"MCMC Diagnostics: Season {season} Week {week}", fontsize=16)

        save_path = os.path.join(self.fig_dir, f"trace_S{season}W{week}_manual.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        self.logger.info(f"Trace Plot 已保存: {save_path}")


# --- 单元测试 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 1. 模拟 Mock 数据 (Platinum 格式)
    mock_df = pd.DataFrame({
        'season': [1] * 10 + [2] * 10,
        'week_num': list(range(1, 11)) * 2,
        'r_hat': np.random.uniform(1.00, 1.05, 20),  # 良好的收敛
        'ess': np.random.uniform(800, 2000, 20)
    })
    # 插入一个故意未收敛的点
    mock_df.loc[15, 'r_hat'] = 1.8

    analyzer = ConvergenceAnalyzer()

    # 2. 测试全局审计
    analyzer.run_global_audit(mock_df)

    # 3. 测试微观 Trace 绘图
    # 模拟 (2 chains, 1000 samples, 3 contestants)
    mock_traces = np.random.beta(2, 5, size=(2, 1000, 3))
    names = ["Star_A", "Star_B", "Star_C"]
    analyzer.plot_trace_diagnostics(mock_traces, names, season=99, week=1)