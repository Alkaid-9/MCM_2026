# ==============================================================================
# src/analysis/sensitivity.py
# Role: Robustness & Sensitivity Audit Engine (Task 1 & Task 2)
# Function: Monte Carlo Stress Testing (Noise Injection & Prior Perturbation)
# Physics: Proving the "Low-Pass Filter" hypothesis and Inference Stability.
# Standard: Industrial Reliability / O-Prize "Stress Test" Section.
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata, spearmanr
import logging
import os
from tqdm import tqdm

# 引入项目统一绘图引擎
from src.utils.plotting import DWTSPlotter


class SensitivityAnalyzer:
    """
    灵敏度与鲁棒性审计师：
    1. Noise Stress Test: 向投票注入高斯噪声，测试赛制稳定性 (Task 2)。
    2. Prior Robustness: 扰动反演结果，测试核心排名的置信度区间 (Task 1)。
    """

    def __init__(self, df_platinum: pd.DataFrame, figures_dir: str = "reports/figures/"):
        self.logger = logging.getLogger("SENSITIVITY_AUDIT")
        # 仅分析有效数据
        self.df = df_platinum.dropna(subset=['est_fan_vote_mu', 'week_avg_score']).copy()
        self.fig_dir = figures_dir
        os.makedirs(self.fig_dir, exist_ok=True)
        self.plotter = DWTSPlotter(output_dir=figures_dir)

    def _add_simplex_noise(self, votes: np.ndarray, noise_level: float) -> np.ndarray:
        """
        在单纯形 (Simplex) 上注入噪声。
        物理逻辑：乘法对数正态噪声 -> 归一化 (保持 Sum=1)。
        """
        if noise_level <= 1e-9: return votes
        n = len(votes)
        # 噪声幅度与原始票数成正比 (Heteroscedasticity)
        noise = np.random.normal(0, noise_level, n)
        perturbed_votes = votes * np.exp(noise)
        return perturbed_votes / np.sum(perturbed_votes)

    def _calculate_winner(self, j_scores, f_votes, mechanism):
        """原子裁决函数"""
        if mechanism == "RANK":
            # 排名制：Rank和最小 (1=Best)
            total = rankdata(-j_scores, method='min') + rankdata(-f_votes, method='min')
            # 增加微小抖动处理平局 (优先看评委分)
            return np.argmin(total - j_scores * 1e-6)
        else:
            # 百分比制：分数和最大
            j_pct = j_scores / (np.sum(j_scores) + 1e-9)
            return np.argmax(j_pct + f_votes)

    def run_noise_stress_test(self, n_sims: int = 500, max_noise: float = 0.3):
        """
        [实验 A]: 机制鲁棒性对比 (Mechanism Robustness)。
        向投票注入噪声，观察 Rank 与 Percent 机制下冠军翻转的概率。
        """
        self.logger.info(f">>> 启动机制抗噪压力测试 (Sims={n_sims})...")

        # 筛选竞争激烈的周次 (Top 3 之后)
        target_weeks = self.df.groupby(['season', 'week_num']).filter(lambda x: len(x) >= 2)
        if target_weeks.empty: return None

        noise_levels = np.linspace(0.0, max_noise, 15)
        results = []

        # 选取有代表性的 20 个比赛周进行加速测试
        sample_groups = list(target_weeks.groupby(['season', 'week_num']))
        # 固定随机种子抽样
        np.random.seed(2026)
        selected_indices = np.random.choice(len(sample_groups), min(20, len(sample_groups)), replace=False)
        selected_groups = [sample_groups[i] for i in selected_indices]

        for sigma in tqdm(noise_levels, desc="Stress Testing"):
            flip_r, flip_p = 0, 0
            total = 0

            for _, group in selected_groups:
                j_scores = group['week_avg_score'].values
                f_base = group['est_fan_vote_mu'].values

                # 基准结果
                win_r_base = self._calculate_winner(j_scores, f_base, "RANK")
                win_p_base = self._calculate_winner(j_scores, f_base, "PERCENT")

                # 蒙特卡洛循环
                for _ in range(n_sims):
                    f_noisy = self._add_simplex_noise(f_base, sigma)

                    if self._calculate_winner(j_scores, f_noisy, "RANK") != win_r_base:
                        flip_r += 1
                    if self._calculate_winner(j_scores, f_noisy, "PERCENT") != win_p_base:
                        flip_p += 1
                    total += 1

            results.append({
                'noise_level': sigma,
                'flip_rate_rank': flip_r / total,
                'flip_rate_percent': flip_p / total
            })

        return pd.DataFrame(results)

    def run_prior_sensitivity_check(self):
        """
        [实验 B]: 先验灵敏度检查 (Prior Sensitivity)。
        物理意义：如果我们的反演结果仅仅依赖于先验（Zipf），那模型就是无效的。
        验证：计算后验结果与均匀分布（无信息先验）的相关性。如果相关性低，说明数据（Likelihood）起到了决定性作用。
        """
        self.logger.info(">>> 执行先验灵敏度与数据驱动度检查...")

        corrs = []
        for (s, w), group in self.df.groupby(['season', 'week_num']):
            if len(group) < 3: continue

            # 1. 均匀先验 (Uniform Prior)
            uniform_dist = np.ones(len(group)) / len(group)

            # 2. 后验结果
            posterior = group['est_fan_vote_mu'].values

            # 3. 计算“数据驱动度” (Data Drivenness)
            # 如果后验完全等于先验(均匀)，相关性=1，说明数据没用。
            # 如果相关性低，说明数据强力修正了先验。
            rho, _ = spearmanr(uniform_dist, posterior)

            # 这里的逻辑：我们希望后验 *不* 等于均匀分布
            # 使用 KL 散度可能更严谨，但 Spearman 更直观展示排名变化
            data_impact = 1 - abs(rho) if not np.isnan(rho) else 0  # 简化代理指标

            corrs.append(data_impact)

        avg_impact = np.mean(corrs)
        self.logger.info(f"数据驱动度 (Data Impact Score): {avg_impact:.4f} (越高越好)")
        return avg_impact

    def plot_stability_curve(self, df_res):
        """
        绘制鲁棒性曲线 (Stability Curve)。
        证明：Rank 机制是低通滤波器，Percent 机制是噪音放大器。
        """
        if df_res is None or df_res.empty: return

        plt.figure(figsize=(10, 6))

        # 绘制 Rank 曲线 (蓝色)
        plt.plot(df_res['noise_level'], df_res['flip_rate_rank'],
                 marker='o', linewidth=2.5, color=self.plotter.colors['fan'],
                 label='Rank System (Ordinal)')

        # 绘制 Percent 曲线 (橙色)
        plt.plot(df_res['noise_level'], df_res['flip_rate_percent'],
                 marker='s', linewidth=2.5, linestyle='--', color=self.plotter.colors['judge'],
                 label='Percent System (Cardinal)')

        # 填充差异区
        plt.fill_between(df_res['noise_level'],
                         df_res['flip_rate_rank'], df_res['flip_rate_percent'],
                         color='gray', alpha=0.1, label='Robustness Gap')

        plt.title("Mechanism Stability Audit: Noise Tolerance Analysis", fontsize=14, pad=15)
        plt.xlabel("Noise Intensity ($\sigma$)", fontsize=12)
        plt.ylabel("Winner Flip Probability (Instability)", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.5)

        # 标注关键阈值
        plt.axhline(0.1, color='red', linestyle=':', alpha=0.5)
        plt.text(0.01, 0.11, "Critical Instability Threshold (10%)", color='red', fontsize=9)

        self.plotter.save_figure("task2_stability_curve.png")