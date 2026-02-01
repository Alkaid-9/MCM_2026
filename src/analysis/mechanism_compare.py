# ==============================================================================
# src/analysis/mechanism_compare.py
# Role: Mechanism Comparative Analytics Engine (v6.0 - Signal Processing Edition)
# Function: Quantifying Signal-to-Noise Ratio (SNR) and Mechanism Bias.
# Physics: Defining Rank as a "Low-Pass Filter" and Percent as a "Signal Amplifier".
# Standard: IEEE Signal Processing / Econometrics Rigor.
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import logging
import os

# 引入项目统一绘图引擎
from src.utils.plotting import DWTSPlotter

class MechanismComparator:
    """
    机制比对器：
    基于平行宇宙（Multiverse）的模拟数据，计算不同赛制的信噪比与偏差。
    """

    def __init__(self, df_sim_wide: pd.DataFrame, figures_dir: str = "reports/figures/"):
        """
        :param df_sim_wide: 透视后的宽表，必须包含: 
               ['week_avg_score', 'est_fan_vote_mu', 'sim_rank_placement', 'sim_pct_placement']
        """
        self.logger = logging.getLogger("MECH_COMPARE")
        self.df = df_sim_wide.copy()
        self.fig_dir = figures_dir
        self.plotter = DWTSPlotter(output_dir=figures_dir)
        os.makedirs(self.fig_dir, exist_ok=True)

    def _calculate_snr(self, result_ranks: pd.Series, judge_scores: pd.Series, fan_votes: pd.Series):
        """
        [核心算子] 计算单数据集的信噪比 (SNR).
        物理直觉：SNR 越高，结果越取决于技术分而非流量。
        """
        # 计算相关性 (Spearman Rho)
        # 注意：排名越小越好 (1st)，分数越高越好 -> 取负相关以衡量对齐度
        rho_signal, _ = spearmanr(result_ranks, -judge_scores)
        rho_noise, _ = spearmanr(result_ranks, -fan_votes)
        
        # 提取强度 (绝对值)
        signal_strength = abs(rho_signal)
        noise_strength = max(abs(rho_noise), 0.01) # 防御性除零保护
        
        return signal_strength / noise_strength, signal_strength, noise_strength

    def run_snr_analysis(self) -> pd.DataFrame:
        """
        [主程序] 全赛季 SNR 演化分析。
        量化 Rank 机制相对于 Percent 机制的“信噪比增益” (SNR Gain)。
        """
        self.logger.info(">>> 启动机制信噪比 (SNR) 审计...")
        records = []
        
        # 按赛季聚合分析
        groups = self.df.groupby('season')
        
        for season, group in groups:
            # 1. 计算 Rank 宇宙的 SNR
            snr_r, sig_r, noise_r = self._calculate_snr(
                group['sim_rank_placement'],
                group['week_avg_score'],
                group['est_fan_vote_mu']
            )
            
            # 2. 计算 Percent 宇宙的 SNR
            snr_p, sig_p, noise_p = self._calculate_snr(
                group['sim_pct_placement'],
                group['week_avg_score'],
                group['est_fan_vote_mu']
            )
            
            # 3. 计算增益 (Gain in dB)
            # 物理意义：Gain > 0 代表 Rank 制能更有效地过滤掉选票中的不合理噪音
            gain_db = 10 * np.log10(snr_r / (snr_p + 1e-9)) if snr_p > 0 else 0
            
            records.append({
                'season': season,
                'snr_rank': snr_r,
                'snr_percent': snr_p,
                'snr_gain_db': gain_db,
                'signal_rank': sig_r,
                'noise_rank': noise_r,
                'signal_percent': sig_p,
                'noise_percent': noise_p
            })
            
        res_df = pd.DataFrame(records)
        
        # 输出汇总结论
        avg_gain = res_df['snr_gain_db'].mean()
        self.logger.info(f"全赛季平均 SNR 增益: {avg_gain:.2f} dB")
        if avg_gain > 0:
            self.logger.info("结论：Rank 机制在全样本维度上表现出显著的噪声压制能力 (Filter effect)。")
            
        return res_df

    def plot_snr_evolution(self, snr_df: pd.DataFrame):
        """
        绘制 SNR 演化曲线。
        修复：使用 r"" 原始字符串防止 \r 被解析为回车符。
        """
        plt.figure(figsize=(12, 6))

        # 绘制主曲线
        plt.plot(snr_df['season'], snr_df['snr_rank'],
                 marker='o', color=self.plotter.colors['fan'], linewidth=2.5,
                 label='Rank System (Ordinal Filter)')

        plt.plot(snr_df['season'], snr_df['snr_percent'],
                 marker='x', color=self.plotter.colors['judge'], linewidth=2,
                 linestyle='--', label='Percent System (Cardinal Amplifier)')

        # 标注 S28 断点
        plt.axvline(28, color=self.plotter.colors['highlight'], linestyle=':', linewidth=1.5)

        # --- 关键修复点 1：加 r 前缀 ---
        plt.title(r"Mechanism Forensics: Signal-to-Noise Ratio (SNR) Analysis", fontsize=15, pad=20)
        plt.xlabel("Competition Season", fontsize=12)

        # --- 关键修复点 2：加 r 前缀，确保 \rho 被正确解析 ---
        plt.ylabel(r"SNR ($\rho_{Merit} / \rho_{Popularity}$)", fontsize=12)

        plt.legend(loc='upper left', frameon=True)
        plt.grid(True, linestyle=':', alpha=0.5)

        self.plotter.save_figure("task2_snr_evolution.png")

    def calculate_misalignment_rate(self) -> float:
        """
        计算“冠军错配率” (Champion Misalignment Rate)。
        量化在 Percent 规则下侥幸获胜但在 Rank 规则下会落选的概率。
        """
        # 找到 Percent 宇宙中的冠军 (sim_pct_placement == 1)
        winners_pct = self.df[self.df['sim_pct_placement'] == 1]
        
        if winners_pct.empty:
            return 0.0
            
        # 检查这些“百分比冠军”在 Rank 宇宙中的实际排位
        # 如果排名 > 3，视为严重错配（连前三都进不了）
        mismatched = winners_pct[winners_pct['sim_rank_placement'] > 3]
        rate = len(mismatched) / len(winners_pct)
        
        self.logger.info(f"审计发现：在 Percent 赛制冠军中，有 {rate:.1%} 的选手在更严格的 Rank 赛制下无法进入前三。")
        return rate

# --- 单元测试 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Mock 数据测试列名对齐
    mock_df = pd.DataFrame({
        'season': [1, 1, 1],
        'week_avg_score': [30, 25, 20],
        'est_fan_vote_mu': [0.1, 0.4, 0.5],
        'sim_rank_placement': [1, 2, 3],
        'sim_pct_placement': [3, 1, 2]
    })
    comparator = MechanismComparator(mock_df)
    snr_res = comparator.run_snr_analysis()
    print(snr_res)