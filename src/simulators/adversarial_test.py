# ==============================================================================
# src/simulators/adversarial_test.py
# Role: Adversarial Stress Test & System Resilience Engine (v5.8 - O-Prize)
# Function: Simulating Vote Brigading & Populist Shocks to audit Mechanism Safety.
# Physics: Proving the "Low-Pass Filter" vs. "Signal Amplifier" Hypotheses.
# Standard: MIT/Stanford Algorithmic Game Theory / JASA Stress Test Standards.
# ==============================================================================

import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import rankdata

# --- 核心组件导入 ---
from src.etl.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.utils.plotting import DWTSPlotter
from src.solvers.daw_engine import DAWEngine


class AdversarialAuditor:
    """
    对抗性审计师：
    通过注入“恶意脉冲噪声”，量化各机制对抗刷票攻击（Brigading）的鲁棒性。

    [学术逻辑]:
    1. 选定攻击目标：当周技术分最低的选手 (The Weakest Link)。
    2. 注入脉冲：向其票数施加 Multiplicative Shock (1x -> 10x)。
    3. 测量击穿点 (Breakdown Point)：攻击强度达到多少时，该选手能逆袭夺冠？
    """

    def __init__(self, df_platinum: pd.DataFrame, results_dir: str = "reports/mechanism_audit/"):
        self.logger = setup_logger("ADVERSARIAL_TEST")
        # 仅使用决赛周数据进行高压测试 (Season Finale High Stakes)
        self.df = df_platinum.copy()
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        self.plotter = DWTSPlotter(output_dir=results_dir)
        self.daw = DAWEngine()  # 用于获取动态权重

    def _inject_brigading_spike(self, votes: np.ndarray, target_idx: int, intensity: float) -> np.ndarray:
        """
        [恶意攻击算子]: 模拟“水军刷票”场景。
        物理意义：向特定选手的选票流中注入一个 Delta 函数脉冲，并维持单纯形约束。
        intensity: 1.0 = +100% 票数, 5.0 = +500% 票数
        """
        perturbed_votes = votes.copy()
        # 模拟攻击：目标选手票数暴涨
        perturbed_votes[target_idx] *= (1.0 + intensity)
        # 重新归一化至单纯形 (Simplex Projection)
        return perturbed_votes / (perturbed_votes.sum() + 1e-9)

    def _calculate_outcome(self, j_scores, f_votes, mode="PERCENT", w_j=0.5):
        """原子裁决：返回冠军索引"""
        if mode == "PERCENT":
            # 百分比制：分数和最大者赢
            j_pct = j_scores / (j_scores.sum() + 1e-9)
            total = j_pct + f_votes
            return np.argmax(total)

        elif mode == "RANK":
            # 排名制：排名和最小者赢
            j_rank = rankdata(-j_scores, method='min')
            f_rank = rankdata(-f_votes, method='min')
            total = j_rank + f_rank
            # 增加评委分微扰作为 Tie-Breaker
            return np.argmin(total - j_scores * 1e-6)

        elif mode == "DAW":
            # DAW制：动态加权
            j_rank = rankdata(-j_scores, method='min')
            f_rank = rankdata(-f_votes, method='min')
            total = w_j * j_rank + (1 - w_j) * f_rank
            return np.argmin(total - j_scores * 1e-6)

        return -1

    def run_brigading_simulation(self, target_season: int = 27, n_sims: int = 100):
        """
        [核心实验]: 遍历攻击强度，寻找各机制的崩溃阈值。
        """
        self.logger.info(f">>> 启动对抗性刷票压力测试 (Target: S{target_season})...")

        # 1. 选取该赛季最后一周 (决赛周)
        max_week = self.df[self.df['season'] == target_season]['week_num'].max()
        week_data = self.df[(self.df['season'] == target_season) & (self.df['week_num'] == max_week)]

        if len(week_data) < 2:
            self.logger.warning("数据不足，无法进行对抗测试。")
            return None

        # 2. 锁定攻击目标：技术分最低者 (The Underdog)
        j_scores = week_data['week_avg_score'].values
        f_base = week_data['est_fan_vote_mu'].values
        target_idx = np.argmin(j_scores)
        target_name = week_data.iloc[target_idx]['celebrity_name']

        self.logger.info(f" 攻击目标锁定: {target_name} (Ranked Last in Tech Score)")

        # 3. 遍历攻击强度 (0% -> 500% boost)
        intensities = np.linspace(0.0, 5.0, 20)
        results = []

        # DAW 在决赛周的权重 (模拟 Meritocracy Phase)
        w_daw_final = self.daw.compute_judge_weight(max_week, max_week, k=10.0, t0=0.6)

        for intensity in tqdm(intensities, desc="Injecting Spikes"):
            success_count = {'PERCENT': 0, 'RANK': 0, 'DAW': 0}

            for _ in range(n_sims):
                # 基础波动 (随机性) + 攻击脉冲
                # 先加一点随机波动
                f_noisy = f_base * np.random.lognormal(0, 0.1, len(f_base))
                f_attacked = self._inject_brigading_spike(f_noisy, target_idx, intensity)

                # 判定各机制下目标是否夺冠
                if self._calculate_outcome(j_scores, f_attacked, "PERCENT") == target_idx:
                    success_count['PERCENT'] += 1
                if self._calculate_outcome(j_scores, f_attacked, "RANK") == target_idx:
                    success_count['RANK'] += 1
                if self._calculate_outcome(j_scores, f_attacked, "DAW", w_daw_final) == target_idx:
                    success_count['DAW'] += 1

            results.append({
                'intensity': intensity,
                'ASR_Percent': success_count['PERCENT'] / n_sims,
                'ASR_Rank': success_count['RANK'] / n_sims,
                'ASR_DAW': success_count['DAW'] / n_sims
            })

        df_res = pd.DataFrame(results)
        self._plot_defense_frontier(df_res, target_name)
        return df_res

    def _plot_defense_frontier(self, df: pd.DataFrame, target_name: str):
        """
        绘制防御前沿图 (Defense Frontier Plot)。
        物理意义：ASR (Attack Success Rate) 越低，系统的安全性越高。
        """
        plt.figure(figsize=(10, 6))

        # 绘制三条曲线
        plt.plot(df['intensity'], df['ASR_Percent'], 'x--', color='gray', label='Percent System (Historical)',
                 alpha=0.6)
        plt.plot(df['intensity'], df['ASR_Rank'], 'o-', color='#1f77b4', label='Rank System', linewidth=2)
        plt.plot(df['intensity'], df['ASR_DAW'], '*-', color='#d62728', label='DAW System (Proposed)', linewidth=3)

        # 标注安全阈值 (Safety Margin)
        plt.axhline(0.5, color='black', linestyle=':', label='System Collapse Threshold (50%)')

        plt.title(f"Adversarial Resilience Audit: Targeting '{target_name}'", fontsize=14, pad=15)
        plt.xlabel("Brigading Intensity (Vote Surge Multiplier)", fontsize=12)
        plt.ylabel("Attack Success Rate (Probability of Winning)", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)

        # 标注
        plt.text(2.0, 0.1, "DAW Defense Zone", color='#d62728', fontsize=12, fontweight='bold')

        save_path = os.path.join(self.results_dir, "adversarial_defense_frontier.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"防御前沿图已生成: {save_path}")


# --- 单元测试 ---
if __name__ == "__main__":
    # Mock data for testing
    logging.basicConfig(level=logging.INFO)
    mock_data = pd.DataFrame({
        'season': [27] * 3, 'week_num': [10] * 3,
        'celebrity_name': ['Bobby', 'Milo', 'Evanna'],
        'week_avg_score': [24.0, 30.0, 30.0],  # Bobby 最低
        'est_fan_vote_mu': [0.5, 0.25, 0.25]
    })
    auditor = AdversarialAuditor(mock_data)
    auditor.run_brigading_simulation(target_season=27, n_sims=50)