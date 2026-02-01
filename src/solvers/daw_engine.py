# ==============================================================================
# src/solvers/daw_engine.py
# Role: Dynamic Adaptive Weighting (DAW) System Core
# Function: Generating time-dependent weights w(t) using Sigmoid transition.
# Physics: Smoothly shifting power from "Populism" (Fans) to "Meritocracy" (Judges).
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from scipy.stats import rankdata
from src.etl.config_loader import ConfigLoader


class DAWEngine:
    """
    DAW (Dynamic Adaptive Weighting) 引擎：
    实现基于 Sigmoid 函数的动态权重分配机制。

    [数学原理]
    Judge_Weight(t) = Base + Range * Sigmoid(k * (t - t0))

    其中：
    - t: 归一化赛程进度 [0, 1]
    - k: 切换斜率 (Slope)，代表“改革的激进程度”
    - t0: 切换中点 (Midpoint)，代表“权力移交的时间节点”
    """

    def __init__(self, fig_dir: str = "reports/figures/"):
        self.logger = logging.getLogger("DAW_ENGINE")
        self.cfg_loader = ConfigLoader()
        self.fig_dir = fig_dir
        os.makedirs(self.fig_dir, exist_ok=True)

        # 加载默认参数 (作为冷启动基准)
        # 实际运行时，这些参数通常由 ParetoOptimizer 动态注入
        daw_cfg = self.cfg_loader._config.get('task4_mechanism_design', {}).get('dynamic_weighting', {})
        self.default_k = daw_cfg.get('sigmoid_k', 10.0)
        self.default_t0 = daw_cfg.get('sigmoid_t0', 0.6)  # 赛程 60% 时达到权重平衡

        # 权重边界 (防止极端情况)
        self.min_judge_weight = 0.3  # 即使在娱乐为主的前期，评委也至少有 30% 话语权
        self.max_judge_weight = 0.8  # 即使在专业为主的决赛，观众也保留 20% 话语权

    def _sigmoid(self, x: float) -> float:
        """标准 Sigmoid 激活函数"""
        return 1.0 / (1.0 + np.exp(-x))

    def compute_judge_weight(self, progress: float, k: float = None, t0: float = None) -> float:
        """
        计算当前赛程进度下的评委权重 w_J(t)。

        :param progress: 当前赛程进度 (current_week / total_weeks)，范围 [0, 1]
        :param k: Sigmoid 斜率 (控制切换速度)
        :param t0: Sigmoid 中点 (控制切换时机)
        """
        if k is None: k = self.default_k
        if t0 is None: t0 = self.default_t0

        # 1. 计算 Sigmoid 响应
        # 放大输入范围以利用 Sigmoid 的非线性区
        x = k * (progress - t0)
        sig_val = self._sigmoid(x)

        # 2. 映射到 [Min, Max] 物理约束区间
        w_range = self.max_judge_weight - self.min_judge_weight
        w_judge = self.min_judge_weight + w_range * sig_val

        return w_judge

    def calculate_combined_score(self,
                                 judge_ranks: np.ndarray,
                                 fan_ranks: np.ndarray,
                                 progress: float,
                                 k: float = None,
                                 t0: float = None) -> np.ndarray:
        """
        计算 DAW 机制下的最终混合得分。

        [机制逻辑]
        采用“加权排名和” (Weighted Rank Sum)。
        Score = w_J(t) * Rank_J + (1 - w_J(t)) * Rank_F
        注意：排名越小越好 (1st, 2nd...)，因此 Score 越小越安全。
        """
        # 1. 获取动态权重
        w_j = self.compute_judge_weight(progress, k, t0)
        w_f = 1.0 - w_j

        # 2. 加权融合
        # 假设输入已经是排名 (1-based)
        combined_score = w_j * judge_ranks + w_f * fan_ranks

        return combined_score, w_j

    def simulate_week_outcome(self,
                              week_data: pd.DataFrame,
                              total_weeks: int,
                              k: float = None,
                              t0: float = None) -> pd.DataFrame:
        """
        模拟单周 DAW 裁决结果。
        """
        df = week_data.copy()
        current_week = df['week_num'].iloc[0]
        progress = current_week / total_weeks

        # 1. 准备排名信号
        # 技术分排名 (method='min', 分数高排名小)
        j_ranks = rankdata(-df['week_avg_score'], method='min')
        # 粉丝票排名 (method='min', 票数高排名小)
        # 注意：这里必须使用 Task 1 反演出的 'est_fan_vote_mu'
        f_ranks = rankdata(-df['est_fan_vote_mu'], method='min')

        # 2. 计算 DAW 混合分
        daw_scores, w_current = self.calculate_combined_score(j_ranks, f_ranks, progress, k, t0)

        df['daw_score'] = daw_scores
        df['daw_rank'] = rankdata(daw_scores, method='min')
        df['judge_weight_applied'] = w_current

        return df

    def plot_weight_trajectory(self, total_weeks: int = 10, k_list=None, t0_list=None):
        """
        [论文配图]：绘制权力移交曲线 (Power Transfer Curve)。
        展示不同参数配置下，评委权重随赛程的变化。
        """
        if k_list is None: k_list = [5, 10, 20]  # 平缓 -> 激进
        if t0_list is None: t0_list = [0.4, 0.6, 0.8]  # 早期 -> 晚期

        weeks = np.arange(1, total_weeks + 1)
        progress = weeks / total_weeks

        plt.figure(figsize=(10, 6))

        # 1. 绘制不同 k 值 (固定 t0)
        t0_fixed = self.default_t0
        colors = sns.color_palette("viridis", len(k_list))
        for i, k in enumerate(k_list):
            weights = [self.compute_judge_weight(p, k=k, t0=t0_fixed) for p in progress]
            plt.plot(weeks, weights, label=f'Aggressiveness $k={k}$',
                     color=colors[i], linewidth=2.5, linestyle='-')

        # 2. 绘制基准线
        plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Equal Weight (50/50)')
        plt.axhline(self.min_judge_weight, color='red', linestyle=':', alpha=0.3, label='Min Judge Floor')

        plt.title(f"DAW Mechanism: Dynamic Power Transfer Function\n(Transition Point $t_0={t0_fixed}$)", fontsize=14)
        plt.xlabel("Competition Week", fontsize=12)
        plt.ylabel("Weight of Judge's Score ($w_J$)", fontsize=12)
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.2)

        # 添加物理意义标注
        plt.text(1.5, self.min_judge_weight + 0.02, "Populism Phase\n(Traffic Driven)", color='darkred', fontsize=10)
        plt.text(total_weeks - 2, self.max_judge_weight - 0.05, "Meritocracy Phase\n(Skill Driven)", color='darkgreen',
                 fontsize=10)

        save_path = os.path.join(self.fig_dir, "daw_weight_trajectory.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"DAW 权力曲线图已生成: {save_path}")


# --- 单元测试 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = DAWEngine()

    # 1. 生成轨迹图
    engine.plot_weight_trajectory(total_weeks=12)

    # 2. 模拟单周数据
    print("\n--- DAW Simulation Test (Week 8/10) ---")
    mock_df = pd.DataFrame({
        'season': 99, 'week_num': 8,
        'celebrity_name': ['Tech_Pro', 'Fan_Fav', 'Mediocre'],
        'week_avg_score': [29.0, 20.0, 24.0],  # Tech > Mediocre > Fan
        'est_fan_vote_mu': [0.1, 0.6, 0.3]  # Fan > Mediocre > Tech
    })

    # 在 Week 8 (后期)，评委权重应该很高，技术好的 Tech_Pro 应该反超
    res = engine.simulate_week_outcome(mock_df, total_weeks=10)
    print(res[['celebrity_name', 'week_avg_score', 'est_fan_vote_mu', 'daw_score', 'daw_rank']])

    print(f"\nApplied Judge Weight: {res['judge_weight_applied'].iloc[0]:.4f}")