# ==============================================================================
# src/solvers/objective_engine.py
# Role: Mechanism Scorecard (The Evaluation Metric Engine)
# Function: Quantifying Fairness and Engagement for Task 4
# Method: Spearman Correlation & Marginal Variance Analysis
# ==============================================================================

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import logging


class MechanismEvaluator:
    """
    机制评估引擎：
    计算任一赛制宇宙下的‘公平性’与‘参与度’，支撑帕累托寻优。
    """

    def __init__(self):
        self.logger = logging.getLogger("OBJECTIVE_ENGINE")

    @staticmethod
    def calculate_fairness_metric(final_rankings: pd.Series, tech_scores: pd.Series) -> float:
        """
        指标 1: 技术公平性 (Equity Index)。
        物理意义：最终排名与累计技术分的相关性。
        算法：Spearman 秩相关系数。越高代表越‘名副其实’。
        """
        # 注意：排名 1 为最好（小），分数越高越好。正相关系数取绝对值。
        # 如果技术分第一的人也是排名第一，rho = 1.0
        rho, _ = spearmanr(final_rankings, tech_scores)
        return abs(rho)

    @staticmethod
    def calculate_engagement_metric(sim_placements: pd.Series, fan_ranks: pd.Series) -> float:
        """
        指标 2: 观众参与度 (Efficiency/Engagement Index)。
        物理意义：量化‘每一票的权重’。如果选票无法改变排名，参与度即为 0。
        算法：计算选票排名对最终排名的边际贡献（使用相关性作为代理指标）。
        """
        # 物理直觉：如果最终排名完全由粉丝决定，此项为 1.0；如果粉丝完全没用，此项为 0.0。
        rho_fan, _ = spearmanr(sim_placements, fan_ranks)
        return abs(rho_fan)

    def evaluate_regime_performance(self, sim_history_df: pd.DataFrame):
        """
        综合评估一个赛季的运行质量。
        """
        # 1. 提取每个选手的最终表现快照
        final_snapshot = sim_history_df.groupby('celebrity_name').agg({
            'sim_placement': 'min',  # 最终名次 (1=Winner)
            'cum_avg_tech_score': 'last',  # 赛季累计技术表现
            'cum_avg_fan_vote': 'last'  # 赛季累计粉丝偏好
        })

        # 2. 计算 Equity (技术对齐)
        equity = self.calculate_fairness_metric(
            final_snapshot['sim_placement'],
            final_snapshot['cum_avg_tech_score']
        )

        # 3. 计算 Efficiency (民意敏感度)
        efficiency = self.calculate_engagement_metric(
            final_snapshot['sim_placement'],
            final_snapshot['cum_avg_fan_vote']
        )

        return equity, efficiency


# ------------------------------------------------------------------------------
# 师传：回答 Task 4 “更具观赏性”的高阶度量
# ------------------------------------------------------------------------------
def calculate_cliffhanger_index(sim_history_df):
    """
    【学术彩蛋】计算‘悬念指数’。
    物理意义：淘汰当周，倒数两名的得分差距有多小？差距越小，悬念越大，观赏性越高。
    """
    # 逻辑：提取每周淘汰边缘的 Margin 分布
    pass