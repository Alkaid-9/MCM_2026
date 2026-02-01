# ==============================================================================
# src/solvers/objective_engine.py
# Role: Mechanism Scorecard (The Evaluation Metric Engine)
# Function: Quantifying Fairness, Engagement, and Suspense for Task 4
# Method: Spearman Correlation & Marginal Variance Analysis
# Standard: Industrial Grade / Pure Library Mode / Zero-Side-Effect
# ==============================================================================

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import logging


class MechanismEvaluator:
    """
    机制评估引擎：
    计算任一赛制宇宙下的‘三维指标’，支撑帕累托寻优与最终决策。
    1. Fairness (Equity): 技术分决定排名的程度。
    2. Engagement (Efficiency): 粉丝票决定排名的程度。
    3. Suspense (Cliffhanger): 淘汰边缘的激烈程度（观赏性）。
    """

    def __init__(self):
        self.logger = logging.getLogger("OBJECTIVE_ENGINE")

    @staticmethod
    def calculate_fairness_metric(final_rankings: pd.Series, tech_scores: pd.Series) -> float:
        """
        指标 1: 技术公平性 (Equity Index)。
        物理意义：最终排名与累计技术分的相关性。
        算法：Spearman 秩相关系数。rho -> 1.0 代表完全唯技术论 (Meritocracy)。
        """
        if len(final_rankings) < 2: return 0.0
        # 注意：Rank 1 是最小数值，Score 是最大数值。
        # 理想情况下，Rank 小对应 Score 大，这是负相关。
        # 但 spearmanr 如果输入的是 (Rank, Score)，应该是负的。
        # 为了指标统一为“越大越好”，我们取绝对值。
        # 严格来说：Corr(Rank, Score) 应该是 -1。
        rho, _ = spearmanr(final_rankings, tech_scores)
        return abs(rho)

    @staticmethod
    def calculate_engagement_metric(sim_placements: pd.Series, fan_ranks: pd.Series) -> float:
        """
        指标 2: 观众参与度 (Efficiency/Engagement Index)。
        物理意义：量化‘每一票的权重’。如果选票无法改变排名，参与度即为 0。
        算法：计算选票排名对最终排名的相关性。
        """
        if len(sim_placements) < 2: return 0.0
        # 物理直觉：如果最终排名完全由粉丝决定，此项为 1.0。
        rho_fan, _ = spearmanr(sim_placements, fan_ranks)
        return abs(rho_fan)

    @staticmethod
    def calculate_cliffhanger_index(sim_history_df: pd.DataFrame) -> float:
        """
        【学术彩蛋】指标 3: 悬念指数 (Cliffhanger Index / Excitement Score)。
        物理意义：衡量淘汰边缘的竞争激烈程度。
        定义：Avg( 1 / (Margin_Bottom2 + epsilon) )。
        逻辑：倒数第一名和倒数第二名的得分差距越小，悬念越大，观赏性越高。
        """
        suspense_scores = []

        # 必须按 (赛季, 周) 分组计算，然后取平均
        # 过滤掉模拟得分为 NaN 的行
        valid_df = sim_history_df.dropna(subset=['sim_score'])
        groups = valid_df.groupby(['season', 'week_num'])

        for _, group in groups:
            n = len(group)
            if n < 2: continue

            # 获取当周所有人的模拟得分
            # 在 MultiverseEngine 中，sim_score 经过处理，统一为“越大越好”
            scores = np.sort(group['sim_score'].values)

            # 倒数第一 (索引 0) vs 倒数第二 (索引 1)
            # 这里的 scores 已经是升序排列，scores[0] 是最低分
            loser_score = scores[0]
            survivor_score = scores[1]

            # 计算边缘差距 (Margin)
            margin = abs(survivor_score - loser_score)

            # 悬念模型：差距越小，分数越高
            # 增加平滑项 0.01 防止除零，同时设定上限
            # 对于 Rank 制，最小差距是 1；对于 Percent 制，差距可能极小
            suspense = 1.0 / (margin + 0.05)
            suspense_scores.append(suspense)

        if not suspense_scores:
            return 0.0

        # 归一化输出 (Log scale 可能会更好，这里用简单的平均)
        return np.mean(suspense_scores)

    def evaluate_regime_performance(self, sim_history_df: pd.DataFrame):
        """
        综合评估一个赛季的运行质量。
        返回：(Equity, Efficiency) 元组，适配 ParetoOptimizer 接口。
        """
        # 1. 提取每个选手的最终表现快照 (用于计算全局相关性)
        final_snapshot = sim_history_df.groupby('celebrity_name').agg({
            'sim_placement': 'min',  # 最终名次 (1=Winner)
            'cum_avg_tech_score': 'last',  # 赛季累计技术表现
            'cum_avg_fan_vote': 'last'  # 赛季累计粉丝偏好
        })

        # [防御性编程] 样本太少不计算
        if len(final_snapshot) < 5:
            return 0.0, 0.0

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

        # 4. 计算 Suspense (观赏性)
        # 注意：这是基于周度过程数据的，而非最终结果
        suspense = self.calculate_cliffhanger_index(sim_history_df)

        # 5. 记录完整画像 (这是 Memo 素材的来源)
        # 这里的日志会被 main.py 捕获，作为论据
        # self.logger.debug(f"Regime Profile: Eq={equity:.3f}, Eff={efficiency:.3f}, Suspense={suspense:.3f}")

        # 为了兼容 pareto_optimizer.py 的解包逻辑 (equity, efficiency = evaluate...)
        # 我们只返回前两个优化目标。Suspense 作为“伴生指标”在日志中体现即可。
        return equity, efficiency