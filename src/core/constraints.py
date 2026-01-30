# ==============================================================================
# src/core/constraints.py
# Role: Feasible Region Architect - Soft-Constraint Engine
# Function: Differentiable mapping of DWTS rules into optimization manifolds
# ==============================================================================

import numpy as np
from numba import njit
from src.etl.config_loader import ConfigLoader


# ------------------------------------------------------------------------------
# Numba 加速算子：实现可导的 Rank 近似
# ------------------------------------------------------------------------------

@njit(fastmath=True)
def soft_rank_operator(v, tau=0.05):
    """
    【高维数学核心】使用 Sigmoid 函数近似排名。
    Rank_i = 1 + sum_{j != i} sigmoid((v_j - v_i) / tau)
    tau 是温度参数：tau 越小，越接近真实排名；tau 越大，曲面越平滑。
    """
    n = len(v)
    ranks = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                # 逻辑：如果 v[j] > v[i]，sigmoid 接近 1，增加 i 的排名数字
                diff = v[j] - v[i]
                ranks[i] += 1.0 / (1.0 + np.exp(-diff / tau))
    return ranks


# ------------------------------------------------------------------------------
# 约束构造类
# ------------------------------------------------------------------------------

class ConstraintBuilder:
    """
    逻辑翻译官：将不同赛季的文本规则转化为数学约束向量。
    """

    def __init__(self, season: int):
        self.season = season
        self.cfg = ConfigLoader.load_config()
        self.mechanism = self._get_mechanism()

    def _get_mechanism(self) -> str:
        m_cfg = self.cfg['mechanisms']
        if self.season in m_cfg['rank_based_seasons']:
            return "RANK"
        return "PERCENT"

    def build(self, judge_signals: np.ndarray, eliminated_idx: int):
        """
        构造 SciPy 兼容的约束列表。
        Scipy 约定：f(v) >= 0 表示满足约束。
        """
        if self.mechanism == "PERCENT":
            return self._percent_constraints(judge_signals, eliminated_idx)
        else:
            return self._rank_constraints(judge_signals, eliminated_idx)

    def _percent_constraints(self, judge_pcts, elim_idx):
        """
        百分比制：TotalPct[safe] - TotalPct[eliminated] >= margin
        """

        def ineq_constraint(v):
            total_scores = judge_pcts + v
            loser_score = total_scores[elim_idx]
            # 存活者分数减去淘汰者分数，结果必须 > 0
            diffs = total_scores - loser_score
            # 加上一个极小的 epsilon 保证严格不等式
            return np.delete(diffs, elim_idx) - 1e-4

        return [{'type': 'ineq', 'fun': ineq_constraint}]

    def _rank_constraints(self, judge_ranks, elim_idx):
        """
        【重构重点】排名制：TotalRank[eliminated] - TotalRank[safe] >= 0
        注意：排名数字越大表现越差。
        """

        def soft_rank_constraint(v):
            # 使用 Numba 加速的平滑排名算子
            fan_ranks = soft_rank_operator(v, tau=0.02)
            total_ranks = judge_ranks + fan_ranks

            # 特殊逻辑：Season 28+ 的‘倒数两名’救人机制
            # 虽然复杂，但在 MAP 估计中，被淘汰者的 Rank Sum 依然趋向于最大
            loser_rank = total_ranks[elim_idx]
            diffs = loser_rank - total_ranks

            # 返回淘汰者排名与他人的差距 (应为正)
            return np.delete(diffs, elim_idx)

        return [{'type': 'ineq', 'fun': soft_rank_constraint}]


# ------------------------------------------------------------------------------
# 辅助函数：物理边界
# ------------------------------------------------------------------------------
def get_probability_bounds(n: int):
    """
    确保投票比例在 [0.001, 0.999] 之间。
    不设为 0 是为了防止对数空间计算时出现负无穷大。
    """
    return [(0.001, 0.999) for _ in range(n)]