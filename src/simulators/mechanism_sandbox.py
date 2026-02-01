# ==============================================================================
# src/simulators/mechanism_sandbox.py
# Role: High-Frequency Simulation Kernel (The "Physics" of the Game)
# Function: Differentiable and Numba-accelerated elimination logic
# Mechanisms: Rank-based, Percent-based, and Judge's Save override
# ==============================================================================

import numpy as np
from numba import njit


# ------------------------------------------------------------------------------
# 机制类型定义 (Numba 友好型整数枚举)
# 0: PERCENT_BASED (S3-S27)
# 1: RANK_BASED (S1-S2, S28+)
# 2: DAW_DYNAMIC (Your Proposed System)
# ------------------------------------------------------------------------------

@njit(fastmath=True)
def _get_ranks_numba(x):
    """
    Numba 版本的 rankdata (method='min')。
    分数越高，排名数字越小 (1 为第一名)。
    """
    n = len(x)
    ranks = np.zeros(n, dtype=np.float64)
    for i in range(n):
        r = 1.0
        for j in range(n):
            if x[j] > x[i]:  # 统计比我分高的人
                r += 1.0
        ranks[i] = r
    return ranks


@njit(fastmath=True)
def evaluate_elimination(judge_scores, fan_votes, mech_type=0, weight_j=0.5, enable_save=False):
    """
    【核心裁决函数】
    输入：评委技术分向量，粉丝得票权重向量。
    返回：被淘汰选手的索引 (eliminated_idx)。
    """
    n = len(judge_scores)
    total_metrics = np.zeros(n, dtype=np.float64)

    # --- 1. 计算博弈综合分 (Survival Metric) ---
    if mech_type == 0:
        # 百分比制：Total = J% + F% (越高越安全)
        j_pct = judge_scores / (np.sum(judge_scores) + 1e-9)
        total_metrics = j_pct + fan_votes

    elif mech_type == 1:
        # 排名制：Total = Rank(J) + Rank(F) (越小越安全)
        j_rank = _get_ranks_numba(judge_scores)
        f_rank = _get_ranks_numba(fan_votes)
        # 注意：这里取负号，是为了统一成“越高越安全”逻辑
        total_metrics = -(j_rank + f_rank)

    elif mech_type == 2:
        # 动态权重制 (DAW): Total = w*J_rank + (1-w)*F_rank
        j_rank = _get_ranks_numba(judge_scores)
        f_rank = _get_ranks_numba(fan_votes)
        total_metrics = -(weight_j * j_rank + (1.0 - weight_j) * f_rank)

    # --- 2. 识别倒数两名 (Bottom Two) ---
    # 按照综合分排序，找出最低的两个索引
    bottom_indices = np.argsort(total_metrics)[:2]

    # --- 3. 执行评委救济机制 (The Judge's Save / Circuit Breaker) ---
    if enable_save and n > 2:
        idx_a = bottom_indices[0]
        idx_b = bottom_indices[1]
        # 评委决定：在倒数两名中，救回技术分高的，淘汰技术分低的
        if judge_scores[idx_a] >= judge_scores[idx_b]:
            return idx_b  # 淘汰 idx_b
        else:
            return idx_a  # 淘汰 idx_a
    else:
        # 传统机制：直接淘汰总分最低者
        return bottom_indices[0]


@njit(parallel=True)
def run_monte_carlo_survival(judge_scores, fan_mu, fan_sigma, n_sims=1000, mech_type=0):
    """
    【学术杀手锏】计算随机扰动下的存活率。
    物理意义：量化‘估计误差’对结局稳定性的影响。
    """
    n = len(judge_scores)
    survival_counts = np.zeros(n)

    for i in range(n_sims):
        # 注入高斯噪音模拟真实投票波动
        noisy_votes = fan_mu + np.random.standard_normal(n) * fan_sigma
        # 投影回单纯形
        noisy_votes = np.maximum(noisy_votes, 1e-5)
        noisy_votes /= np.sum(noisy_votes)

        elim_idx = evaluate_elimination(judge_scores, noisy_votes, mech_type)

        for j in range(n):
            if j != elim_idx:
                survival_counts[j] += 1

    return survival_counts / n_sims