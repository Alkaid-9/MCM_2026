# ==============================================================================
# src/core/optimizer.py
# Role: Inference Engine - High-Performance MAP Solver
# Function: Log-Posterior Optimization with Robust Fallback Mechanism
# ==============================================================================

import numpy as np
import logging
from scipy.optimize import minimize
from numba import njit
from src.core.constraints import ConstraintGenerator


# ------------------------------------------------------------------------------
# Numba 加速内核：性能优化的“核动力”
# ------------------------------------------------------------------------------

@njit(fastmath=True)
def sigmoid(x, tau=0.1):
    return 1.0 / (1.0 + np.exp(-x / tau))


@njit(fastmath=True)
def log_prior_penalty(v, prior_v, eps=1e-12):
    """对数空间的先验惩罚 (Log-MAP)"""
    # 物理意义：惩罚估计值偏离‘先验流形’的程度
    log_v = np.log(v + eps)
    log_p = np.log(prior_v + eps)
    return np.sum((log_v - log_p) ** 2)


# ------------------------------------------------------------------------------
# 主求解器类
# ------------------------------------------------------------------------------

class VoteInferenceOptimizer:
    def __init__(self, season: int):
        self.season = season
        self.constraint_gen = ConstraintGenerator(season)

    def _objective(self, v, prior_v):
        """目标函数：最小化负对数后验"""
        # 先验约束项 + L2正则平滑项
        return log_prior_penalty(v, prior_v) + 0.05 * np.sum(v ** 2)

    def solve_week(self, judge_signals, eliminated_idx, prior_v=None):
        n = len(judge_signals)
        if prior_v is None:
            prior_v = np.ones(n) / n

        # 1. 构造基础约束
        constraints = self.constraint_gen.get_constraints(judge_signals, eliminated_idx)

        # 增加总和等于 1 的等式约束 (SLSQP 专用)
        sum_cons = {'type': 'eq', 'fun': lambda v: np.sum(v) - 1.0}
        slsqp_constraints = constraints + [sum_cons]

        # 2. 首先尝试高效的 SLSQP 求解器
        res = minimize(
            fun=self._objective,
            x0=prior_v,
            args=(prior_v,),
            method='SLSQP',
            bounds=[(0.001, 0.999) for _ in range(n)],
            constraints=slsqp_constraints,
            options={'ftol': 1e-7, 'maxiter': 1000}
        )

        # 3. 核心修复逻辑：如果 SLSQP 失败，执行手动约束转换并降级到 COBYLA
        if not res.success:
            logging.warning(f"[Solver] Season {self.season} SLSQP 未收敛，正在执行约束转换并切换至 COBYLA...")
            res = self._run_cobyla_fallback(prior_v, constraints, n)

        return res.x, res.success

    def _run_cobyla_fallback(self, prior_v, original_ineq_constraints, n):
        """
        【工业级鲁棒性】为 COBYLA 重新构造约束环境
        COBYLA 不支持 'eq' 和 'bounds'，必须全部转化为 'ineq'
        """
        # A. 转化原有的不等式约束
        cobyla_cons = original_ineq_constraints.copy()

        # B. 重点：将等式 sum(v)=1 转化为双向不等式 (sum >= 0.99 且 sum <= 1.01)
        # 物理意义：通过给等式留出微小缝隙，使非梯度下降算法能找到可行域
        cobyla_cons.append({'type': 'ineq', 'fun': lambda v: np.sum(v) - 0.999})
        cobyla_cons.append({'type': 'ineq', 'fun': lambda v: 1.001 - np.sum(v)})

        # C. 重点：将 Bounds 转化为不等式 (0.001 <= v <= 0.999)
        # 注意闭包陷阱：必须在 lambda 中传入默认参数以锁定当前循环的 i
        for i in range(n):
            cobyla_cons.append({'type': 'ineq', 'fun': lambda v, idx=i: v[idx] - 0.001})
            cobyla_cons.append({'type': 'ineq', 'fun': lambda v, idx=i: 0.999 - v[idx]})

        return minimize(
            fun=self._objective,
            x0=prior_v,
            args=(prior_v,),
            method='COBYLA',
            constraints=cobyla_cons,
            options={'maxiter': 2000, 'rhobeg': 0.1}  # rhobeg 控制初始搜索步长
        )


# ------------------------------------------------------------------------------
# 辅助计算函数
# ------------------------------------------------------------------------------
def calculate_estimation_entropy(v_optimized):
    """计算估计分布的熵 (衡量不确定性)"""
    v = np.clip(v_optimized, 1e-12, 1.0)
    return -np.sum(v * np.log2(v))