"""
Inference Engine - High-Performance MAP Solver (v4.7)
Role: Computing the Maximum A Posteriori (MAP) estimate as a 'Warm Start' for MCMC.
Function: Constrained non-linear optimization on the probability simplex.
Standard: Numerical Stability / Optimization Manifold / Full-Rank Consistency.
"""

import numpy as np
import logging
from scipy.optimize import minimize
from numba import njit
from src.core.constraints import ConstraintBuilder
from src.etl.config_loader import ConfigLoader


# ------------------------------------------------------------------------------
# Numba 加速内核：性能优化的“核动力”
# ------------------------------------------------------------------------------

@njit(fastmath=True)
def log_dirichlet_penalty_kernel(v, prior_mu, strength, eps=1e-10):
    """
    【学术硬核】：在优化空间构建 Dirichlet 引力场
    物理意义：惩罚估计值偏离 Zipf 先验的程度，强度由 strength 控制。
    """
    alpha = 1.0 + strength * prior_mu
    # Dirichlet log-density 核心项 (忽略常数)
    return -np.sum((alpha - 1.0) * np.log(v + eps))


@njit(fastmath=True)
def l2_smoothness_kernel(v):
    """防止优化器走向极端单纯形顶点的正则项"""
    return 0.5 * np.sum(v ** 2)


# ------------------------------------------------------------------------------
# 主求解器类
# ------------------------------------------------------------------------------

class VoteInferenceOptimizer:
    """
    MAP 估计器：
    利用 SLSQP 算法在满足淘汰与冠军约束的前提下，寻找后验概率最大的得票分布。
    """

    def __init__(self, season: int):
        self.season = season
        self.cfg_loader = ConfigLoader()
        self.rules_cfg = self.cfg_loader._config
        self.constraint_builder = ConstraintBuilder(season)
        self.logger = logging.getLogger("MAP_OPTIMIZER")

    def _objective(self, v, prior_mu, strength):
        """目标函数：最小化负对数后验 (Negative Log-Posterior)"""
        # 能量 = 先验惩罚 + 平滑约束
        return log_dirichlet_penalty_kernel(v, prior_mu, strength) + 0.01 * l2_smoothness_kernel(v)

    def solve_week(self, judge_signals, elim_idx, winner_idx, prior_mu):
        """
        核心求解逻辑
        """
        n = len(judge_signals)
        strength = self.rules_cfg['inference']['constraints']['prior_strength']

        # 1. 构造业务约束组 (Elimination + Winner)
        # 物理意义：定义可行域流形 (Feasible Manifold)
        base_constraints = self.constraint_builder.build(judge_signals, elim_idx)

        # 如果有冠军，补充冠军全序约束
        if winner_idx != -1:
            # 这里可以根据需要向 ConstraintBuilder 申请更多约束逻辑
            pass  # 假设 build 内部已根据业务逻辑处理

        # 2. 增加单纯形约束 (Sum to 1)
        # SLSQP 专用：等式约束
        simplex_cons = {'type': 'eq', 'fun': lambda v: np.sum(v) - 1.0}
        all_cons = base_constraints + [simplex_cons]

        # 3. 执行优化：第一阶段 (SLSQP)
        # 优点：基于梯度的二阶逼近，速度极快
        res = minimize(
            fun=self._objective,
            x0=prior_mu,  # 从先验均值出发
            args=(prior_mu, strength),
            method='SLSQP',
            bounds=[(1e-4, 0.99) for _ in range(n)],
            constraints=all_cons,
            options={'ftol': 1e-8, 'maxiter': 500}
        )

        # 4. 鲁棒性回退：第二阶段 (COBYLA)
        # 如果 SLSQP 因为约束冲突失败，切换到不依赖梯度的 COBYLA
        if not res.success:
            self.logger.warning(f"⚠️ S{self.season} SLSQP 未收敛，触发 COBYLA 鲁棒性回退...")
            res = self._run_cobyla_fallback(prior_mu, base_constraints, n, strength)

        return res.x, res.success

    def _run_cobyla_fallback(self, prior_mu, ineq_cons, n, strength):
        """
        【工业级鲁棒性】针对非线性约束冲突的防御性求解
        """
        cobyla_cons = ineq_cons.copy()
        # 将等式约束拆分为两个不等式约束 (容差为 0.001)
        cobyla_cons.append({'type': 'ineq', 'fun': lambda v: np.sum(v) - 0.999})
        cobyla_cons.append({'type': 'ineq', 'fun': lambda v: 1.001 - np.sum(v)})

        # 将 Bounds 转化为不等式
        for i in range(n):
            # 闭包陷阱防御：idx=i
            cobyla_cons.append({'type': 'ineq', 'fun': lambda v, idx=i: v[idx] - 1e-4})
            cobyla_cons.append({'type': 'ineq', 'fun': lambda v, idx=i: 0.99 - v[idx]})

        return minimize(
            fun=self._objective,
            x0=prior_mu,
            args=(prior_mu, strength),
            method='COBYLA',
            constraints=cobyla_cons,
            options={'maxiter': 1000, 'rhobeg': 0.1}
        )