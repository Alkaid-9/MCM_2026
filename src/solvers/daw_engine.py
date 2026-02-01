# ==============================================================================
# src/solvers/daw_engine.py
# Role: Dynamic Adaptive Weighting (DAW) System Core
# Function: Generating time-dependent weights w(t) using Sigmoid transition.
# Physics Intuition: Modeling the transition from "Public Sentiment Exploration"
#                    to "Technical Merit Exploitation" as a Phase Transition.
# Standard: MIT/Stanford Math Dept Standards for Mechanism Design.
# ==============================================================================

import numpy as np
import pandas as pd
import logging
from scipy.stats import rankdata
from src.etl.config_loader import ConfigLoader


class DAWEngine:
    """
    DAW (Dynamic Adaptive Weighting) 物理引擎：
    实现基于广义 Sigmoid 函数的动态权力移交机制。

    [博弈论背景]
    该机制旨在建立“激励相容性 (Incentive Compatibility)”。
    通过随赛程动态调整权重，确保选手在决赛阶段的最优策略（Nash Equilibrium）
    必然是“提升舞技”而非“社交媒体营销”。
    """

    def __init__(self, config_loader: ConfigLoader = None):
        self.logger = logging.getLogger("DAW_ENGINE")
        self.cfg_loader = config_loader if config_loader else ConfigLoader()

        # 加载持久化配置
        daw_cfg = self.cfg_loader.load_config().get('task4_mechanism_design', {}).get('dynamic_weighting', {})

        # 核心超参数
        self.default_k = float(daw_cfg.get('sigmoid_k', 10.0))
        self.default_t0 = float(daw_cfg.get('sigmoid_t0', 0.6))

        # 机制约束边界 (Meritocratic Floor & Populist Ceiling)
        # 物理意义：防止任何一方拥有绝对裁决权（Dictatorship），保持系统的“脆弱性平衡”以提高观赏性
        self.w_min = 0.3  # 评委权重下限
        self.w_max = 0.8  # 评委权重上限

    @staticmethod
    def _stable_sigmoid(x: float) -> float:
        """
        数值稳定的 Sigmoid 实现。
        物理意义：防止在高斜率 k 寻优过程中产生指数爆炸（Floating Point Overflow）。
        """
        if x >= 0:
            z = np.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = np.exp(x)
            return z / (1.0 + z)

    def compute_judge_weight(self, progress: float, k: float = None, t0: float = None) -> float:
        """
        计算即时评委权重 w_J(t)。

        公式：w(t) = w_min + (w_max - w_min) * Sigmoid(k * (t - t0) * 10)

        :param progress: 归一化时间进度 t \in [0, 1]
        :param k: 转移速率 (Aggressiveness of policy shift)
        :param t0: 权力交接中点 (Phase transition pivot)
        """
        # 参数鲁棒性回退
        k = k if k is not None else self.default_k
        t0 = t0 if t0 is not None else self.default_t0

        # 边界防御：progress 必须在 [0, 1]
        t = np.clip(progress, 0.0, 1.0)

        # 线性空间映射到 Sigmoid 敏感区
        # 乘以 10.0 是为了让 k=1 对应于温和的过渡，k=10 对应于剧烈的体制转换
        logit = k * (t - t0) * 10.0
        sig_val = self._stable_sigmoid(logit)

        # 仿射变换映射回目标权重区间
        return self.w_min + (self.w_max - self.w_min) * sig_val

    def get_weight_trajectory(self, total_weeks: int, k: float = None, t0: float = None) -> np.ndarray:
        """
        [高性能接口] 生成全赛季权重演化向量。
        物理意义：预热计算图，供大规模 Monte Carlo 仿真调用。
        """
        if total_weeks < 1:
            return np.array([self.w_min])

        weeks = np.arange(1, total_weeks + 1)
        progress_vec = weeks / total_weeks

        # 利用向量化运算提速
        weights = np.array([self.compute_judge_weight(p, k, t0) for p in progress_vec])
        return weights

    def calculate_combined_score(self,
                                 judge_ranks: np.ndarray,
                                 fan_ranks: np.ndarray,
                                 progress: float,
                                 k: float = None,
                                 t0: float = None):
        """
        执行 DAW 机制下的混合裁决。

        [算法本质]
        这是一个“动态序数聚合问题”。
        Score(t) = w_J(progress) * Rank_Judge + (1 - w_J(progress)) * Rank_Fan

        注意：在本项目坐标系中，排名越小越优秀 (1=Best)。
        """
        w_j = self.compute_judge_weight(progress, k, t0)
        w_f = 1.0 - w_j

        # 执行加权序数叠加
        # 使用线性组合保持排名空间的拓扑结构
        combined_score = w_j * judge_ranks + w_f * fan_ranks

        return combined_score, w_j

    def simulate_week_outcome(self,
                              week_data: pd.DataFrame,
                              total_weeks: int,
                              k: float = None,
                              t0: float = None) -> pd.DataFrame:
        """
        [单周审计核]：评估 DAW 机制在特定周次的决策结果。

        物理意义：用于对比“Bobby Bones 悖论”在 DAW 下是否会触发“体制自愈”。
        """
        if week_data.empty:
            return week_data

        df = week_data.copy()
        current_week = int(df['week_num'].iloc[0])
        progress = current_week / max(total_weeks, 1)

        # 1. 信号提取（强制转换为降序排名）
        # 分数越高 -> 排名数字越小 (1st)
        j_ranks = rankdata(-df['week_avg_score'].values, method='min')

        # 粉丝票占比越高 -> 排名数字越小 (1st)
        # 注意：这里必须注入 Task 1 产出的 est_fan_vote_mu
        f_ranks = rankdata(-df['est_fan_vote_mu'].values, method='min')

        # 2. 机制执行
        daw_scores, applied_weight = self.calculate_combined_score(
            j_ranks, f_ranks, progress, k, t0
        )

        # 3. 结果固化
        df['daw_score'] = daw_scores
        # 最终排名再次进行 rank 以处理并列情况
        df['daw_rank'] = rankdata(daw_scores, method='min')
        df['judge_weight_applied'] = applied_weight

        return df

    def validate_params(self, k: float, t0: float) -> bool:
        """
        [工业级质量检查]：验证优化器生成的参数是否符合物理直觉。
        """
        # 切换中点不能超出生赛程范围，斜率不能为负
        if not (0.1 <= t0 <= 0.9):
            return False
        if k <= 0:
            return False
        return True