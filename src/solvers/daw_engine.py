# ==============================================================================
# src/solvers/daw_engine.py
# Role: Dynamic Adaptive Weighting (DAW) System Core (v5.0)
# Function: Generating time-dependent weights w(t) using Sigmoid transition.
# Physics: Modeling the "Power Handover" from Populism to Meritocracy.
# Standard: Numerical Stability / Vectorized Operations / Configurable Hyperparams.
# ==============================================================================

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Union


class DAWEngine:
    """
    DAW (Dynamic Adaptive Weighting) 物理引擎：
    实现基于广义 Sigmoid 函数的动态权力移交机制。

    Formula:
    w_judge(t) = w_min + (w_max - w_min) * Sigmoid(k * (t - t0))
    """

    def __init__(self,
                 default_k: float = 10.0,
                 default_t0: float = 0.6,
                 w_min: float = 0.2,
                 w_max: float = 0.9):
        """
        初始化 DAW 引擎参数。
        :param default_k: 切换斜率 (Steepness)。k 越大，切换越迅速。
        :param default_t0: 切换中点 (Pivot Point)。0.6 表示赛程 60% 处达到 50/50 平衡。
        :param w_min: 评委权重的硬下限 (初期给观众留足面子)。
        :param w_max: 评委权重的硬上限 (决赛期防止独裁)。
        """
        self.logger = logging.getLogger("DAW_ENGINE")
        self.default_k = default_k
        self.default_t0 = default_t0
        self.w_min = w_min
        self.w_max = w_max

    @staticmethod
    def _stable_sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        【数值防御】数值稳定的 Sigmoid 实现。
        防止在高斜率 k 寻优过程中产生指数爆炸 (Overflow)。
        """
        # 使用 numpy 的 where 处理数组输入，防止 exp(709) 溢出
        return np.where(x >= 0,
                        1.0 / (1.0 + np.exp(-x)),
                        np.exp(x) / (1.0 + np.exp(x)))

    def compute_judge_weight(self,
                             current_week: int,
                             total_weeks: int,
                             k: float = None,
                             t0: float = None) -> float:
        """
        计算特定周次的评委权重 w_J(t)。
        """
        # 1. 参数回退机制 (支持 Grid Search 动态注入)
        k = k if k is not None else self.default_k
        t0 = t0 if t0 is not None else self.default_t0

        # 2. 归一化赛程进度 t \in [0, 1]
        # 使用 max(1, total-1) 防止除零
        if total_weeks <= 1:
            progress = 1.0
        else:
            # 赛程进度从 0 开始到 1 结束
            progress = (current_week - 1) / (total_weeks - 1)

        # 3. Sigmoid 映射
        # 乘以 10 是为了让 k 的量级在 [1, 20] 之间比较直观
        logit = k * (progress - t0) * 10.0
        sigmoid_val = self._stable_sigmoid(logit)

        # 4. 仿射变换映射到 [w_min, w_max]
        w_judge = self.w_min + (self.w_max - self.w_min) * sigmoid_val

        return w_judge

    def calculate_combined_score(self,
                                 j_rank: np.ndarray,
                                 f_rank: np.ndarray,
                                 current_week: int,
                                 total_weeks: int,
                                 k: float = None,
                                 t0: float = None) -> Tuple[np.ndarray, float]:
        """
        【核心裁决接口】计算 DAW 机制下的混合得分。

        Input:
            j_rank: 评委排名 (1=Best, 数值越小越好)
            f_rank: 粉丝排名 (1=Best, 数值越小越好)
        Output:
            combined_score: 加权排名和 (数值越小越好)
            applied_weight: 当周实际使用的评委权重
        """
        # 1. 获取当周权重
        w_j = self.compute_judge_weight(current_week, total_weeks, k, t0)
        w_f = 1.0 - w_j

        # 2. 执行加权序数聚合 (Weighted Ordinal Aggregation)
        # 物理意义：在保持 Rank 机制“低通滤波”特性的同时，引入动态增益
        combined_score = w_j * j_rank + w_f * f_rank

        return combined_score, w_j

    def get_trajectory_preview(self, total_weeks=10, k=None, t0=None) -> pd.DataFrame:
        """
        生成全赛季权重演化预览表。
        用于可视化绘制 (Task 4 Trajectory Plot)。
        """
        weeks = np.arange(1, total_weeks + 1)
        weights = [self.compute_judge_weight(w, total_weeks, k, t0) for w in weeks]

        return pd.DataFrame({
            'week': weeks,
            'judge_weight': weights,
            'fan_weight': [1 - w for w in weights],
            'phase': ['Populism' if w < 0.5 else 'Meritocracy' for w in weights]
        })


# --- 单元测试 (Unit Test) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 1. 实例化引擎
    engine = DAWEngine(k=5.0, t0=0.6)

    # 2. 测试权重轨迹
    print(">>> DAW 权重演化预览 (10周赛季):")
    df_traj = engine.get_trajectory_preview(total_weeks=10)
    print(df_traj.round(3))

    # 3. 测试裁决逻辑 (模拟 Bobby Bones 场景)
    # 假设：评委给倒数第一(Rank 10)，观众给正数第一(Rank 1)
    j_r = np.array([10.0])
    f_r = np.array([1.0])

    # 在第 2 周 (应由观众主导)
    s_early, w_early = engine.calculate_combined_score(j_r, f_r, current_week=2, total_weeks=10)
    print(f"\nWeek 2 (Early) Score: {s_early[0]:.2f} (w_j={w_early:.2f}) -> 倾向于粉丝")

    # 在第 9 周 (应由评委主导)
    s_late, w_late = engine.calculate_combined_score(j_r, f_r, current_week=9, total_weeks=10)
    print(f"Week 9 (Late) Score:  {s_late[0]:.2f} (w_j={w_late:.2f}) -> 倾向于评委")