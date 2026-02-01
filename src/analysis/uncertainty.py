"""
Uncertainty Quantification (UQ) & Forensic Engine (v4.9 - Interface Finalization)
Role: Measuring the "Confidence" and "Historical Consistency" of Bayesian Estimates.
Functions:
    - Full Audit: CI, Entropy, RDI in a single pass.
Standard: Bayesian Rigor / Industrial Robustness.
"""

import numpy as np
import pandas as pd
from scipy.special import softmax
import logging

class UncertaintyAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger("UQ_ENGINE")

    def _calculate_credible_intervals(self, mu: float, sigma: float):
        """内部辅助函数：计算估计参数的 95% 置信区间 (CI)"""
        # 使用高斯近似 (1.96*sigma)，并确保票数在 [0.001, 0.999] 物理约束内
        low_95 = max(0.001, mu - 1.96 * sigma)
        high_95 = min(0.999, mu + 1.96 * sigma)
        return low_95, high_95

    def _calculate_elimination_entropy(self, survival_scores: np.ndarray, tau: float = 0.1) -> float:
        """
        内部辅助函数：计算“淘汰悬念”熵 (Shannon Entropy)。
        p_elim = Softmax(-Survival_Scores / tau)
        """
        p_elim = softmax(-survival_scores / tau)
        eps = 1e-12
        entropy = -np.sum(p_elim * np.log2(p_elim + eps))
        return entropy

    def _calculate_rdi(self, inferred_ranks: np.ndarray, actual_elim_idx: int) -> float:
        """
        内部辅助函数：计算 Rank Displacement Index (RDI) —— 排名位移指标。
        RDI = |实际淘汰者在模型中的倒数排名位置| / (N - 1)
        """
        n = len(inferred_ranks)
        if actual_elim_idx < 0 or actual_elim_idx >= n or n <= 1:
            return 0.0

        # inferred_ranks 是 1-based (1=Best, N=Worst)
        actual_loser_inferred_rank = inferred_ranks[actual_elim_idx]

        # 归一化位移：1.0 表示模型认为他应该是第一名，0.0 表示模型认为他应该是倒数第一
        # displacement = (actual_loser_inferred_rank - 1) / (n - 1)

        # 修正：如果 inferred_ranks 是 Soft-Rank Sum (越小越好)，则逻辑为：
        # 我们使用 Fidelity Score 已经计算了精确的排名还原度，RDI 直接用于位移的量化

        # 假设 inferred_ranks 已经是 Total Rank Sum (越小越好)
        # 找到实际淘汰者在模型中的倒数排名位置 (0=Worst, n-1=Best)
        sorted_indices = np.argsort(inferred_ranks) # [Worst_IDX, ..., Best_IDX]
        predicted_pos = np.where(sorted_indices == actual_elim_idx)[0][0] # 0-based index

        # RDI = 偏离倒数第一的距离 / 最大可能距离
        return predicted_pos / (n - 1)

    def run_full_audit(self, df_platinum: pd.DataFrame) -> pd.DataFrame:
        """
        【统一接口】：处理所有 UQ 指标。
        """
        self.logger.info("正在执行 UQ 全景审计 (CI, Entropy, RDI)...")
        audit_records = []
        groups = df_platinum.groupby(['season', 'week_num'])

        for (s, w), group in groups:
            group = group.reset_index(drop=True)
            n = len(group)
            if n < 2: continue

            # --- 构造周度信号 ---
            j_raw = group['week_avg_score'].values
            j_share = j_raw / (j_raw.sum() + 1e-9)
            f_share = group['est_fan_vote_mu'].values
            survival_score = j_share + f_share

            # 1. 计算系统熵 (H)
            p_elim = softmax(-survival_score / 0.1)
            h_sys = -np.sum(p_elim * np.log2(p_elim + 1e-12))
            certainty = 1.0 - (h_sys / np.log2(n)) if n > 1 else 1.0

            # 2. 识别真实淘汰者 (Elimination Index)
            elim_mask = (group['final_status'] == 'Eliminated') & \
                        (group['eliminated_week'] == w)
            elim_indices = np.where(elim_mask)[0]

            # 3. 计算 RDI (Rank Displacement Index) - [核心修复逻辑]
            current_rdi = 0.0
            if len(elim_indices) > 0:
                actual_loser_idx = elim_indices[0]

                # 计算模型预测的排名 (1=Worst, n=Best)
                # 排序总生存分：分数最低（最应该淘汰）的人排在第 0 位
                inferred_rank_indices = np.argsort(survival_score)

                # 找到实际淘汰者在模型预测的“应该淘汰”队列中的位置
                predicted_pos = np.where(inferred_rank_indices == actual_loser_idx)[0][0]  # 0-based index

                # RDI = 偏离倒数第一（0）的距离 / 最大可能距离
                current_rdi = predicted_pos / (n - 1)

            # --- 4. 结果分发与 CI ---
            for i in range(len(group)):
                mu = group.loc[i, 'est_fan_vote_mu']
                sigma = group.loc[i, 'est_fan_vote_sigma']

                low_95, high_95 = self._calculate_credible_intervals(mu, sigma)

                audit_records.append({
                    'season': s,
                    'week_num': w,
                    'celebrity_name': group.loc[i, 'celebrity_name'],
                    'uq_entropy': h_sys,
                    'uq_certainty': certainty,
                    'uq_low_95': low_95,
                    'uq_high_95': high_95,
                    'forensic_rdi': current_rdi,
                    'is_high_suspicion': h_sys > 1.5
                })

        return pd.DataFrame(audit_records)

    # 兼容旧调用的方法 (防止 main.py 报错)
    def process_platinum_uncertainty(self, df_platinum: pd.DataFrame) -> pd.DataFrame:
        # 为了与 main.py 的 Stage 3 逻辑同步，我们返回一个不含 Entropy 和 RDI 的子集
        df_full = self.run_full_audit(df_platinum)
        return df_full[['season', 'week_num', 'celebrity_name', 'uq_low_95', 'uq_high_95']]