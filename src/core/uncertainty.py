"""
MCM 2026 Problem C: Uncertainty Mathematics Kernel (v5.6)
Role: Low-level Mathematical Primitives for UQ and Forensics.
Function: Calculation of Entropy, KL-Divergence, and Rank Topology Metrics.
Physics: Quantifying the 'Thermodynamic Entropy' of the voting system.
Standard: IEEE Floating Point Safety / Information Theory Definitions.
"""

import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
import logging

# 全局数值稳定常数
EPSILON = 1e-12

class UncertaintyQuantifier:
    """
    不确定性量化算子库 (Static Math Kernel).
    为上层分析管道提供纯粹的数学计算服务。
    """

    @staticmethod
    def calculate_shannon_entropy(probs: np.ndarray) -> float:
        """
        计算香农熵 (Shannon Entropy) - 单位: Bits.
        公式: H(X) = - sum(p(x) * log2(p(x)))

        [物理意义]:
        衡量系统的“混乱度”。H=0 表示结果完全确定（独裁）；
        H=log2(N) 表示结果完全随机（最大熵状态）。
        """
        # 1. 归一化校验 (防御性编程)
        p_sum = np.sum(probs)
        if abs(p_sum - 1.0) > 1e-6:
            # 如果不是概率分布，尝试归一化（针对 Score 场景）
            probs = probs / (p_sum + EPSILON)

        # 2. 截断 (Clipping) 防止 log(0)
        p_safe = np.clip(probs, EPSILON, 1.0)

        # 3. 计算熵 (Base 2)
        return entropy(p_safe, base=2)

    @staticmethod
    def calculate_normalized_certainty(h: float, n: int) -> float:
        """
        计算相对确定性指数 (Relative Certainty Index, RCI).
        范围: [0.0, 1.0]. 1.0 = 完全确定.
        """
        if n <= 1: return 1.0
        h_max = np.log2(n)
        # RCI = 1 - (H / H_max)
        rci = 1.0 - (h / h_max)
        return max(0.0, rci)

    @staticmethod
    def calculate_kl_divergence(p_posterior: np.ndarray, q_prior: np.ndarray) -> float:
        """
        计算库尔巴克-莱布勒散度 (KL Divergence).
        公式: D_KL(P || Q) = sum(p * log(p / q))

        [学术价值]:
        衡量“观测数据”提供了多少“额外信息量”。
        如果 KL 很大，说明 MCMC 剧烈修正了先验，意味着当周比赛发生了“意料之外”的反转（如 S27 Bobby Bones）。
        """
        p = np.clip(p_posterior, EPSILON, 1.0)
        q = np.clip(q_prior, EPSILON, 1.0)

        return np.sum(p * np.log2(p / q))

    @staticmethod
    def calculate_rdi(inferred_values: np.ndarray, actual_loser_idx: int, mode: str = "RANK") -> float:
        """
        计算排名位移指数 (Rank Displacement Index, RDI).

        [定义]:
        RDI = |Predicted_Rank(Loser) - Worst_Rank| / (N - 1)

        :param inferred_values: 模型推断值 (若是 RANK 模式，值越小越好；若是 SCORE 模式，值越大越好)
        :param actual_loser_idx: 真实淘汰者的数组索引
        :param mode: "RANK" (序数) 或 "SCORE" (基数/概率)
        :return: 0.0 (完美预测) -> 1.0 (完全颠倒)
        """
        n = len(inferred_values)
        if n <= 1 or actual_loser_idx < 0 or actual_loser_idx >= n:
            return 0.0

        # 1. 统一转化为“名次” (0-based, 0=倒数第一/最差, n-1=第一名/最好)
        if mode == "SCORE":
            # 分数越高，质量越好 -> 排序后 index 越大
            # argsort 会把小值排前面，所以 argsort 的结果就是从 差 -> 好 的索引
            sorted_indices = np.argsort(inferred_values)
        else:
            # RANK 模式：数值越小越好 (1, 2, 3...)
            # 我们需要 0=最差(数值大), n-1=最好(数值小)
            # argsort(-rank) -> 大值(差)排前面
            sorted_indices = np.argsort(-inferred_values)

        # 2. 找到真实淘汰者在“能力序列”中的位置
        # 如果模型认为他是最差的，他应该出现在 sorted_indices[0]
        predicted_performance_pos = np.where(sorted_indices == actual_loser_idx)[0][0]

        # 3. 计算位移
        # 理想位置是 0 (他就是最差的)
        displacement = abs(predicted_performance_pos - 0)

        # 4. 归一化
        return displacement / (n - 1)

    @staticmethod
    def calculate_elimination_probability(survival_scores: np.ndarray, tau: float = 0.1) -> np.ndarray:
        """
        基于生存分计算淘汰概率分布 (Softmax).
        物理意义：将连续的生存强度转化为离散的淘汰风险。
        P(Elim) = Softmax(-Score / Temperature)
        """
        # 负号：生存分越高，淘汰概率越低
        return softmax(-survival_scores / tau)

# --- 单元测试 ---
if __name__ == "__main__":
    # 配置日志以便观察
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("UQ_KERNEL_TEST")

    uq = UncertaintyQuantifier()

    # 1. 测试熵
    probs = np.array([0.5, 0.25, 0.25])
    h = uq.calculate_shannon_entropy(probs)
    print(f"Entropy: {h:.4f} bits (Expected: 1.5)")

    # 2. 测试 RDI (Score 模式)
    # 场景：C 是淘汰者。
    # Case A: 模型认为 C 最差 (0.1) -> RDI 应为 0
    scores_a = np.array([0.9, 0.5, 0.1]) # A, B, C
    rdi_a = uq.calculate_rdi(scores_a, actual_loser_idx=2, mode="SCORE")
    print(f"RDI (Perfect): {rdi_a:.4f}")

    # Case B: 模型认为 A 最差 (0.1), C 最好 (0.9) -> RDI 应为 1.0 (完全颠倒)
    scores_b = np.array([0.1, 0.5, 0.9]) # A, B, C(淘汰者)
    rdi_b = uq.calculate_rdi(scores_b, actual_loser_idx=2, mode="SCORE")
    print(f"RDI (Worst): {rdi_b:.4f}")

    # 3. 测试 KL 散度
    prior = np.array([0.33, 0.33, 0.33])
    post = np.array([0.9, 0.05, 0.05])
    kl = uq.calculate_kl_divergence(post, prior)
    print(f"KL Divergence: {kl:.4f} bits (Evidence Strength)")