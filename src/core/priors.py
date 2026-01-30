"""
MCM 2026 Problem C: Bayesian Prior Architect
Role: Transforming ETL features into Informative Priors (Zipf's Law & Log-Normal).
Standard: O-Prize Academic Rigor / Quantitative Finance Pre-Computation.
"""

import numpy as np
import pandas as pd
import logging
from scipy.stats import rankdata
from src.etl.config_loader import ConfigLoader


class VotePriors:
    """
    贝叶斯先验生成器：为逆向优化提供带有物理意义的初始正则化流形。
    理论基础：社交网络关注度的长尾分布特性 (Zipf's Law)。
    """

    def __init__(self):
        self.cfg = ConfigLoader()
        self.p_cfg = self.cfg.get_priors_config()
        self.logger = logging.getLogger("PRIOR_ARCHITECT")

    @staticmethod
    def calculate_shannon_entropy(v: np.ndarray) -> float:
        """计算分布的香农熵，用于对比‘后验’相对‘先验’的信息增益。"""
        v_safe = np.clip(v, 1e-12, 1.0)
        return -np.sum(v_safe * np.log2(v_safe))

    def _generate_zipf_distribution(self, n: int, alpha: float) -> np.ndarray:
        """
        生成纯粹的 Zipf (幂律) 分布。
        公式: P(r) = C / (r^alpha)，r 为排名。
        物理意义：alpha 越大，头部效应越明显（‘顶流通吃’）。
        """
        ranks = np.arange(1, n + 1)
        weights = 1.0 / (ranks ** alpha)
        return weights / weights.sum()

    def informative_prior(self, week_df: pd.DataFrame) -> np.ndarray:
        """
        【核心学术亮点】融合‘黄金因子’生成启发式先验。
        逻辑流：
        1. 提取 Partner Alpha (基本盘) 和 Score Momentum (当周势能)。
        2. 线性融合生成‘潜在大众热度指数’。
        3. 将热度转换为 Rank，映射到 Zipf 幂律分布上。
        """
        n = len(week_df)
        if n == 0:
            return np.array([])

        # --- 1. 提取因子并归一化 (防量纲污染) ---
        alpha = week_df['partner_alpha'].fillna(0).values
        momentum = week_df['score_delta'].fillna(0).values

        # 标准化函数
        def _minmax(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-9) if len(x) > 1 else np.zeros_like(x)

        alpha_norm = _minmax(alpha)
        momentum_norm = _minmax(momentum)

        # --- 2. 获取先验权重配置 ---
        weights = self.p_cfg['factor_anchors']
        w_alpha = weights['partner_alpha_weight']
        w_mom = weights['technical_momentum_weight']

        # 计算‘隐热度’ (Latent Heat)
        latent_heat = (w_alpha * alpha_norm) + (w_mom * momentum_norm)

        # --- 3. 映射到 Zipf 空间 ---
        # 热度越高，排名数字越小 (1st, 2nd...)
        # 使用 -latent_heat 进行排名，最大的热度得到 rank 1
        predicted_ranks = rankdata(-latent_heat, method='min')

        # 特殊赛季（如 Season 15 全明星）使用更平滑的 alpha_allstar
        season = week_df['season'].iloc[0]
        if season == 15:
            zipf_alpha = self.p_cfg['inference']['priors']['alpha_allstar']
        else:
            zipf_alpha = self.p_cfg['vote_distribution']['zipf_params']['alpha']

        # 应用幂律公式
        prior_weights = 1.0 / (predicted_ranks ** zipf_alpha)
        prior_probs = prior_weights / prior_weights.sum()

        return prior_probs

    def flat_prior(self, n: int) -> np.ndarray:
        """
        退化对比基准：均匀分布 (Maximum Entropy Initial State)。
        用于证明 Informative Prior 显著提升了优化器的收敛速度。
        """
        return np.ones(n) / n


# ------------------------------------------------------------------------------
# 单元测试与直觉验证 (Mock Data)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    priors = VotePriors()

    # 模拟第 27 季决赛周数据 (Bobby Bones 夺冠周)
    # 假设 Bobby (Idx 3) 技术分低，但舞伴 Alpha 高，且有黑马动量
    mock_df = pd.DataFrame({
        'celebrity_name': ['Dancer_A', 'Dancer_B', 'Dancer_C', 'Bobby_Bones'],
        'season': [27, 27, 27, 27],
        'week_num': [10, 10, 10, 10],
        'partner_alpha': [0.5, 0.4, 0.6, 0.9],  # Bobby 的舞伴极强 (Sharna Burgess)
        'score_delta': [0.1, 0.0, -0.2, 0.5]  # Bobby 最后一周进步巨大
    })

    # 生成先验
    inf_prior = priors.informative_prior(mock_df)

    print("\n--- Informative Prior (Zipf-based) ---")
    for name, p in zip(mock_df['celebrity_name'], inf_prior):
        print(f"{name}: {p:.2%}")

    entropy = priors.calculate_shannon_entropy(inf_prior)
    print(f"\nPrior Shannon Entropy: {entropy:.3f} bits")