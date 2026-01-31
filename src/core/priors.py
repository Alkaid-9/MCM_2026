"""
MCM 2026 Problem C: Bayesian Prior Architect (Industrial Refactor v4.1)
Role: Transforming ETL features into Informative Priors (Zipf's Law).
Function: Defensive parameter mapping and heuristic heat calculation.
Standard: Academic Rigor / Production-Grade Robustness.
"""

import numpy as np
import pandas as pd
import logging
from scipy.stats import rankdata
from src.etl.config_loader import ConfigLoader


class VotePriors:
    """
    贝叶斯先验生成器：
    利用“社会网络效应”建立初始分布。我们假设观众投票遵循优先连接过程（Preferential Attachment），
    在宏观上表现为齐夫定律（Zipf's Law）的幂律特征。
    """

    def __init__(self):
        self.cfg = ConfigLoader()
        # 这里的 p_cfg 是从 priors.yaml 加载的字典
        self.p_cfg = self.cfg.get_priors_config()
        self.logger = logging.getLogger("PRIOR_ARCHITECT")

    def _get_zipf_alpha(self, season: int) -> float:
        """
        【防御性寻址】获取幂律指数 Alpha。
        物理意义：Alpha 越大，流量越向顶层明星集中。全明星赛（S15）通常分布更扁平。
        """
        # 1. 提取全局标准 Alpha (默认 1.2)
        dist_cfg = self.p_cfg.get('vote_distribution', {})
        standard_alpha = dist_cfg.get('zipf_params', {}).get('alpha', 1.2)

        # 2. 处理全明星特殊逻辑 (Season 15)
        if season == 15:
            # 使用链式 .get() 彻底杜绝 KeyError
            # 逻辑：先找 inference.priors.alpha_allstar -> 找不到则取标准值的 80%
            return self.p_cfg.get('inference', {}).get('priors', {}).get('alpha_allstar', standard_alpha * 0.8)

        return standard_alpha

    def informative_prior(self, week_df: pd.DataFrame) -> np.ndarray:
        """
        【核心学术亮点】融合因子库构建启发式先验。
        物理逻辑：
        1. 提取舞伴溢价 (Partner Alpha) 和 表现增量 (Score Delta)。
        2. 线性合成“潜在大众热度”。
        3. 将热度映射到概率单纯形（Probability Simplex）上的幂律分布。
        """
        n = len(week_df)
        if n == 0:
            return np.array([])

        season = int(week_df['season'].iloc[0]) if 'season' in week_df.columns else 0

        # --- 1. 因子提取与防御性清洗 ---
        # 即使上游漏掉了因子，我们也给个中性默认值，确保程序不崩
        alpha = week_df['partner_alpha'].fillna(1.0).values
        momentum = week_df['score_delta'].fillna(0.0).values

        # 向量化归一化函数
        def _minmax(x):
            denom = x.max() - x.min()
            return (x - x.min()) / (denom + 1e-9) if denom > 1e-6 else np.zeros_like(x)

        alpha_norm = _minmax(alpha)
        momentum_norm = _minmax(momentum)

        # --- 2. 获取先验融合权重 (从配置文件读取) ---
        anchors = self.p_cfg.get('factor_anchors', {})
        w_alpha = anchors.get('partner_alpha_weight', 0.45)
        w_mom = anchors.get('technical_momentum_weight', 0.25)

        # 计算“隐热度”指数
        latent_heat = (w_alpha * alpha_norm) + (w_mom * momentum_norm)

        # --- 3. 映射到 Zipf 空间 ---
        # 物理直觉：热度越高的明星（Rank=1），其先验概率按幂律下降
        # 使用 rankdata 计算竞争性排名，处理并列情况
        predicted_ranks = rankdata(-latent_heat, method='min')

        # 获取当前赛季应使用的 Alpha
        zipf_alpha = self._get_zipf_alpha(season)

        # 应用齐夫定律公式: P(r) \propto 1 / r^alpha
        prior_weights = 1.0 / (np.power(predicted_ranks, zipf_alpha))

        # 归一化：将权重投影回单纯形空间 (Sum=1.0)
        prior_probs = prior_weights / (prior_weights.sum() + 1e-12)

        return prior_probs

    @staticmethod
    def calculate_shannon_entropy(v: np.ndarray) -> float:
        """计算信息熵 (Shannon Entropy)，用于度量先验的不确定性。"""
        v_safe = np.clip(v, 1e-12, 1.0)
        return -np.sum(v_safe * np.log2(v_safe))

    def flat_prior(self, n: int) -> np.ndarray:
        """无信息先验 (Uniform Prior)：作为收敛速度对比的基准。"""
        return np.ones(n) / n