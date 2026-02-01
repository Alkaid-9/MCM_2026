# ==============================================================================
# src/core/priors.py
# Role: Bayesian Prior Architect (v5.1 - Interface Sync)
# Fix: Replaced private _config access with load_config().
# ==============================================================================

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from src.etl.config_loader import ConfigLoader


class VotePriors:
    def __init__(self):
        self.cfg_loader = ConfigLoader()
        # [修复] 此时 ConfigLoader 已有 load_config 方法
        self.rules_cfg = self.cfg_loader.load_config()
        self.priors_cfg = self.cfg_loader.get_priors_config()

    def _get_zipf_alpha(self, season: int) -> float:
        overrides = self.priors_cfg.get('season_overrides', {})
        s_key = season if season in overrides else str(season)
        if s_key in overrides:
            return float(overrides[s_key].get('alpha', 1.2))
        return float(self.priors_cfg.get('vote_distribution', {}).get('zipf_params', {}).get('alpha_standard', 1.2))

    def informative_prior(self, week_df: pd.DataFrame) -> np.ndarray:
        n = len(week_df)
        if n == 0: return np.array([])

        # 1. 行业偏置
        bias_map = self.rules_cfg.get('factor_anchors', {}).get('industry_base_bias', {})
        ind_bias = week_df['celebrity_industry'].map(bias_map).fillna(0.0).values

        # 2. 舞伴与动量
        p_alpha = week_df['partner_alpha'].fillna(1.0).values
        momentum = week_df['score_delta'].fillna(0.0).values if 'score_delta' in week_df else np.zeros(n)

        # 3. 归一化与合成
        def _scale(x):
            s = x.max() - x.min()
            return (x - x.min()) / (s + 1e-9) if s > 1e-7 else np.zeros_like(x)

        anchors = self.rules_cfg.get('factor_anchors', {})
        w_p = float(anchors.get('partner_alpha_weight', 0.45))
        w_m = float(anchors.get('technical_momentum_weight', 0.25))

        gravity = ind_bias + w_p * _scale(p_alpha) + w_m * _scale(momentum)

        # 4. Zipf 映射
        ranks = rankdata(-gravity, method='min')
        alpha = self._get_zipf_alpha(int(week_df['season'].iloc[0]))
        weights = 1.0 / np.power(ranks, alpha)

        # 5. 单纯形归一化
        return weights / weights.sum()
    def flat_prior(self, n: int) -> np.ndarray:
        """无信息先验 (Uniform Baseline)：用于 Section 7.1 的鲁棒性对比"""
        return np.ones(n) / n

    @staticmethod
    def calculate_prior_entropy(v_prior: np.ndarray) -> float:
        """
        计算先验分布的香农熵 (H)。
        物理意义：量化“在看到当周表现前，民意的混乱程度”。
        """
        v_safe = np.clip(v_prior, 1e-12, 1.0)
        return -np.sum(v_safe * np.log2(v_safe))