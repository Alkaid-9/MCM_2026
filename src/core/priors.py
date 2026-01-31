"""
MCM 2026 Problem C: Bayesian Prior Architect (Industrial Refactor v4.5)
Role: Transforming ETL features into Informative Priors based on Zipf's Law.
Function: Building the "Popularity Gravity Field" to anchor MCMC sampling.
Standard: Academic Rigor / Bayesian Regularization / Social Network Theory.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from src.etl.config_loader import ConfigLoader

class VotePriors:
    """
    贝叶斯先验生成器：
    利用“社会网络择优连接效应”建立初始分布。我们假设观众投票遵循幂律分布，
    在宏观上表现为齐夫定律（Zipf's Law）的特征。
    """
    def __init__(self):
        self.cfg_loader = ConfigLoader()
        # 加载 rules.yaml 中的因子权重和行业偏置
        self.rules_cfg = self.cfg_loader._config
        # 加载 priors.yaml 中的分布形态参数
        self.priors_cfg = self.cfg_loader.get_priors_config()

    def _get_zipf_alpha(self, season: int) -> float:
        """
        [防御性寻址] 获取当前赛季的幂律指数 Alpha。
        物理意义：Alpha 越大，流量越向头部集中；全明星赛（S15）通常竞争均衡，分布扁平。
        """
        # 优先级：季节特定覆盖 > 默认值
        overrides = self.priors_cfg.get('season_overrides', {})
        if season in overrides:
            return overrides[season].get('alpha', 1.2)

        dist_cfg = self.priors_cfg.get('vote_distribution', {}).get('zipf_params', {})
        if season == 15:
            return dist_cfg.get('alpha_allstar', 0.8)
        return dist_cfg.get('alpha_standard', 1.2)

    def informative_prior(self, week_df: pd.DataFrame) -> np.ndarray:
        """
        【核心学术亮点】：融合多源异构因子构建贝叶斯先验场。

        逻辑链：
        1. 提取静态行业偏置 (Industry Bias)
        2. 提取舞伴历史溢价 (Partner Alpha)
        3. 提取动态表现反馈 (Score Momentum)
        4. 线性合成“人气引力指数” -> 排序 -> Zipf 映射 -> 概率单纯形
        """
        n = len(week_df)
        if n == 0: return np.array([])

        # --- 1. 因子提取与防御性清洗 ---
        # 行业原生流量 (Static Bias)
        bias_map = self.rules_cfg.get('factor_anchors', {}).get('industry_base_bias', {})
        # 物理意义：电视真人秀明星天然比政客更容易吸粉
        ind_bias = week_df['celebrity_industry'].map(bias_map).fillna(0.0).values

        # 舞伴溢价 (Partner Alpha) - 体现“老带新”能力
        partner_alpha = week_df['partner_alpha'].fillna(1.0).values

        # 表现动量 (Technical Momentum) - 体现“成长剧本”对粉丝的刺激
        momentum = week_df['score_delta'].fillna(0.0).values

        # --- 2. 鲁棒特征缩放 (Robust Scaling) ---
        def _scale(x):
            # 将不同量纲因子映射到 [0, 1]，增加 1e-9 防止除零
            span = x.max() - x.min()
            return (x - x.min()) / (span + 1e-9) if span > 1e-7 else np.zeros_like(x)

        # --- 3. 计算综合人气引力 (Popularity Gravity) ---
        anchors = self.rules_cfg.get('factor_anchors', {})
        w_partner = anchors.get('partner_alpha_weight', 0.45)
        w_momentum = anchors.get('technical_momentum_weight', 0.25)

        # 物理直觉：Gravity = 行业原生引力 + 舞伴加成引力 + 表现反馈引力
        latent_gravity = ind_bias + (w_partner * _scale(partner_alpha)) + (w_momentum * _scale(momentum))

        # --- 4. 映射至 Zipf 空间 (The Power-Law Bridge) ---
        # 使用 rankdata 计算竞争性排名，处理并列情况。
        # 物理直觉：Gravity 指数最高的选手排名第一（Rank=1）。
        predicted_ranks = rankdata(-latent_gravity, method='min')

        # 获取当前赛季的衰减参数
        season = int(week_df['season'].iloc[0]) if 'season' in week_df.columns else 1
        zipf_alpha = self._get_zipf_alpha(season)

        # 应用齐夫定律公式: P(r) \propto 1 / r^alpha
        # 物理意义：人气排名第 1 的选手拥有的选票潜力远超第 2 名。
        prior_weights = 1.0 / (np.power(predicted_ranks, zipf_alpha))

        # --- 5. 归一化至概率单纯形 (Normalization to Simplex) ---
        # 增加 EPSILON 防止数值下溢，确保后续 C++ 对数似然计算安全
        eps = self.priors_cfg.get('vote_distribution', {}).get('zipf_params', {}).get('min_vote_share', 0.001)
        prior_probs = (prior_weights + eps) / (prior_weights.sum() + n * eps)

        return prior_probs

    @staticmethod
    def calculate_prior_entropy(v_prior: np.ndarray) -> float:
        """
        计算先验分布的香农熵。
        物理意义：衡量当前周选手的“人气均质化”程度。熵越高，代表竞争越胶着。
        """
        v_safe = np.clip(v_prior, 1e-12, 1.0)
        return -np.sum(v_safe * np.log2(v_safe))

    def flat_prior(self, n: int) -> np.ndarray:
        """无信息先验 (Uniform Prior)：作为收敛速度对比的基准线。"""
        return np.ones(n) / n