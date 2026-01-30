# ==============================================================================
# src/core/bayes_sampler.py
# Role: Mission Commander - BIO Engine Orchestrator
# Function: Iteratively recovering latent fan votes and uncertainty metrics
# ==============================================================================

import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from src.core.priors import VotePriors
from src.core.optimizer import VoteInferenceOptimizer
from src.core.uncertainty import UncertaintyQuantifier
from src.etl.config_loader import ConfigLoader


class BayesianVoteInference:
    """
    贝叶斯投票反演系统：
    核心逻辑：Posteriors ∝ Likelihood(Outcome | V) * Priors(V)
    """

    def __init__(self, df_gold: pd.DataFrame):
        self.df = df_gold.copy()
        self.cfg = ConfigLoader.load_config()
        self.results_store = []

    def _get_judge_signals(self, week_df: pd.DataFrame, mechanism: str) -> np.ndarray:
        """
        根据赛制提取并预处理评委信号。
        """
        if mechanism == "PERCENT":
            # 百分比制：输入是分数的占比
            scores = week_df['week_avg_score'].values
            return scores / (scores.sum() + 1e-9)
        else:
            # 排名制：输入是技术排名 (1为最好)
            # 注意：ETL中算的是分数，这里需要转为 Rank
            return week_df['week_avg_score'].rank(ascending=False).values

    def run_inference(self):
        """
        全量反演流水线。
        """
        logging.info(">>> 启动贝叶斯反向优化引擎 (BIO-Engine)...")

        # 1. 确定处理序列
        game_points = self.df.groupby(['season', 'week_num']).groups.keys()
        game_points = sorted(list(game_points))

        # 2. 迭代求解
        for season, week in tqdm(game_points, desc="BIO Sampling"):
            # 提取周快照
            week_mask = (self.df['season'] == season) & (self.df['week_num'] == week)
            week_df = self.df[week_mask].reset_index(drop=True)

            if len(week_df) <= 1: continue  # 过滤异常样本

            # 确定赛制
            m_cfg = self.cfg['mechanisms']
            mechanism = "RANK" if season in m_cfg['rank_based_seasons'] else "PERCENT"

            # 3. 寻找被淘汰者 (The 'Censored' Label)
            # 逻辑：找出 final_status 为 Eliminated 且 eliminated_week 等于当前周的人
            elim_mask = (week_df['final_status'] == 'Eliminated') & (week_df['eliminated_week'] == week)
            elim_indices = week_df.index[elim_mask].tolist()
            elim_idx = elim_indices[0] if elim_indices else None

            # 4. 构造先验与评委信号
            judge_signals = self._get_judge_signals(week_df, mechanism)
            prior_v = VotePriors.informative_prior(week_df)

            # 5. 执行优化求解 (MAP Estimation)
            optimizer = VoteInferenceOptimizer(season)
            if elim_idx is not None:
                v_opt, success = optimizer.solve_week(judge_signals, elim_idx, prior_v=prior_v)
            else:
                # 非淘汰周：结局未提供额外信息，解退化为先验分布
                v_opt = prior_v
                success = True

            # 6. 不确定性量化
            entropy = UncertaintyQuantifier.calculate_shannon_entropy(v_opt)
            certainty = UncertaintyQuantifier.calculate_relative_certainty(entropy, len(week_df))

            # 7. 存储结果
            for i, row in week_df.iterrows():
                self.results_store.append({
                    'celebrity_name': row['celebrity_name'],
                    'season': season,
                    'week_num': week,
                    'est_fan_vote_pct': v_opt[i],
                    'est_fan_vote_rank': pd.Series(-v_opt).rank().values[i],
                    'inference_entropy': entropy,
                    'est_certainty_score': certainty,
                    'solver_converged': success
                })

        return self._post_process()

    def _post_process(self) -> pd.DataFrame:
        """合并结果并生成‘铂金层’数据"""
        res_df = pd.DataFrame(self.results_store)

        # 与 Gold 层因子库对齐
        platinum_df = self.df.merge(
            res_df,
            on=['celebrity_name', 'season', 'week_num'],
            how='left'
        )

        # 统计审计
        convergence_rate = res_df['solver_converged'].mean()
        logging.info(f"BIO 反演结束。优化器收敛率: {convergence_rate:.2%}")

        return platinum_df