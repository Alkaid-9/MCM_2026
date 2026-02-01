"""
Mechanism Simulator & Counterfactual Engine (v4.6)
Role: Re-running DWTS history under alternative "Parallel Universes".
Functions:
    - Simulating Rank-based outcomes for Percent seasons.
    - Simulating Percent-based outcomes for Rank seasons.
    - Calculating 'Rank Volatility' across different rules.
Standard: Policy Impact Analysis / Robustness Testing.
"""

import pandas as pd
import numpy as np
from scipy.stats import rankdata
import logging


class MechanismSimulator:
    def __init__(self, df_platinum: pd.DataFrame):
        self.logger = logging.getLogger("MECHANISM_COMPARE")
        self.df = df_platinum.copy()

    def _calculate_rank_sum_outcome(self, judge_scores, fan_shares):
        """
        [规则 A]: 排名结合法 (Ordinal Aggregation)
        物理意义：抑制极值，对齐序数。
        """
        # 评委排名 (分数高则排名小, 1为最好)
        j_rank = rankdata(-judge_scores, method='min')
        # 粉丝排名 (票数高则排名小)
        f_rank = rankdata(-fan_shares, method='min')

        # 总排名和：越小越好
        total_rank_sum = j_rank + f_rank
        # 返回最终顺位 (1 为第一，N 为最后)
        return rankdata(total_rank_sum, method='min')

    def _calculate_percent_sum_outcome(self, judge_scores, fan_shares):
        """
        [规则 B]: 百分比结合法 (Cardinal Aggregation)
        物理意义：信号放大，保留绝对差距。
        """
        # 评委分比例化
        j_pct = judge_scores / (judge_scores.sum() + 1e-9)
        # 粉丝票比例化 (反演结果已经是占比)
        f_pct = fan_shares

        # 总分：越大越好
        total_score = j_pct + f_pct
        # 返回最终顺位 (1 为第一，N 为最后)
        return rankdata(-total_score, method='min')

    def run_comparison_pipeline(self) -> pd.DataFrame:
        """
        执行全量“平行宇宙”模拟。
        """
        self.logger.info("正在执行机制鲁棒性模拟 (Parallel Universe Simulation)...")

        results = []
        groups = self.df.groupby(['season', 'week_num'])

        for (s, w), group in groups:
            group = group.reset_index(drop=True)
            j_scores = group['week_avg_score'].values
            f_shares = group['est_fan_vote_mu'].values

            # 1. 在当前宇宙模拟两种规则
            rank_universe_placements = self._calculate_rank_sum_outcome(j_scores, f_shares)
            percent_universe_placements = self._calculate_percent_sum_outcome(j_scores, f_shares)

            # 2. 识别实际淘汰者
            elim_mask = (group['final_status'] == 'Eliminated') & (group['eliminated_week'] == w)
            actual_loser_idx = np.where(elim_mask)[0]

            for i in range(len(group)):
                results.append({
                    'season': s,
                    'week_num': w,
                    'celebrity_name': group.loc[i, 'celebrity_name'],
                    'actual_judges_score': j_scores[i],
                    'inferred_fan_share': f_shares[i],
                    'rank_system_pos': rank_universe_placements[i],
                    'percent_system_pos': percent_universe_placements[i],
                    # 规则敏感度：如果换了规则，排名变动了多少？
                    'rule_sensitivity': abs(rank_universe_placements[i] - percent_universe_placements[i]),
                    'would_be_eliminated_under_rank': rank_universe_placements[i] == max(rank_universe_placements),
                    'would_be_eliminated_under_percent': percent_universe_placements[i] == max(
                        percent_universe_placements)
                })

        return pd.DataFrame(results)

    def analyze_systemic_bias(self, df_sim: pd.DataFrame):
        """
        量化分析：哪种规则更倾向于观众？
        """
        # 计算每种规则与评委评分的相关性 (Spearman)
        # 评委排名 vs 机制最终排名
        rank_corr = df_sim[['actual_judges_score', 'rank_system_pos']].corr(method='spearman').iloc[0, 1]
        pct_corr = df_sim[['actual_judges_score', 'percent_system_pos']].corr(method='spearman').iloc[0, 1]

        self.logger.info(f">>> 机制偏见审计报告:")
        self.logger.info(f"Rank 规则与评委分相关性: {abs(rank_corr):.4f}")
        self.logger.info(f"Percent 规则与评委分相关性: {abs(pct_corr):.4f}")

        if abs(rank_corr) > abs(pct_corr):
            self.logger.info("结论：Rank 规则更尊重技术（评委分），更具 Meritocracy。")
        else:
            self.logger.info("结论：Percent 规则更容易被选票左右。")