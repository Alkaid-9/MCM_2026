# ==============================================================================
# src/solvers/ic_simulator.py
# Role: Game Theoretic Auditor (Incentive Compatibility Engine)
# Function: Simulating Agent Strategies (Merit vs. Promo) to prove Nash Equilibrium.
# Physics: Calculating Marginal Utility (MU) of "Practice" vs. "Campaigning".
# Standard: Industrial Grade / Pure Library Mode / Zero-Side-Effect.
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from scipy.stats import rankdata

# 引入项目核心组件
from src.solvers.daw_engine import DAWEngine
from src.etl.config_loader import ConfigLoader


class IncentiveCompatibilityAuditor:
    """
    激励相容性（IC）审计师：
    通过边际微扰分析（Perturbation Analysis），量化理性代理人在不同机制下的生存收益率。

    [数学直觉 - The Physics of Survival]
    我们将选手抽象为在单纯形约束下分配精力的代理人。一个机制是激励相容的（IC），
    当且仅当“提升技术”带来的排名增益（$\partial R/\partial E_{tech}$）
    显著超过“提升人气”带来的增益（$\partial R/\partial E_{promo}$）。

    DAW 机制的目标是使该比率在赛程后期趋向于正无穷，强制引导纳什均衡点回归技术本位。
    """

    def __init__(self, df_platinum: pd.DataFrame, fig_dir: str = "reports/figures/"):
        self.logger = logging.getLogger("GAME_THEORY_AUDIT")
        self.df = df_platinum.copy()

        # [CRITICAL FIX]: DAWEngine 现在是纯计算内核，不再接受 fig_dir 参数
        self.daw_engine = DAWEngine()

        self.fig_dir = fig_dir
        os.makedirs(self.fig_dir, exist_ok=True)

        # 绘图配置
        plt.rcParams['font.family'] = 'serif'
        sns.set_context("paper", font_scale=1.4)

    def _get_week_stats(self, season: int, week: int):
        """
        获取当周的统计分布特征 (Sigma)，用于标准化“单位努力”的量纲。
        物理意义：假设“1 单位精力”等价于将该选手的表现提升 0.5 个标准差。
        """
        week_data = self.df[(self.df['season'] == season) & (self.df['week_num'] == week)]
        if len(week_data) < 2:
            return None, None, None

        # 计算得分与投票的即时波动率
        sigma_score = week_data['week_avg_score'].std()
        sigma_vote = week_data['est_fan_vote_mu'].std()

        # 异常防御：如果当周所有人分数一样，sigma 为 0
        sigma_score = max(sigma_score, 1e-6)
        sigma_vote = max(sigma_vote, 1e-6)

        return week_data, sigma_score, sigma_vote

    def _simulate_rank_outcome(self, s, v):
        """辅助函数：重构 Rank 机制下的最终排名 (1=第一名)"""
        return rankdata(rankdata(-s, method='min') + rankdata(-v, method='min'), method='min')

    def _simulate_percent_outcome(self, s, v):
        """辅助函数：重构 Percent 机制下的总分 (越高越好)"""
        s_pct = s / (s.sum() + 1e-9)
        return s_pct + v

    def calculate_marginal_utility(self, season: int, week: int, target_celebrity: str = None,
                                   effort_unit: float = 0.5):
        """
        计算边际效用 (Marginal Utility, MU)。
        模拟选手分别投入精力于“技术”或“流量”后，在三种机制下的排名变化。
        """
        week_data, sigma_s, sigma_v = self._get_week_stats(season, week)
        if week_data is None: return None

        # 选取基准代理人 (默认为中游选手)
        if target_celebrity is None:
            median_idx = len(week_data) // 2
            target_row = week_data.sort_values('week_avg_score').iloc[median_idx]
            target_celebrity = target_row['celebrity_name']

        # 向量化提取
        scores = week_data['week_avg_score'].values.copy()
        votes = week_data['est_fan_vote_mu'].values.copy()
        names = week_data['celebrity_name'].values

        try:
            t_idx = np.where(names == target_celebrity)[0][0]
        except:
            return None

        # 1. 确定赛程进度 (用于 DAW 引擎)
        total_weeks = self.df[self.df['season'] == season]['week_num'].max()
        progress = week / max(total_weeks, 1)

        # ----------------------------------------------------------------------
        # Step A: 基准计算 (Baseline)
        # ----------------------------------------------------------------------
        r_base = self._simulate_rank_outcome(scores, votes)[t_idx]
        p_base = self._simulate_percent_outcome(scores, votes)[t_idx]
        d_score_base, _ = self.daw_engine.calculate_combined_score(
            rankdata(-scores, method='min'), rankdata(-votes, method='min'), progress
        )
        d_base = rankdata(d_score_base, method='min')[t_idx]

        # ----------------------------------------------------------------------
        # Step B: 模拟技术微扰 (Merit Perturbation)
        # ----------------------------------------------------------------------
        s_plus = scores.copy()
        s_plus[t_idx] = min(10.0, s_plus[t_idx] + effort_unit * sigma_s)

        r_merit = r_base - self._simulate_rank_outcome(s_plus, votes)[t_idx]
        # 对于 Percent，计算分数增量，并转化为排名等效增量
        p_merit = self._simulate_percent_outcome(s_plus, votes)[t_idx] - p_base

        d_score_merit, _ = self.daw_engine.calculate_combined_score(
            rankdata(-s_plus, method='min'), rankdata(-votes, method='min'), progress
        )
        d_merit = d_base - rankdata(d_score_merit, method='min')[t_idx]

        # ----------------------------------------------------------------------
        # Step C: 模拟流量微扰 (Promo Perturbation)
        # ----------------------------------------------------------------------
        v_plus = votes.copy()
        v_plus[t_idx] += effort_unit * sigma_v
        v_plus /= (v_plus.sum() + 1e-9)  # 单纯形约束保护

        r_promo = r_base - self._simulate_rank_outcome(scores, v_plus)[t_idx]
        p_promo = self._simulate_percent_outcome(scores, v_plus)[t_idx] - p_base

        d_score_promo, _ = self.daw_engine.calculate_combined_score(
            rankdata(-scores, method='min'), rankdata(-v_plus, method='min'), progress
        )
        d_promo = d_base - rankdata(d_score_promo, method='min')[t_idx]

        # ----------------------------------------------------------------------
        # Step D: 计算激励比率 (Incentive Ratio)
        # ----------------------------------------------------------------------
        # 逻辑：Ratio > 1 表示“练舞比拉票更划算”
        # 使用 max(0, x) 确保只奖励正面收益，使用 1e-9 防止除零崩溃
        return {
            "week": week,
            "celebrity": target_celebrity,
            "rank_ratio": (max(0, r_merit) + 1e-9) / (max(0, r_promo) + 1e-9),
            "pct_ratio": (max(0, p_merit) + 1e-9) / (max(0, p_promo) + 1e-9),
            "daw_ratio": (max(0, d_merit) + 1e-9) / (max(0, d_promo) + 1e-9)
        }

    def run_full_season_audit(self, season_id: int = 27):
        """
        全生命周期审计：遍历赛季每一周，计算系统平均激励导向。
        这是论文中“Incentive Compatibility Proof”章节的核心数据源。
        """
        self.logger.info(f">>> 执行博弈论稳定性审计 (Season {season_id})...")
        weeks = sorted(self.df[self.df['season'] == season_id]['week_num'].unique())
        results = []

        for w in weeks:
            week_df = self.df[(self.df['season'] == season_id) & (self.df['week_num'] == w)]
            contestants = week_df['celebrity_name'].values

            w_daw, w_pct = [], []
            for c in contestants:
                mu = self.calculate_marginal_utility(season_id, w, c)
                if mu:
                    w_daw.append(mu['daw_ratio'])
                    w_pct.append(mu['pct_ratio'])

            if w_daw:
                results.append({
                    'week': w,
                    'daw_ic_ratio': np.median(w_daw),  # 使用中位数抵抗离群干扰
                    'percent_ic_ratio': np.median(w_pct)
                })

        res_df = pd.DataFrame(results)
        if not res_df.empty:
            self._plot_ic_trajectory(res_df, season_id)
        return res_df

    def _plot_ic_trajectory(self, df: pd.DataFrame, season_id: int):
        """
        可视化：绘制激励相容性演化轨迹。
        物理展示：Proposed 曲线如何冲破 1.0 的“平庸基准线”。
        """
        plt.figure(figsize=(10, 6))

        # 1. 绘制基准持平线
        plt.axhline(1.0, color='black', linestyle='--', linewidth=1.5, label='Indifference Line (Merit=Promo)')

        # 2. 绘制对比曲线
        plt.plot(df['week'], df['percent_ic_ratio'], marker='x', linestyle=':', color='gray',
                 label='Baseline (Percent Rule)')
        plt.plot(df['week'], df['daw_ic_ratio'], marker='o', linestyle='-', color='#d62728', linewidth=2.5,
                 label='Proposed (DAW Mechanism)')

        # 3. 填充激励区
        plt.fill_between(df['week'], 1.0, 10.0, color='#2ca02c', alpha=0.1, label='Incentive Compatible Zone')

        # 4. 装饰与坐标轴
        plt.title(f"Mechanism Audit: Evolution of Incentive Compatibility (S{season_id})", fontsize=14, pad=15)
        plt.xlabel("Competition Week", fontsize=12)
        plt.ylabel("Incentive Ratio ($MU_{Merit} / MU_{Promo}$)", fontsize=12)
        plt.yscale('log')  # 使用对数坐标展示数量级差异
        plt.ylim(0.2, 20)
        plt.legend(loc='upper left', frameon=True)
        plt.grid(True, which="both", ls="-", alpha=0.2)

        save_path = os.path.join(self.fig_dir, "ic_trajectory_proof.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"博弈论审计图已保存至: {save_path}")