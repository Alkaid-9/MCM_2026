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
from src.simulators.mechanism_sandbox import run_monte_carlo_survival
from src.etl.config_loader import ConfigLoader


class IncentiveCompatibilityAuditor:
    """
    激励相容性（IC）审计师：
    通过数值仿真，量化理性代理人在不同机制下的生存收益率。

    [数学直觉 - The Physics of Survival]
    我们将选手抽象为在单纯形约束下分配精力的代理人。一个机制是激励相容的（IC），
    当且仅当“提升技术”带来的排名增益（∂P/∂E_tech）
    显著超过“提升人气”带来的增益（∂P/∂E_promo）。

    DAW 机制的目标是使该比率在赛程后期趋向于正无穷，强制引导纳什均衡点回归技术本位。
    """

    def __init__(self, df_platinum: pd.DataFrame, fig_dir: str = "reports/figures/"):
        self.logger = logging.getLogger("GAME_THEORY_AUDIT")
        self.df = df_platinum.copy()
        self.daw_engine = DAWEngine()
        self.fig_dir = fig_dir
        os.makedirs(self.fig_dir, exist_ok=True)

        # 绘图配置
        try:
            plt.rcParams['font.family'] = 'serif'
            sns.set_context("paper", font_scale=1.4)
        except:
            pass

    def _get_week_stats(self, season: int, week: int):
        """
        [关键修复]：确保无论何种情况，均返回且仅返回 3 个值。
        """
        week_data = self.df[(self.df['season'] == season) & (self.df['week_num'] == week)]
        if len(week_data) < 2:
            return None, 0.0, 0.0  # 严格返回 3 个元素

        # 计算得分与投票的即时波动率
        sigma_score = week_data['week_avg_score'].std()
        sigma_vote = week_data['est_fan_vote_mu'].std()

        # 防御性处理
        sigma_score = max(sigma_score, 1e-6)
        sigma_vote = max(sigma_vote, 1e-6)

        return week_data, sigma_score, sigma_vote

    def calculate_marginal_utility(self, season: int, week: int, target_celebrity: str = None,
                                   effort_unit: float = 0.5):
        """计算边际效用 (MU)"""
        # [调用对齐]
        week_data, sigma_s, sigma_v = self._get_week_stats(season, week)

        if week_data is None: return None

        # 选取基准代理人
        if target_celebrity is None:
            median_idx = len(week_data) // 2
            target_celebrity = week_data.sort_values('week_avg_score').iloc[median_idx]['celebrity_name']

        try:
            t_idx = np.where(week_data['celebrity_name'].values == target_celebrity)[0][0]
        except:
            return None

        j_scores = week_data['week_avg_score'].values.astype(np.float64)
        f_votes_mu = week_data['est_fan_vote_mu'].values.astype(np.float64)
        total_weeks = self.df[self.df['season'] == season]['week_num'].max()

        # 1. 模拟“提升技术” (Merit)
        j_boost = j_scores.copy()
        j_boost[t_idx] = min(10.0, j_boost[t_idx] + effort_unit * sigma_s)

        # 2. 模拟“提升营销” (Promo)
        f_boost = f_votes_mu.copy()
        f_boost[t_idx] += effort_unit * sigma_v
        f_boost /= f_boost.sum()

        # 3. 计算生存概率增量 (Using 0.05 jitter)
        sim_sigma = 0.05
        # 基准概率
        p_base_pct = run_monte_carlo_survival(j_scores, f_votes_mu, sim_sigma, n_sims=500, mech_type=0)
        p_base_rank = run_monte_carlo_survival(j_scores, f_votes_mu, sim_sigma, n_sims=500, mech_type=1)

        # 技术投入收益
        p_merit_pct = run_monte_carlo_survival(j_boost, f_votes_mu, sim_sigma, n_sims=500, mech_type=0)
        p_merit_rank = run_monte_carlo_survival(j_boost, f_votes_mu, sim_sigma, n_sims=500, mech_type=1)

        # 营销投入收益
        p_promo_pct = run_monte_carlo_survival(j_scores, f_boost, sim_sigma, n_sims=500, mech_type=0)
        p_promo_rank = run_monte_carlo_survival(j_scores, f_boost, sim_sigma, n_sims=500, mech_type=1)

        # 4. 计算 DAW 混合收益 (基于当前权重)
        w_j = self.daw_engine.compute_judge_weight(week, total_weeks)

        mu_merit_daw = w_j * (p_merit_rank[t_idx] - p_base_rank[t_idx]) + (1 - w_j) * (
                    p_merit_pct[t_idx] - p_base_pct[t_idx])
        mu_promo_daw = w_j * (p_promo_rank[t_idx] - p_base_rank[t_idx]) + (1 - w_j) * (
                    p_promo_pct[t_idx] - p_base_pct[t_idx])

        # 计算历史 Percent 机制收益比作为对照
        mu_merit_pct = p_merit_pct[t_idx] - p_base_pct[t_idx]
        mu_promo_pct = p_promo_pct[t_idx] - p_base_pct[t_idx]

        return {
            "daw_ratio": (mu_merit_daw + 1e-6) / (mu_promo_daw + 1e-6),
            "pct_ratio": (mu_merit_pct + 1e-6) / (mu_promo_pct + 1e-6)
        }

    def run_full_season_audit(self, season_id: int = 27):
        """执行全赛季博弈论演化审计"""
        self.logger.info(f">>> 正在执行博弈论稳定性审计 (Season {season_id})...")
        weeks = sorted(self.df[self.df['season'] == season_id]['week_num'].unique())
        results = []
        for w in weeks:
            week_df = self.df[(self.df['season'] == season_id) & (self.df['week_num'] == w)]
            # 为了速度，每两周抽样一次或限制选手数量
            c = week_df.iloc[len(week_df) // 2]['celebrity_name']  # 选中间人
            mu = self.calculate_marginal_utility(season_id, w, c)
            if mu:
                results.append({'week': w, 'daw_ic_ratio': mu['daw_ratio'], 'percent_ic_ratio': mu['pct_ratio']})

        res_df = pd.DataFrame(results)
        self._plot_ic_trajectory(res_df, season_id)
        return res_df

    def _plot_ic_trajectory(self, df: pd.DataFrame, season_id: int):
        """绘制激励相容轨迹图"""
        plt.figure(figsize=(10, 6))
        plt.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Indifference (Merit=Promo)')
        plt.plot(df['week'], df['percent_ic_ratio'], marker='x', color='gray', label='Baseline (Percent)')
        plt.plot(df['week'], df['daw_ic_ratio'], marker='o', color='#d62728', lw=2.5, label='Proposed (DAW)')
        plt.fill_between(df['week'], 1.0, 5.0, color='green', alpha=0.1, label='Incentive Compatible Zone')
        plt.title(f"Game Theory Audit: Incentive Ratio Evolution (S{season_id})")
        plt.xlabel("Competition Week")
        plt.ylabel("Incentive Ratio ($MU_{Merit} / MU_{Promo}$)")
        plt.yscale('log')
        plt.legend()
        plt.savefig(os.path.join(self.fig_dir, "ic_trajectory_proof.png"), dpi=300)
        plt.close()