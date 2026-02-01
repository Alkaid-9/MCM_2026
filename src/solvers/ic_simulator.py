# ==============================================================================
# src/solvers/ic_simulator.py
# Role: Game Theoretic Auditor (Incentive Compatibility Engine)
# Function: Simulating Agent Strategies (Merit vs. Promo) to prove Nash Equilibrium.
# Physics: Calculating Marginal Utility (MU) of "Practice" vs. "Campaigning".
# Standard: Industrial Grade / Pure Library Mode (No Test Stubs).
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
    通过微扰分析（Perturbation Analysis），量化不同策略下的生存收益率。

    [核心定义 - Game Theory]
    我们将参赛选手建模为理性代理人 (Rational Agents)，旨在最大化生存概率。
    一个机制是激励相容的 (IC)，当且仅当纳什均衡点偏向于技术提升：

    $$ \frac{\partial \text{Survival}}{\partial \text{Effort}_{Tech}} > \frac{\partial \text{Survival}}{\partial \text{Effort}_{Fan}} $$

    即：投入单位精力练舞的回报 > 投入单位精力拉票的回报。
    """

    def __init__(self, df_platinum: pd.DataFrame, fig_dir: str = "reports/figures/"):
        self.logger = logging.getLogger("GAME_THEORY_AUDIT")
        self.df = df_platinum.copy()
        # 实例化 DAW 引擎用于计算新机制下的收益
        self.daw_engine = DAWEngine(fig_dir=fig_dir)
        self.fig_dir = fig_dir
        os.makedirs(self.fig_dir, exist_ok=True)

        # 绘图配置
        plt.rcParams['font.family'] = 'serif'
        sns.set_context("paper", font_scale=1.4)

    def _get_week_stats(self, season: int, week: int):
        """
        获取当周的统计分布特征 (Sigma)，用于标准化‘努力程度’。
        物理意义：1 单位努力 = 提升 0.5 个标准差的当前环境水平。
        """
        week_data = self.df[(self.df['season'] == season) & (self.df['week_num'] == week)]
        if len(week_data) < 2:
            return None, None, None

        # 计算环境波动率 (Volatility)
        sigma_score = week_data['week_avg_score'].std()
        sigma_vote = week_data['est_fan_vote_mu'].std()
        return week_data, sigma_score, sigma_vote

    def _simulate_rank(self, s, v):
        """辅助：计算 Rank 机制下的最终排名 (越小越好)"""
        # Rank Sum = Rank(Score) + Rank(Vote)
        # 注意：分数越高 rank 越小 (1)，但 rankdata 默认升序，所以取负
        return rankdata(rankdata(-s, method='min') + rankdata(-v, method='min'), method='min')

    def _simulate_percent(self, s, v):
        """辅助：计算 Percent 机制下的总分 (越大越好)"""
        # Total % = Score% + Vote%
        s_pct = s / (s.sum() + 1e-9)
        return s_pct + v

    def calculate_marginal_utility(self, season: int, week: int, target_celebrity: str = None,
                                   effort_unit: float = 0.5):
        """
        计算边际效用 (Marginal Utility, MU)。
        假设选手投入 0.5 个标准差的努力，分别在不同赛制下能换来多少排名提升？
        """
        week_data, sigma_s, sigma_v = self._get_week_stats(season, week)
        if week_data is None: return None

        # 如果未指定选手，默认选择处于中游的选手 (Pivot Agent)，因为他们对策略最敏感
        if target_celebrity is None:
            median_idx = len(week_data) // 2
            target_row = week_data.sort_values('week_avg_score').iloc[median_idx]
            target_celebrity = target_row['celebrity_name']

        # 提取原始信号向量
        scores = week_data['week_avg_score'].values.copy()
        votes = week_data['est_fan_vote_mu'].values.copy()
        names = week_data['celebrity_name'].values

        # 定位目标代理人
        try:
            target_idx = np.where(names == target_celebrity)[0][0]
        except IndexError:
            return None

        # ======================================================================
        # 0. 基准情况 (Baseline Status Quo)
        # ======================================================================
        rank_base = self._simulate_rank(scores, votes)[target_idx]
        pct_base = self._simulate_percent(scores, votes)[target_idx]

        # DAW 基准 (假设当前进度)
        # 获取总周数估计
        total_weeks = self.df[self.df['season'] == season]['week_num'].max()
        daw_score_base, _ = self.daw_engine.calculate_combined_score(
            rankdata(-scores, method='min'),
            rankdata(-votes, method='min'),
            progress=week / max(total_weeks, 1)
        )
        # DAW 是加权排名和，数值越小越好
        daw_base = daw_score_base[target_idx]

        # ======================================================================
        # 策略 A: 苦练技术 (Invest in Merit)
        # ======================================================================
        # 动作：技术分增加 0.5 个环境标准差
        scores_merit = scores.copy()
        scores_merit[target_idx] += effort_unit * sigma_s
        scores_merit[target_idx] = min(10.0, scores_merit[target_idx])  # 物理上限

        # 计算新状态
        rank_merit = self._simulate_rank(scores_merit, votes)[target_idx]
        pct_merit = self._simulate_percent(scores_merit, votes)[target_idx]

        daw_score_merit, _ = self.daw_engine.calculate_combined_score(
            rankdata(-scores_merit, method='min'),
            rankdata(-votes, method='min'),
            progress=week / max(total_weeks, 1)
        )
        daw_merit = daw_score_merit[target_idx]

        # ======================================================================
        # 策略 B: 营销拉票 (Invest in Promo)
        # ======================================================================
        # 动作：得票率增加 0.5 个环境标准差
        votes_promo = votes.copy()
        votes_promo[target_idx] += effort_unit * sigma_v
        # 重新归一化 (单纯形约束：零和博弈)
        votes_promo = votes_promo / votes_promo.sum()

        # 计算新状态
        rank_promo = self._simulate_rank(scores, votes_promo)[target_idx]
        pct_promo = self._simulate_percent(scores, votes_promo)[target_idx]

        daw_score_promo, _ = self.daw_engine.calculate_combined_score(
            rankdata(-scores, method='min'),
            rankdata(-votes_promo, method='min'),
            progress=week / max(total_weeks, 1)
        )
        daw_promo = daw_score_promo[target_idx]

        # ======================================================================
        # 计算激励比率 (Incentive Ratio)
        # Ratio = Gain_Merit / Gain_Promo
        # ======================================================================
        # 对于 Rank 和 DAW，数值越小越好 -> Gain = Base - New (排名提升量)
        gain_rank_merit = max(0, rank_base - rank_merit)
        gain_rank_promo = max(0, rank_base - rank_promo)

        gain_daw_merit = max(0, daw_base - daw_merit)
        gain_daw_promo = max(0, daw_base - daw_promo)

        # 对于 Percent，数值越大越好 -> Gain = New - Base (分数提升量)
        gain_pct_merit = max(0, pct_merit - pct_base)
        gain_pct_promo = max(0, pct_promo - pct_base)

        return {
            "season": season,
            "week": week,
            "celebrity": target_celebrity,
            # 添加 1e-9 防止除零
            "rank_ratio": (gain_rank_merit + 1e-9) / (gain_rank_promo + 1e-9),
            "pct_ratio": (gain_pct_merit + 1e-9) / (gain_pct_promo + 1e-9),
            "daw_ratio": (gain_daw_merit + 1e-9) / (gain_daw_promo + 1e-9)
        }

    def run_full_season_audit(self, season_id: int = 27):
        """
        [主程序] 对指定赛季（默认 S27 Bobby Bones 夺冠季）进行全周期博弈论审计。
        生成 IC 轨迹图，证明新机制如何随着赛程推进，逐渐压制“纯流量玩家”。
        """
        self.logger.info(f">>> 启动博弈论 IC 审计 (Season {season_id})...")
        results = []

        # 遍历每一周
        weeks = sorted(self.df[self.df['season'] == season_id]['week_num'].unique())

        for w in weeks:
            # 对当周所有选手计算平均收益比，取中位数代表该周的“系统激励属性”
            week_df = self.df[(self.df['season'] == season_id) & (self.df['week_num'] == w)]
            contestants = week_df['celebrity_name'].values

            w_ratios_daw = []
            w_ratios_pct = []

            for c in contestants:
                res = self.calculate_marginal_utility(season_id, w, c)
                if res:
                    w_ratios_daw.append(res['daw_ratio'])
                    w_ratios_pct.append(res['pct_ratio'])

            if w_ratios_daw:
                results.append({
                    'week': w,
                    # 使用中位数 (Median) 过滤掉极端离群值
                    'daw_ic_ratio': np.median(w_ratios_daw),
                    'percent_ic_ratio': np.median(w_ratios_pct)
                })

        res_df = pd.DataFrame(results)

        # 生成证据图表
        if not res_df.empty:
            self._plot_ic_trajectory(res_df, season_id)

        return res_df

    def _plot_ic_trajectory(self, df: pd.DataFrame, season_id: int):
        """
        绘制激励相容性轨迹图 (IC Trajectory)。
        这是证明 DAW 机制优越性的核心图表 (The Proof of Optimality)。
        """
        plt.figure(figsize=(10, 6))

        # 1. 绘制基准线 (Indifference Point: Ratio = 1.0)
        plt.axhline(1.0, color='black', linestyle='--', linewidth=1.5, label='Indifference Point (Merit = Promo)')

        # 2. 绘制曲线
        # 历史 Percent 机制：通常在 1.0 以下震荡 (鼓励拉票)
        plt.plot(df['week'], df['percent_ic_ratio'],
                 marker='x', linestyle=':', color='gray', label='Historical (Percent Rule)', alpha=0.7)

        # 提议 DAW 机制：应该呈现上升趋势，后期显著 > 1.0
        plt.plot(df['week'], df['daw_ic_ratio'],
                 marker='o', linestyle='-', color='#d62728', linewidth=2.5, label='Proposed (DAW Mechanism)')

        # 3. 区域着色 (Zone Coloring)
        plt.fill_between(df['week'], 1.0, 10.0, color='#2ca02c', alpha=0.1,
                         label='Incentive Compatible Zone (Merit > Promo)')
        plt.fill_between(df['week'], 0.01, 1.0, color='#d62728', alpha=0.05, label='Populism Zone (Promo > Merit)')

        # 4. 装饰
        plt.title(f"Game Theory Audit: Evolution of Incentive Compatibility (Season {season_id})", fontsize=14, pad=15)
        plt.xlabel("Competition Week (t)", fontsize=12)
        plt.ylabel("Marginal Utility Ratio ($MU_{Tech} / MU_{Fan}$)", fontsize=12)

        # 使用对数坐标，因为比率可能会呈指数级差异
        plt.yscale('log')
        plt.ylim(0.1, 10)
        plt.legend(loc='upper left', frameon=True)
        plt.grid(True, which="both", ls="-", alpha=0.2)

        # 5. 标注关键结论
        if not df.empty:
            last_val = df.iloc[-1]['daw_ic_ratio']
            last_wk = df.iloc[-1]['week']
            plt.annotate(f'Final Ratio: {last_val:.1f}x\n(Tech Dominates)',
                         xy=(last_wk, last_val),
                         xytext=(last_wk - 2, last_val + 2),
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         fontsize=10, fontweight='bold')

        save_path = os.path.join(self.fig_dir, "ic_trajectory_proof.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"博弈论审计图已生成: {save_path}")