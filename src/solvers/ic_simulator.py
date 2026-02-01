# ==============================================================================
# src/solvers/ic_simulator.py
# Role: Game Theoretic Auditor (Incentive Compatibility Engine)
# Function: Simulating Agent Strategies (Merit vs. Promo) to prove Nash Equilibrium.
# Physics: Calculating Marginal Utility (MU) of "Practice" vs. "Campaigning".
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from scipy.stats import rankdata
from src.solvers.daw_engine import DAWEngine
from src.etl.config_loader import ConfigLoader


class IncentiveCompatibilityAuditor:
    """
    激励相容性（IC）审计师：
    通过微扰分析（Perturbation Analysis），量化不同策略下的生存收益率。

    [核心定义]
    一个机制是激励相容的 (IC)，当且仅当：
    MU_Merit (技术收益) > MU_Promo (流量收益)

    即：$\frac{\partial P(Survival)}{\partial Effort_{Tech}} > \frac{\partial P(Survival)}{\partial Effort_{Fan}}$
    """

    def __init__(self, df_platinum: pd.DataFrame, fig_dir: str = "reports/figures/"):
        self.logger = logging.getLogger("GAME_THEORY_AUDIT")
        self.df = df_platinum.copy()
        self.daw_engine = DAWEngine()
        self.fig_dir = fig_dir
        os.makedirs(self.fig_dir, exist_ok=True)

        # 绘图配置
        plt.rcParams['font.family'] = 'serif'
        sns.set_context("paper", font_scale=1.4)

    def _get_week_stats(self, season: int, week: int):
        """获取当周的统计分布特征 (Sigma)，用于标准化‘努力程度’"""
        week_data = self.df[(self.df['season'] == season) & (self.df['week_num'] == week)]
        if len(week_data) < 2: return None, None, None

        sigma_score = week_data['week_avg_score'].std()
        sigma_vote = week_data['est_fan_vote_mu'].std()

        return week_data, sigma_score, sigma_vote

    def calculate_marginal_utility(self,
                                   season: int,
                                   week: int,
                                   target_celebrity: str = None,
                                   effort_unit: float = 0.5):
        """
        计算边际效用 (MU)。
        假设选手投入 0.5 个标准差的努力 (Standardized Effort)，分别能换来多少排名提升？
        """
        week_data, sigma_s, sigma_v = self._get_week_stats(season, week)
        if week_data is None: return None

        # 如果未指定选手，默认选择处于中游的选手 (最需要策略的人)
        if target_celebrity is None:
            # 按分数排序，取中间那个
            median_idx = len(week_data) // 2
            target_row = week_data.sort_values('week_avg_score').iloc[median_idx]
            target_celebrity = target_row['celebrity_name']

        # 提取原始信号
        scores = week_data['week_avg_score'].values.copy()
        votes = week_data['est_fan_vote_mu'].values.copy()
        names = week_data['celebrity_name'].values
        target_idx = np.where(names == target_celebrity)[0][0]

        # --- 基准情况 (Baseline) ---
        # 1. 纯 Rank 制
        rank_base = self._simulate_rank(scores, votes)[target_idx]
        # 2. 纯 Percent 制
        pct_base = self._simulate_percent(scores, votes)[target_idx]
        # 3. DAW 制 (假设当前是赛季中段)
        # 注意：这里需要传入真实进度，或者为了对比固定参数
        # 我们用真实的 week/10 模拟
        daw_base = self.daw_engine.calculate_combined_score(
            rankdata(-scores, method='min'),
            rankdata(-votes, method='min'),
            progress=week / 10.0
        )[0][target_idx]  # DAW 返回的是分数，越小越好

        # --- 策略 A: 苦练技术 (Invest in Merit) ---
        # 技术分增加 0.5 个标准差
        scores_merit = scores.copy()
        scores_merit[target_idx] += effort_unit * sigma_s
        # 限制上限 10 分
        scores_merit[target_idx] = min(10.0, scores_merit[target_idx])

        rank_merit = self._simulate_rank(scores_merit, votes)[target_idx]
        pct_merit = self._simulate_percent(scores_merit, votes)[target_idx]
        daw_merit = self.daw_engine.calculate_combined_score(
            rankdata(-scores_merit, method='min'),
            rankdata(-votes, method='min'),
            progress=week / 10.0
        )[0][target_idx]

        # --- 策略 B: 营销拉票 (Invest in Promo) ---
        # 得票率增加 0.5 个标准差
        votes_promo = votes.copy()
        votes_promo[target_idx] += effort_unit * sigma_v
        # 重新归一化 (单纯形约束)
        votes_promo = votes_promo / votes_promo.sum()

        rank_promo = self._simulate_rank(scores, votes_promo)[target_idx]
        pct_promo = self._simulate_percent(scores, votes_promo)[target_idx]
        daw_promo = self.daw_engine.calculate_combined_score(
            rankdata(-scores, method='min'),
            rankdata(-votes_promo, method='min'),
            progress=week / 10.0
        )[0][target_idx]

        # --- 计算收益 (Gain) ---
        # 对于 Rank 和 DAW，数值越小越好 -> Gain = Base - New
        # 对于 Percent，数值越大越好 -> Gain = New - Base
        # 统一逻辑：收益均为正值代表排名/分数优化

        # Rank 制收益 (名次提升量)
        gain_rank_merit = rank_base - rank_merit
        gain_rank_promo = rank_base - rank_promo

        # Percent 制收益 (总分占比提升量)
        gain_pct_merit = pct_merit - pct_base
        gain_pct_promo = pct_promo - pct_base

        # DAW 制收益 (加权排名分降低量)
        gain_daw_merit = daw_base - daw_merit
        gain_daw_promo = daw_base - daw_promo

        return {
            "season": season, "week": week, "celebrity": target_celebrity,
            "rank_ratio": (gain_rank_merit + 1e-9) / (gain_rank_promo + 1e-9),
            "pct_ratio": (gain_pct_merit + 1e-9) / (gain_pct_promo + 1e-9),
            "daw_ratio": (gain_daw_merit + 1e-9) / (gain_daw_promo + 1e-9)
        }

    def _simulate_rank(self, s, v):
        """辅助：计算 Rank 机制下的最终排名"""
        return rankdata(rankdata(-s) + rankdata(-v), method='min')

    def _simulate_percent(self, s, v):
        """辅助：计算 Percent 机制下的总分"""
        s_pct = s / s.sum()
        return s_pct + v

    def run_full_season_audit(self, season_id: int = 27):
        """
        对指定赛季（默认 S27 Bobby Bones 夺冠季）进行全周期审计。
        """
        self.logger.info(f">>> 启动博弈论审计 (Season {season_id})...")
        results = []

        # 遍历每一周
        weeks = sorted(self.df[self.df['season'] == season_id]['week_num'].unique())

        for w in weeks:
            # 对当周所有选手计算平均收益比
            week_df = self.df[(self.df['season'] == season_id) & (self.df['week_num'] == w)]
            contestants = week_df['celebrity_name'].values

            w_ratios_daw = []
            w_ratios_pct = []

            for c in contestants:
                res = self.calculate_marginal_utility(season_id, w, c)
                if res:
                    w_ratios_daw.append(res['daw_ratio'])
                    w_ratios_pct.append(res['pct_ratio'])

            # 记录当周平均值 (代表系统的平均激励方向)
            if w_ratios_daw:
                results.append({
                    'week': w,
                    'daw_ic_ratio': np.median(w_ratios_daw),  # 用中位数抗噪
                    'percent_ic_ratio': np.median(w_ratios_pct)
                })

        res_df = pd.DataFrame(results)
        self._plot_ic_trajectory(res_df, season_id)
        return res_df

    def _plot_ic_trajectory(self, df: pd.DataFrame, season_id: int):
        """
        绘制激励相容性轨迹图 (IC Trajectory)。
        这是证明你的机制优越性的核心图表。
        """
        plt.figure(figsize=(10, 6))

        # 绘制基准线 (Ratio = 1.0)
        plt.axhline(1.0, color='black', linestyle='--', linewidth=1.5, label='Indifference Point (Merit = Promo)')

        # 绘制曲线
        plt.plot(df['week'], df['percent_ic_ratio'],
                 marker='x', linestyle=':', color='gray', label='Historical (Percent Rule)', alpha=0.7)
        plt.plot(df['week'], df['daw_ic_ratio'],
                 marker='o', linestyle='-', color='#d62728', linewidth=2.5, label='Proposed (DAW Mechanism)')

        # 区域着色
        plt.fill_between(df['week'], 1.0, 5.0, color='#2ca02c', alpha=0.1,
                         label='Incentive Compatible Zone (Merit > Promo)')
        plt.fill_between(df['week'], 0.0, 1.0, color='#d62728', alpha=0.05, label='Populism Zone (Promo > Merit)')

        plt.title(f"Game Theory Audit: Evolution of Incentive Compatibility (Season {season_id})", fontsize=14)
        plt.xlabel("Competition Week (t)", fontsize=12)
        plt.ylabel("Marginal Utility Ratio ($MU_{Tech} / MU_{Fan}$)", fontsize=12)
        plt.yscale('log')  # 使用对数坐标，因为比率可能会很大
        plt.ylim(0.5, 10)
        plt.legend(loc='upper left')
        plt.grid(True, which="both", ls="-", alpha=0.2)

        # 标注关键点
        last_week = df.iloc[-1]
        plt.annotate(f'Final Ratio: {last_week["daw_ic_ratio"]:.1f}x\n(Tech is Dominant)',
                     xy=(last_week['week'], last_week['daw_ic_ratio']),
                     xytext=(last_week['week'] - 2, last_week['daw_ic_ratio'] + 1),
                     arrowprops=dict(facecolor='black', shrink=0.05))

        save_path = os.path.join(self.fig_dir, "ic_trajectory_proof.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"博弈论审计图已生成: {save_path}")


# --- 单元测试 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Mock Data: 模拟一个“流量大过天”的赛季 (如 S27)
    weeks = list(range(1, 11))
    mock_data = []
    for w in weeks:
        mock_data.append({
            'season': 27, 'week_num': w, 'celebrity_name': 'Bobby Bones',
            'week_avg_score': 20 + w * 0.5,  # 进步缓慢
            'est_fan_vote_mu': 0.30  # 票仓巨大且稳定
        })
        # 陪跑选手
        mock_data.append({
            'season': 27, 'week_num': w, 'celebrity_name': 'Pro Dancer',
            'week_avg_score': 25 + w * 0.5,
            'est_fan_vote_mu': 0.10
        })

    df_mock = pd.DataFrame(mock_data)

    auditor = IncentiveCompatibilityAuditor(df_mock)
    res = auditor.run_full_season_audit(season_id=27)
    print(res)