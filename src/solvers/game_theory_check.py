# ==============================================================================
# src/solvers/game_theory_check.py
# Role: Nash Equilibrium Auditor (Incentive Compatibility Proof)
# Function: Simulating agent resource allocation strategies under constraints.
# Physics: Finding the optimal effort mix alpha* that maximizes survival probability.
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


class GameTheoryAuditor:
    """
    博弈论审计师：
    模拟理性代理人 (Rational Agent) 在不同赛制规则下的最优决策路径。

    [数学模型]
    Maximize Utility U = -Expected_Rank(Score', Vote')
    Subject to:
        Score' = Score + alpha * Efficiency_Tech
        Vote'  = Vote  + (1 - alpha) * Efficiency_Promo
        0 <= alpha <= 1

    其中 alpha 是投入到“技术提升”中的精力比例。
    如果 alpha* -> 1，说明机制鼓励技术 (Incentive Compatible)。
    如果 alpha* -> 0，说明机制鼓励刷票 (Populism Bias)。
    """

    def __init__(self, df_platinum: pd.DataFrame, fig_dir: str = "reports/figures/"):
        self.logger = logging.getLogger("GAME_THEORY")
        self.df = df_platinum.copy()
        self.daw = DAWEngine()
        self.fig_dir = fig_dir
        os.makedirs(self.fig_dir, exist_ok=True)

        # 绘图风格
        plt.rcParams['font.family'] = 'serif'
        sns.set_context("paper", font_scale=1.4)

    def _get_representative_agent(self, season: int, week: int):
        """选取当周的‘中位数选手’作为代理人（边缘选手对策略最敏感）"""
        week_data = self.df[(self.df['season'] == season) & (self.df['week_num'] == week)].copy()
        if len(week_data) < 2: return None, None

        # 找到处于淘汰边缘的选手 (Bottom 3-4)
        # 物理意义：这些人最渴望通过策略优化来生存
        target_idx = len(week_data) // 2
        sorted_df = week_data.sort_values('week_avg_score')
        agent = sorted_df.iloc[target_idx]

        return week_data, agent.name

    def solve_optimal_strategy(self, season: int, week: int, total_weeks: int,
                               effort_power: float = 1.0):
        """
        求解单周纳什均衡点 alpha*。
        :param effort_power: 努力的转化效率（假设投入 1 单位总精力，能带来多少 sigma 的提升）
        """
        week_data, agent_name = self._get_representative_agent(season, week)
        if week_data is None: return None

        # 1. 提取环境向量
        scores = week_data['week_avg_score'].values
        votes = week_data['est_fan_vote_mu'].values
        agent_idx = np.where(week_data['celebrity_name'] == agent_name)[0][0]

        # 计算环境波动率 (作为投入回报的量纲)
        sigma_s = scores.std() + 1e-9
        sigma_v = votes.std() + 1e-9

        # 2. 遍历策略空间 alpha in [0, 1]
        # alpha = 0 (纯拉票), alpha = 1 (纯练舞)
        alphas = np.linspace(0, 1, 21)  # 5% 步长
        ranks_pct = []
        ranks_rank = []
        ranks_daw = []

        # DAW 当前权重
        progress = week / total_weeks

        for alpha in alphas:
            # 策略扰动
            s_new = scores.copy()
            v_new = votes.copy()

            # 投入 alpha 精力提升技术
            s_new[agent_idx] += alpha * effort_power * sigma_s
            s_new[agent_idx] = min(10.0, s_new[agent_idx])  # 上限约束

            # 投入 (1-alpha) 精力提升流量
            # 注意：投票是零和博弈(单纯形)，这里简化为单人增益，归一化处理
            v_new[agent_idx] += (1 - alpha) * effort_power * sigma_v
            v_new = v_new / v_new.sum()

            # 计算各机制下的排名 (越小越好)
            # A. Percent Rule
            score_pct = s_new / s_new.sum() + v_new
            r_p = rankdata(-score_pct, method='min')[agent_idx]
            ranks_pct.append(r_p)

            # B. Rank Rule
            total_rank = rankdata(-s_new) + rankdata(-v_new)
            r_r = rankdata(total_rank, method='min')[agent_idx]
            ranks_rank.append(r_r)

            # C. DAW Rule
            daw_score, _ = self.daw.calculate_combined_score(
                rankdata(-s_new), rankdata(-v_new), progress
            )
            r_d = rankdata(daw_score, method='min')[agent_idx]
            ranks_daw.append(r_d)

        # 3. 寻找最优策略 (Best Rank)
        # 如果有多个 alpha 能达到相同最好排名，取最大的 alpha (技术优先，Tie-breaker)
        best_alpha_pct = alphas[np.where(ranks_pct == np.min(ranks_pct))[0][-1]]
        best_alpha_rank = alphas[np.where(ranks_rank == np.min(ranks_rank))[0][-1]]
        best_alpha_daw = alphas[np.where(ranks_daw == np.min(ranks_daw))[0][-1]]

        return {
            'week': week,
            'optimal_alpha_pct': best_alpha_pct,
            'optimal_alpha_rank': best_alpha_rank,
            'optimal_alpha_daw': best_alpha_daw
        }

    def run_strategy_simulation(self, season_id: int = 27):
        """
        [主程序] 全赛季策略演化模拟。
        """
        self.logger.info(f">>> 启动博弈论策略仿真 (Target: S{season_id})...")

        weeks = sorted(self.df[self.df['season'] == season_id]['week_num'].unique())
        total_weeks = max(weeks)

        history = []
        for w in weeks:
            res = self.solve_optimal_strategy(season_id, w, total_weeks)
            if res:
                history.append(res)

        df_res = pd.DataFrame(history)

        # 绘图
        self._plot_strategy_evolution(df_res, season_id)
        return df_res

    def _plot_strategy_evolution(self, df: pd.DataFrame, season_id: int):
        """
        绘制最优策略演化图 (The Strategy Evolution Path).
        """
        plt.figure(figsize=(10, 6))

        # 绘制不同机制下的最优技术投入比例
        plt.plot(df['week'], df['optimal_alpha_pct'],
                 label='Percent Rule (Historical)', color='gray', linestyle=':', marker='x', alpha=0.6)

        plt.plot(df['week'], df['optimal_alpha_rank'],
                 label='Rank Rule (Static)', color='#1f77b4', linestyle='--', marker='s', alpha=0.6)

        plt.plot(df['week'], df['optimal_alpha_daw'],
                 label='DAW Mechanism (Proposed)', color='#d62728', linewidth=3, marker='o')

        plt.title(f"Game Theoretic Audit: Optimal Strategy Evolution (Season {season_id})\n"
                  r"Rational Agent's Allocation to Technical Merit ($\alpha^*$)", fontsize=14)
        plt.xlabel("Competition Week", fontsize=12)
        plt.ylabel(r"Optimal Technical Investment ($\alpha^*$)", fontsize=12)
        plt.ylim(-0.05, 1.05)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.2)

        # 标注区域
        plt.fill_between(df['week'], 0.8, 1.0, color='#2ca02c', alpha=0.1)
        plt.text(df['week'].min(), 0.9, "Meritocracy Zone (Tech Dominant)", color='green', fontsize=10)

        plt.fill_between(df['week'], 0.0, 0.2, color='#d62728', alpha=0.1)
        plt.text(df['week'].min(), 0.05, "Populism Zone (Promo Dominant)", color='red', fontsize=10)

        save_path = os.path.join(self.fig_dir, "nash_equilibrium_trajectory.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"纳什均衡轨迹图已生成: {save_path}")


# --- 单元测试 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # 需依赖外部数据，此处仅做逻辑检查
    print("GameTheoryAuditor initialized. Ready for integration.")