# ==============================================================================
# src/analysis/survival_analyst.py
# Role: Actuarial Forensics Engine (Task 2 - Survival Bias)
# Function: Kaplan-Meier Survival Analysis & Log-Rank Significance Testing
# Logic: Comparing "Technical Talent Longevity" across different regimes.
# Standard: Medical-Grade Statistical Rigor (using lifelines library)
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import os
import logging


class SurvivalAnalyst:
    """
    生存分析专家：
    专门研究“技术殉道者”（高技术、低人气选手）在不同赛制下的生存期望。

    [学术逻辑]
    我们不仅看谁赢了，更看谁“死得冤”。
    通过分析技术分 Top 30% 选手的生存曲线，量化赛制对专业能力的“损耗率”。
    """

    def __init__(self, df_platinum: pd.DataFrame, figures_dir: str = "reports/figures/"):
        self.logger = logging.getLogger("SURVIVAL_ANALYST")
        self.df = df_platinum.copy()
        self.fig_dir = figures_dir
        os.makedirs(self.fig_dir, exist_ok=True)

        # 设置学术绘图风格
        try:
            plt.rcParams['font.family'] = 'serif'
            sns.set_context("paper", font_scale=1.5)
        except:
            pass

    def _prepare_survival_panel(self, data_source: pd.DataFrame = None):
        """
        构造生存分析的基础面板。
        物理意义：将周度流水数据转换为“一人一行”的寿命记录。
        """
        target = data_source if data_source is not None else self.df

        # 1. 聚合每个选手的生存终点
        survival_df = target.groupby(['season', 'celebrity_name']).agg({
            'week_num': 'max',  # 坚持到的最后一周
            'week_avg_score': 'mean',  # 赛季平均技术表现
            'est_fan_vote_mu': 'mean',  # 赛季平均人气表现
            'final_status': 'first'  # 最终结局标签
        }).reset_index()

        # 2. 定义 Duration (生存时长)
        survival_df['duration'] = survival_df['week_num']

        # 3. 定义 Event (是否观察到“死亡”)
        # 逻辑：如果状态是 Eliminated，则 Event=1；如果是 Winner 或 Withdrew，则视为右删失 (Censored)，Event=0
        survival_df['observed_event'] = survival_df['final_status'].apply(
            lambda x: 1 if x == 'Eliminated' else 0
        )

        return survival_df

    def define_merit_martyrs(self, panel_df: pd.DataFrame, top_q=0.7, bot_q=0.3):
        """
        【核心筛选】定义“技术流”选手群体。
        准则：平均技术分处于前 30%，但估计人气处于后 30%。
        这些选手是赛制公正性的最佳“矿工金丝雀”。
        """
        score_cutoff = panel_df['week_avg_score'].quantile(top_q)
        vote_cutoff = panel_df['est_fan_vote_mu'].quantile(bot_q)

        martyrs = panel_df[
            (panel_df['week_avg_score'] >= score_cutoff) &
            (panel_df['est_fan_vote_mu'] <= vote_cutoff)
            ].copy()

        self.logger.info(f"识别到 {len(martyrs)} 位‘技术流’选手。进入存活风险评估。")
        return martyrs

    def run_survival_comparison(self, data_source: pd.DataFrame = None):
        """
        [主实验] 对比 Rank 宇宙 vs. Percent 宇宙。
        直接回答 Task 2：Which method favors fan votes more?
        """
        self.logger.info(">>> 启动 Kaplan-Meier 生存偏差审计...")

        panel = self._prepare_survival_panel(data_source)
        martyrs = self.define_merit_martyrs(panel)

        if len(martyrs) < 8:
            self.logger.warning("样本量不足，将扩充筛选范围至 40% 以保证统计效力。")
            martyrs = self.define_merit_martyrs(panel, top_q=0.6, bot_q=0.4)

        # 区分赛制宇宙 (依据 rules.yaml)
        rank_seasons = [1, 2, 28, 29, 30, 31, 32, 33, 34]
        martyrs['regime'] = martyrs['season'].apply(
            lambda x: 'Rank-Based (Ordinal)' if x in rank_seasons else 'Percent-Based (Cardinal)'
        )

        # 执行统计拟合
        kmf = KaplanMeierFitter()
        plt.figure(figsize=(10, 7))

        results = {}

        for label, color, style in [
            ('Rank-Based (Ordinal)', '#1f77b4', '--'),
            ('Percent-Based (Cardinal)', '#d62728', '-')
        ]:
            mask = (martyrs['regime'] == label)
            if mask.any():
                kmf.fit(martyrs.loc[mask, 'duration'],
                        event_observed=martyrs.loc[mask, 'observed_event'],
                        label=label)
                kmf.plot_survival_function(ci_show=True, color=color, linestyle=style, lw=2.5)
                results[label] = {
                    "median_survival": kmf.median_survival_time_,
                    "durations": martyrs.loc[mask, 'duration'],
                    "events": martyrs.loc[mask, 'observed_event']
                }

        # ----------------------------------------------------------------------
        # 统计显著性检验 (Log-Rank Test)
        # ----------------------------------------------------------------------
        p_value = 1.0
        if len(results) == 2:
            lr_res = logrank_test(
                results['Rank-Based (Ordinal)']['durations'],
                results['Percent-Based (Cardinal)']['durations'],
                results['Rank-Based (Ordinal)']['events'],
                results['Percent-Based (Cardinal)']['events']
            )
            p_value = lr_res.p_value

        # 图表美化
        plt.title(f"Survival Analysis of Technical Talent (Top 30% Score)\nLog-rank test p={p_value:.4f}", fontsize=15)
        plt.xlabel("Weeks Survived", fontsize=12)
        plt.ylabel("Survival Probability", fontsize=12)
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.2)

        save_path = os.path.join(self.fig_dir, "survival_comparison_merit.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 结果汇总
        med_r = results.get('Rank-Based (Ordinal)', {}).get('median_survival', 0)
        med_p = results.get('Percent-Based (Cardinal)', {}).get('median_survival', 0)

        self.logger.info(f"审计完成。Rank 寿命中位数: {med_r}周 | Percent 寿命中位数: {med_p}周")

        return med_r, med_p, p_value


# --- 单元测试 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # 模拟数据验证
    mock_p = pd.DataFrame({
        'season': [1, 2, 3, 4],
        'celebrity_name': ['A', 'B', 'C', 'D'],
        'week_num': [10, 8, 5, 4],
        'week_avg_score': [28, 27, 29, 26],
        'est_fan_vote_mu': [0.05, 0.04, 0.03, 0.06],
        'final_status': ['Eliminated', 'Eliminated', 'Eliminated', 'Eliminated']
    })
    analyst = SurvivalAnalyst(mock_p)
    analyst.run_survival_comparison()