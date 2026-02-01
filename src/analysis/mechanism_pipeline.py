# ==============================================================================
# src/analysis/mechanism_pipeline.py
# Role: Mechanism Forensics Orchestrator (Task 2 Command Center)
# Function: Integrating Survival Analysis, Sensitivity Check, and Counterfactuals.
# Key Fix: Defensive data stitching to prevent KeyErrors in downstream analysis.
# ==============================================================================

import pandas as pd
import numpy as np
import logging
import os
import json
from pathlib import Path
from tqdm import tqdm

# 引入子系统
from src.simulators.multiverse_engine import MultiverseEngine
from src.analysis.survival_analyst import SurvivalAnalyst
from src.analysis.sensitivity import SensitivityAnalyzer
from src.etl.config_loader import ConfigLoader

class MechanismAnalysisPipeline:
    """
    机制审计总控台：
    1. 运行多宇宙模拟 (Multiverse Simulation)。
    2. 执行生存分析 (Survival Analysis)，对比技术流选手的寿命。
    3. 执行敏感性测试 (Sensitivity Check)，量化抗噪能力。
    4. 执行法医级案例审计 (Case Study: Bobby Bones)。
    """

    def __init__(self, df_platinum: pd.DataFrame, results_dir: str = "reports/mechanism_audit/"):
        self.logger = logging.getLogger("MECHANISM_PIPELINE")
        self.df = df_platinum.copy()
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        # 实例化子引擎
        self.multiverse = MultiverseEngine(self.df)
        self.survival_analyst = SurvivalAnalyst(self.df, figures_dir=results_dir)
        self.sensitivity_analyst = SensitivityAnalyzer(self.df, figures_dir=results_dir)

    def _infer_simulated_losers(self, sim_df: pd.DataFrame) -> pd.DataFrame:
        """
        基于模拟排名推断谁被淘汰。
        物理意义：每一周排名数字最大（末位）的人即为模拟宇宙中的淘汰者。
        """
        # 计算每周最高排名（即总人数）
        sim_df['max_rank_this_week'] = sim_df.groupby(['season', 'week_num', 'universe'])['sim_placement'].transform('max')

        # 标记谁是败者 (True = Eliminated in Simulation)
        sim_df['is_simulated_loser'] = sim_df['sim_placement'] == sim_df['max_rank_this_week']
        return sim_df

    def _ensure_metadata_integrity(self, sim_df: pd.DataFrame) -> pd.DataFrame:
        """
        【防御性缝合】确保模拟结果包含分析所需的元数据。
        解决 KeyError: "Column(s) ['final_status', ...] do not exist"
        """
        required_meta = ['final_status', 'eliminated_week', 'week_avg_score']
        missing_cols = [c for c in required_meta if c not in sim_df.columns]

        if missing_cols:
            self.logger.warning(f"检测到模拟数据缺失元数据 {missing_cols}，正在执行热修复缝合...")
            # 从 Platinum 母表提取元数据
            meta_source = self.df[['season', 'week_num', 'celebrity_name'] + required_meta].drop_duplicates()

            # 执行左连接缝合
            sim_df = sim_df.merge(
                meta_source,
                on=['season', 'week_num', 'celebrity_name'],
                how='left',
                suffixes=('', '_orig') # 防止重名冲突
            )

            # 清理可能的重名列
            for col in required_meta:
                if f'{col}_orig' in sim_df.columns:
                    sim_df[col] = sim_df[col].fillna(sim_df[f'{col}_orig'])
                    sim_df.drop(columns=[f'{col}_orig'], inplace=True)

        return sim_df

    def run_full_audit(self):
        """
        [主入口] 执行 Task 2 全量审计流程。
        """
        self.logger.info("=" * 60)
        self.logger.info(">>> STAGE 3: 启动机制取证与审计流水线 (Task 2) <<<")
        self.logger.info("=" * 60)

        audit_report = {
            "survival_metrics": {},
            "sensitivity_metrics": {},
            "counterfactual_cases": {},
            "overall_verdict": ""
        }

        try:
            # --- Step 1: 平行宇宙全量推演 ---
            self.logger.info("[Step 1/4] 执行全赛季平行宇宙推演...")
            sim_history_raw = self.multiverse.run_all_universes()

            # 【关键修复】确保元数据完整
            sim_history_df = self._ensure_metadata_integrity(sim_history_raw)
            sim_history_df = self._infer_simulated_losers(sim_history_df)

            # 持久化模拟数据
            sim_path = os.path.join(self.results_dir, "multiverse_history.csv")
            sim_history_df.to_csv(sim_path, index=False)
            self.logger.info(f"平行宇宙历史数据已保存: {sim_path}")

            # --- Step 2: 生存分析 (量化 Meritocracy) ---
            self.logger.info("[Step 2/4] 执行 Kaplan-Meier 生存偏差审计...")
            # 调用 SurvivalAnalyst 处理生成的模拟序列
            med_r, med_p, p_val = self.survival_analyst.run_survival_comparison(data_source=sim_history_df)

            audit_report["survival_metrics"] = {
                "median_weeks_rank": float(med_r),
                "median_weeks_percent": float(med_p),
                "log_rank_p_value": float(p_val)
            }

            # --- Step 3: 鲁棒性压力测试 (Task 1 Consistency) ---
            self.logger.info("[Step 3/4] 执行蒙特卡洛噪声压力测试 (SNR Analysis)...")
            sens_df = self.sensitivity_analyst.run_noise_stress_test(n_sims=500, max_noise=0.2)

            if sens_df is not None and not sens_df.empty:
                self.sensitivity_analyst.plot_stability_curve(sens_df)
                # 提取 0.1 噪声水平下的表现
                idx_01 = (sens_df['noise_level'] - 0.1).abs().idxmin()

                flip_r = sens_df.loc[idx_01, 'flip_rate_rank']
                flip_p = sens_df.loc[idx_01, 'flip_rate_percent']
                advantage = (flip_p - flip_r) / (flip_p + 1e-9)

                audit_report["sensitivity_metrics"] = {
                    "flip_rate_rank_at_0.1": float(flip_r),
                    "flip_rate_percent_at_0.1": float(flip_p),
                    "robustness_advantage": float(advantage)
                }
            else:
                self.logger.warning("敏感性测试未生成有效数据。")

            # --- Step 4: 专项案例取证 (The Bobby Bones Case) ---
            self.logger.info("[Step 4/4] 正在执行 Bobby Bones (S27) 专项反事实推演...")
            self._audit_bobby_bones_case(sim_history_df, audit_report)

            # --- Step 5: 生成 LaTeX 交付物 ---
            self._generate_verdict(audit_report)
            self._export_audit_table(audit_report)

            # 保存报告
            with open(os.path.join(self.results_dir, "mechanism_audit_summary.json"), "w") as f:
                json.dump(audit_report, f, indent=4)

            return audit_report

        except Exception as e:
            self.logger.critical(f"机制审计流水线崩溃: {str(e)}", exc_info=True)
            raise

    def _audit_bobby_bones_case(self, sim_df: pd.DataFrame, report: dict):
        """法医级审计：Bobby Bones 在 Rank 宇宙中死了吗？"""
        # 筛选 S27 在 RANK 宇宙下的轨迹
        bb_case = sim_df[
            (sim_df['season'] == 27) &
            (sim_df['universe'] == 'RANK') &
            (sim_df['celebrity_name'].str.contains("Bones", na=False))
        ]

        if bb_case.empty:
            self.logger.warning("未找到 Bobby Bones 的模拟数据！")
            return

        # 检查是否有任何一周他成为了模拟败者
        death_events = bb_case[bb_case['is_simulated_loser'] == True]

        if not death_events.empty:
            death_week = int(death_events['week_num'].min())
            report["counterfactual_cases"]["Bobby_Bones_S27"] = {
                "actual": "Winner",
                "counterfactual": f"Eliminated Week {death_week}",
                "verdict": "Regime_Dependent (赛制红利受益者)"
            }
            self.logger.info(f"🧑‍⚖️ 法医结论：若采用 RANK 机制，Bobby Bones 将在第 {death_week} 周被淘汰。")
        else:
            report["counterfactual_cases"]["Bobby_Bones_S27"] = {
                "actual": "Winner",
                "counterfactual": "Winner",
                "verdict": "Robust_Popularity (绝对民意统治)"
            }
            self.logger.info("🧑‍⚖️ 法医结论：即便在 RANK 机制下，其粉丝基数仍足以支撑其夺冠。")

    def _generate_verdict(self, report):
        """生成自动化的学术结论摘要"""
        metrics = report.get('sensitivity_metrics', {})
        gain = metrics.get('robustness_advantage', 0)
        p = report.get('survival_metrics', {}).get('log_rank_p_value', 1.0)

        verdict = f"Our audit proves the Rank System provides a {gain:.1%} stability advantage against fan noise. "
        if p < 0.05:
            verdict += f"Log-rank test (p={p:.3e}) confirms significant protection of technical talent."
        else:
            verdict += "However, survival benefits remain marginally insignificant at current sample sizes."

        report['overall_verdict'] = verdict
        self.logger.info(f">>> 审计总结: {verdict}")

    def _export_audit_table(self, report):
        """生成符合顶刊审美的 LaTeX 三线表"""
        sm = report.get('survival_metrics', {})
        sem = report.get('sensitivity_metrics', {})
        case = report.get('counterfactual_cases', {}).get('Bobby_Bones_S27', {})

        # 防御性默认值
        flip_r = sem.get('flip_rate_rank_at_0.1', 0)
        flip_p = sem.get('flip_rate_percent_at_0.1', 0)
        med_r = sm.get('median_weeks_rank', 0)
        med_p = sm.get('median_weeks_percent', 0)
        p_val = sm.get('log_rank_p_value', 1.0)
        bb_outcome = case.get('counterfactual', 'N/A')

        latex = r"""
\begin{table}[htbp]
  \centering
  \caption{Mechanism Robustness and Meritocracy Audit (Task 2)}
  \label{tab:mechanism_audit}
  \begin{tabular}{lcc}
    \toprule
    \textbf{Evaluation Metric} & \textbf{Rank System} & \textbf{Percent System} \\
    \midrule
    Outcome Stability (Flip Rate @ $\sigma=0.1$) & \textbf{""" + f"{flip_r:.1%}" + r"""} & """ + f"{flip_p:.1%}" + r""" \\
    Technical Survival (Median Weeks) & \textbf{""" + f"{med_r:.1f}" + r"""} & """ + f"{med_p:.1f}" + r""" \\
    Bobby Bones Anomaly (S27) & """ + f"{bb_outcome}" + r""" & Winner (Actual) \\
    \midrule
    \textbf{Statistical Significance} & \multicolumn{2}{c}{Log-rank $p = """ + f"{p_val:.4e}" + r"""$} \\
    \bottomrule
  \end{tabular}
\end{table}
"""
        with open(os.path.join(self.results_dir, "mechanism_audit_table.tex"), "w") as f:
            f.write(latex)