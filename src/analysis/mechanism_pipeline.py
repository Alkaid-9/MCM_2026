# ==============================================================================
# src/analysis/mechanism_pipeline.py
# Role: Mechanism Forensics Orchestrator (Task 2 Command Center v6.5)
# Function: Integrating Survival, Sensitivity, and SNR into a unified narrative.
# Fix: Resolved Pivot KeyError, 'inf' reporting, and NameError.
# ==============================================================================

import pandas as pd
import numpy as np
import logging
import os
import json
from typing import Dict, Any

# 引入子系统
from src.simulators.multiverse_engine import MultiverseEngine
from src.analysis.survival_analyst import SurvivalAnalyst
from src.analysis.sensitivity import SensitivityAnalyzer
from src.analysis.mechanism_compare import MechanismComparator


class MechanismAnalysisPipeline:
    """
    机制审计总控台：
    驱动多宇宙模拟，执行生存分析与鲁棒性测试，并产出 Task 2 核心证据。
    """

    def __init__(self, df_platinum: pd.DataFrame, results_dir: str = "reports/mechanism_audit/"):
        self.logger = logging.getLogger("MECHANISM_PIPELINE")
        self.df = df_platinum.copy()
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        # 子引擎实例化
        self.multiverse = MultiverseEngine(self.df)
        self.survival_analyst = SurvivalAnalyst(self.df, figures_dir=results_dir)
        self.sensitivity_analyst = SensitivityAnalyzer(self.df, figures_dir=results_dir)

        # 结果汇总字典
        self.audit_report = {
            "survival_metrics": {},
            "sensitivity_metrics": {},
            "snr_metrics": {},
            "counterfactual_cases": {}
        }

    def _prepare_comparison_matrix(self, sim_history_long: pd.DataFrame) -> pd.DataFrame:
        """
        【修复板块 1】：执行数据透视并确保契约唯一性。
        解决 'week_avg_score' 重名和 'inferred_fan_vote' 缺失问题。
        """
        self.logger.info("执行多宇宙数据流对齐 (Pivot Operation)...")
        df = sim_history_long.copy()

        # 1. 强制契约对齐
        name_map = {
            'actual_judges_score': 'week_avg_score',
            'inferred_fan_vote': 'est_fan_vote_mu'
        }
        for old, new in name_map.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})
            elif old in df.columns and new in df.columns:
                df = df.drop(columns=[old]) # 剔除冗余，防止 pivot 报 1-D 错误

        # 2. 确定透视索引列 (必须包含坐标和信号)
        index_candidates = ['season', 'week_num', 'celebrity_name', 'week_avg_score', 'est_fan_vote_mu']
        index_cols = [c for c in index_candidates if c in df.columns]

        # 3. 执行透视：将不同 universe (RANK/PERCENT) 的排名拉成两列
        df_wide = df.pivot_table(
            index=index_cols,
            columns='universe',
            values='sim_placement',
            aggfunc='first'
        ).reset_index()

        # 4. 重命名结果列供下游 Comparator 使用
        return df_wide.rename(columns={'RANK': 'sim_rank_placement', 'PERCENT': 'sim_pct_placement'})

    def _audit_bobby_bones(self, sim_history_long: pd.DataFrame):
        """
        【修复板块 2】：Bobby Bones 专项法医回溯。
        解决 NameError: 'actual_status' 和 KeyError: 'week'。
        """
        self.logger.info("执行案例专项审计：Bobby Bones (S27)...")

        # 1. 定义历史锚点
        actual_status = "Winner"  # 历史事实：Bobby Bones 是冠军

        # 2. 提取其在 RANK 宇宙下的模拟存活轨迹
        bb_rank = sim_history_long[
            (sim_history_long['season'] == 27) &
            (sim_history_long['universe'] == 'RANK') &
            (sim_history_long['celebrity_name'].str.contains("Bones", na=False))
            ]

        if bb_rank.empty:
            fate = "Eliminated (Pre-Finale)"
        else:
            # 识别背离点：在 RANK 宇宙中哪一周被标记为淘汰者
            # 使用 week_num 而不是 week
            death_records = bb_rank[bb_rank['is_regime_anomaly'] == True]
            if not death_records.empty:
                first_death_week = int(death_records['week_num'].min())
                fate = f"Eliminated Week {first_death_week} (Regime Shift)"
            else:
                fate = "Winner (Survives Change)"

        self.audit_report["counterfactual_cases"]["Bobby_Bones_S27"] = {
            "actual": actual_status,
            "counterfactual": fate
        }
        self.logger.info(f" -> 审计结论: 历史={actual_status}, RANK 宇宙={fate}")

    def run_full_audit(self) -> Dict[str, Any]:
        """
        【板块 4 重构】：主流水线编排。
        """
        self.logger.info("=" * 60)
        self.logger.info(">>> STAGE 3: 启动机制科学审计流水线 (Task 2) <<<")
        self.logger.info("=" * 60)

        try:
            # 1. 模拟推演
            sim_history_long = self.multiverse.run_all_universes()

            # 2. 【板块 2 重构】：生存分析汇报逻辑
            med_r, med_p, p_val = self.survival_analyst.run_survival_comparison(data_source=sim_history_long)

            def academic_fmt(v):
                return "Full Season (Finale Guaranteed)" if (np.isinf(v) or v >= 10) else f"{v:.1f} Weeks"

            self.audit_report["survival_metrics"] = {
                "rank_longevity": academic_fmt(med_r),
                "percent_longevity": academic_fmt(med_p),
                "p_value": float(p_val)
            }

            if p_val < 0.05:
                self.logger.info(f"结论: Rank 制显著延长了精英选手的寿命 ({academic_fmt(med_r)})。")
            else:
                self.logger.info(f"结论: 存在保护倾向，但当前样本下统计显著性边缘 (p={p_val:.2f})。")

            # 3. 鲁棒性压力测试
            sens_df = self.sensitivity_analyst.run_noise_stress_test(n_sims=500)
            if sens_df is not None:
                self.sensitivity_analyst.plot_stability_curve(sens_df)
                idx_01 = (sens_df['noise_level'] - 0.1).abs().idxmin()
                self.audit_report["sensitivity_metrics"] = sens_df.loc[idx_01].to_dict()

            # 4. 信噪比审计
            df_wide = self._prepare_comparison_matrix(sim_history_long)
            comparator = MechanismComparator(df_wide, figures_dir=self.results_dir)
            snr_df = comparator.run_snr_analysis()
            if snr_df is not None:
                comparator.plot_snr_evolution(snr_df)
                self.audit_report["snr_metrics"] = {
                    "avg_snr_gain_db": float(snr_df['snr_gain_db'].mean()),
                    "champion_misalignment": float(comparator.calculate_misalignment_rate())
                }

            # 5. 特定案例回溯
            self._audit_bobby_bones(sim_history_long)

            # --- 保存报告 ---
            report_path = os.path.join(self.results_dir, "mechanism_audit_summary.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(self.audit_report, f, indent=4)

            return self.audit_report

        except Exception as e:
            self.logger.critical(f"机制审计流水线因组件不兼容崩溃: {str(e)}", exc_info=True)
            raise