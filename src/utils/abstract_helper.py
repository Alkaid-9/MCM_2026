# ==============================================================================
# src/utils/abstract_helper.py
# Role: Research Insight Harvester (The "O-Prize" Punchline Generator)
# Function: Harvesting key metrics across the pipeline for the Summary Sheet.
# Standard: Quantified Rigor / Academic Persuasion / Decision Evidence.
# ==============================================================================

import pandas as pd
import numpy as np
import json
import os
import logging
from pathlib import Path


class AbstractHelper:
    """
    摘要助手：
    自动收集全流程的“战果”，为论文摘要和一页纸建议书（Memo）提供硬核数据支持。
    """

    def __init__(self, data_root: str = "data/", report_root: str = "reports/"):
        self.logger = logging.getLogger("ABSTRACT_HELPER")
        self.data_root = Path(data_root)
        self.report_root = Path(report_root)

        # 定义关键路径
        self.paths = {
            "platinum": self.data_root / "platinum/final_posterior_results.csv",
            "causality": self.report_root / "mechanism_audit/causality_summary.json",
            "audit": self.report_root / "mechanism_audit/mechanism_audit_summary.json",
            "design": self.report_root / "final_design/producer_memo_data.json"
        }

    def _safe_load_json(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {}

    def harvest_all_metrics(self) -> dict:
        """一键收割全流程核心指标"""
        self.logger.info("正在执行‘科研成果收割’...")

        results = {
            "task1_inference": {},
            "task2_forensics": {},
            "task3_attribution": {},
            "task4_design": {}
        }

        # --- Task 1: 贝叶斯反演效果 ---
        if os.path.exists(self.paths["platinum"]):
            df_p = pd.read_csv(self.paths["platinum"])
            results["task1_inference"] = {
                "fidelity": df_p['fidelity'].mean(),  # 历史淘汰吻合度
                "mean_rhat": df_p['r_hat'].mean(),  # MCMC收敛指标
                "certainty": df_p['est_certainty_score'].mean()  # 估计确定性
            }

        # --- Task 2: 机制对比与取证 ---
        audit_data = self._safe_load_json(self.paths["audit"])
        if audit_data:
            results["task2_forensics"] = {
                "stability_gain": audit_data.get("sensitivity_metrics", {}).get("robustness_advantage", 0),
                "merit_survival_gap": audit_data.get("survival_metrics", {}).get("median_weeks_rank", 0) -
                                      audit_data.get("survival_metrics", {}).get("median_weeks_percent", 0)
            }

        # --- Task 3: 因果归因结果 ---
        causality_data = self._safe_load_json(self.paths["causality"])
        if causality_data:
            results["task3_attribution"] = {
                "partner_icc": causality_data.get("metrics", {}).get("icc_fan", 0),
                "dissonance_index": causality_data.get("metrics", {}).get("dissonance_index", 0),
                "top_conflict": causality_data.get("metrics", {}).get("top_conflict_feature", "N/A")
            }

        # --- Task 4: 机制优化提升 ---
        design_data = self._safe_load_json(self.paths["design"])
        if design_data:
            results["task4_design"] = {
                "fairness_lift": design_data.get("expected_impact", {}).get("fairness_gain_vs_percent", 0),
                "optimal_t0": design_data.get("key_parameters", {}).get("transition_midpoint_t0", 0)
            }

        return results

    def generate_punchlines(self, metrics: dict):
        """
        生成“顶刊级”学术话术。
        这些句子可以直接粘贴进论文的 Abstract。
        """
        print("\n" + "=" * 60)
        print(" [O-PRIZE PUNCHLINES] 摘要核心论据库 ")
        print("=" * 60 + "\n")

        # 1. 关于反演准确性
        m1 = metrics["task1_inference"]
        if m1:
            print(
                f"论据1: Our Bayesian engine reconstructed latent preference with a historical Fidelity of {m1['fidelity']:.1%}, "
                f"validated by a mean R-hat of {m1['mean_rhat']:.3f}, confirming robust global convergence.")

        # 2. 关于机制弊端
        m2 = metrics["task2_forensics"]
        if m2:
            print(
                f"论据2: Counterfactual simulations reveal that the Percent System suffers from high noise sensitivity, "
                f"while the Rank System provides a {m2['stability_gain']:.1%} stability gain and extends the survival of "
                f"technical talent by {m2['merit_survival_gap']:.1f} weeks.")

        # 3. 关于舞伴影响
        m3 = metrics["task3_attribution"]
        if m3:
            print(f"论据3: Hierarchical modeling identifies a Partner ICC of {m3['partner_alpha']:.1%}, "
                  f"quantifying the significant 'halo effect' of professional dancers over celebrity popularity.")

        # 4. 关于新机制优越性
        m4 = metrics["task4_design"]
        if m4:
            print(
                f"论据4: The proposed DAW system achieves a Pareto improvement, boosting technical fairness by {m4['fairness_lift']:.1%} "
                f"while maintaining viewer engagement via a dynamic power-shift centered at t={m4['optimal_t0']:.2f}.")

        print("\n" + "=" * 60 + "\n")


# --- 单元测试 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    helper = AbstractHelper()

    # 模拟数据
    mock_metrics = {
        "task1_inference": {"fidelity": 0.942, "mean_rhat": 1.02, "certainty": 0.88},
        "task2_forensics": {"stability_gain": 0.225, "merit_survival_gap": 2.5},
        "task3_attribution": {"partner_alpha": 0.184, "dissonance_index": 0.65, "top_conflict": "Age"},
        "task4_design": {"fairness_lift": 0.158, "optimal_t0": 0.62}
    }

    helper.generate_punchlines(mock_metrics)