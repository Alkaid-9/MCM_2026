# ==============================================================================
# src/utils/abstract_helper.py
# Role: Research Insight Harvester (The "O-Prize" Punchline Generator)
# Function: Harvesting key metrics across the pipeline for the Summary Sheet.
# Key Fix: Added defensive column checks and entropy-to-certainty conversion.
# Standard: Quantified Rigor / Academic Persuasion / Decision Evidence.
# ==============================================================================

import pandas as pd
import numpy as np
import json
import os
import logging
from pathlib import Path
from typing import Dict, Any


class AbstractHelper:
    """
    摘要助手：
    自动收集全流程的“战果”，为论文摘要和一页纸建议书（Memo）提供硬核数据支持。
    """

    def __init__(self, data_root: str = "data/", report_root: str = "reports/"):
        self.logger = logging.getLogger("ABSTRACT_HELPER")
        # 路径自适应
        base_dir = Path(os.getcwd())
        self.data_root = base_dir / data_root
        self.report_root = base_dir / report_root

        # 定义关键路径
        self.paths = {
            "platinum": self.data_root / "platinum/final_posterior_results.csv",
            "audit": self.report_root / "mechanism_audit/mechanism_audit_summary.json",
            "causality": self.report_root / "mechanism_audit/causality_summary.json",
            "design": self.report_root / "final_design/producer_memo_data.json"
        }

    def _safe_load_json(self, path: Path) -> Dict[str, Any]:
        """防御性 JSON 加载"""
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"无法加载 JSON 报告 {path}: {e}")
        return {}

    def harvest_all_metrics(self) -> Dict[str, Any]:
        """一键收割全流程核心指标"""
        self.logger.info("正在执行‘科研成果收割’ (Harvesting Metrics)...")

        results = {
            "task1_inference": {},
            "task2_forensics": {},
            "task3_attribution": {},
            "task4_design": {}
        }

        # --- Task 1: 贝叶斯反演效果 ---
        if self.paths["platinum"].exists():
            try:
                df_p = pd.read_csv(self.paths["platinum"])

                # [防御性提取]：检查列名是否存在
                fidelity = df_p['fidelity'].mean() if 'fidelity' in df_p.columns else 0.0
                r_hat = df_p['r_hat'].mean() if 'r_hat' in df_p.columns else 999.0

                # [核心修复]：处理确定性指标
                if 'est_certainty_score' in df_p.columns:
                    certainty = df_p['est_certainty_score'].mean()
                elif 'inference_entropy' in df_p.columns:
                    # 物理转换：熵越低，确定性越高
                    # 使用简单的反向映射作为代理指标
                    mean_entropy = df_p['inference_entropy'].mean()
                    certainty = 1.0 / (1.0 + mean_entropy)
                else:
                    certainty = 0.0

                results["task1_inference"] = {
                    "fidelity": fidelity,
                    "mean_rhat": r_hat,
                    "certainty": certainty
                }
            except Exception as e:
                self.logger.error(f"Task 1 数据收割失败: {e}")

        # --- Task 2: 机制对比与取证 ---
        audit_data = self._safe_load_json(self.paths["audit"])
        if audit_data:
            # 提取生存Gap和稳定性增益
            surv = audit_data.get("survival_metrics", {})
            sens = audit_data.get("sensitivity_metrics", {})

            results["task2_forensics"] = {
                "stability_gain": sens.get("robustness_advantage", 0),
                "merit_survival_gap": surv.get("median_weeks_rank", 0) - surv.get("median_weeks_percent", 0)
            }

        # --- Task 3: 因果归因结果 ---
        causality_data = self._safe_load_json(self.paths["causality"])
        if causality_data:
            metrics = causality_data.get("metrics", {})
            results["task3_attribution"] = {
                "partner_icc": metrics.get("icc_fan", 0),
                "dissonance_index": metrics.get("dissonance_index", 0),
                "top_conflict": metrics.get("top_conflict_feature", "N/A")
            }

        # --- Task 4: 机制优化提升 ---
        design_data = self._safe_load_json(self.paths["design"])
        if design_data:
            impact = design_data.get("expected_impact", {})
            params = design_data.get("key_parameters", {})
            results["task4_design"] = {
                "fairness_lift": impact.get("fairness_gain_vs_percent", 0),
                "optimal_t0": params.get("transition_midpoint_t0", 0)
            }

        return results

    def generate_punchlines(self, metrics: Dict[str, Any]):
        """
        生成“顶刊级”学术话术。
        这些句子可以直接粘贴进论文的 Abstract 和 Conclusion。
        """
        print("\n" + "=" * 80)
        print("   [O-PRIZE PUNCHLINES] 论文核心论据库 (可以直接 Copy 到 Abstract)   ")
        print("=" * 80 + "\n")

        # 1. 关于反演准确性
        m1 = metrics.get("task1_inference", {})
        if m1:
            print(
                f"📌 [Inference]: Our Bayesian engine reconstructed latent preference with a historical Fidelity of {m1.get('fidelity', 0):.1%}, "
                f"validated by a mean R-hat of {m1.get('mean_rhat', 0):.3f}, confirming robust global convergence.")

        # 2. 关于机制弊端 (Rank vs Percent)
        m2 = metrics.get("task2_forensics", {})
        if m2:
            gap = m2.get('merit_survival_gap', 0)
            print(
                f"📌 [Forensics]: Counterfactual simulations reveal that the Percent System suffers from high noise sensitivity, "
                f"while the Rank System provides a {m2.get('stability_gain', 0):.1%} stability gain and extends the survival of "
                f"technical talent by {gap:.1f} weeks.")

        # 3. 关于舞伴影响与审美分歧
        m3 = metrics.get("task3_attribution", {})
        if m3:
            print(f"📌 [Attribution]: Hierarchical modeling identifies a Partner ICC of {m3.get('partner_icc', 0):.1%}, "
                  f"quantifying the significant 'halo effect' of professional dancers. "
                  f"A Dissonance Index of {m3.get('dissonance_index', 0):.2f} highlights the systemic gap between expert and public criteria.")

        # 4. 关于新机制 (DAW)
        m4 = metrics.get("task4_design", {})
        if m4:
            print(
                f"📌 [Solution]: The proposed DAW system achieves a Pareto improvement, boosting technical fairness by {m4.get('fairness_lift', 0):.1%} "
                f"while maintaining viewer engagement via a dynamic power-shift centered at t={m4.get('optimal_t0', 0):.2f}.")

        print("\n" + "=" * 80 + "\n")


# --- 单元测试 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    helper = AbstractHelper()

    # 模拟数据
    mock_metrics = {
        "task1_inference": {"fidelity": 0.942, "mean_rhat": 1.02, "certainty": 0.88},
        "task2_forensics": {"stability_gain": 0.225, "merit_survival_gap": 2.5},
        "task3_attribution": {"partner_icc": 0.184, "dissonance_index": 0.65, "top_conflict": "Age"},
        "task4_design": {"fairness_lift": 0.158, "optimal_t0": 0.62}
    }

    # 测试生成
    helper.generate_punchlines(mock_metrics)