# ==============================================================================
# src/analysis/causality_pipeline.py
# Role: Causal Attribution Orchestrator (Task 3 Command Center v6.8)
# Function: Integrating LMM, Dissonance, and SHAP into a unified causal narrative.
# Feature: Transactional Safety (SHAP failure won't kill LMM results).
# Output: Integrated Attribution Summary & LaTeX Tables.
# ==============================================================================

import pandas as pd
import logging
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

# 导入子引擎
from src.analysis.attribution_lmm import LMMAttributionEngine
from src.analysis.cognitive_divergence import DivergenceAnalyzer
from src.analysis.shap_interpreter import ShapInterpreter


def run_causality_stage(df_platinum: pd.DataFrame, fig_dir: str = "reports/figures/") -> dict:
    """
    [任务总控] 执行 Task 3 全量归因流水线。
    逻辑流：
    1. LMM 线性分解 -> 剥离舞伴效应 (ICC) & 行业偏好
    2. Dissonance 审计 -> 量化评委与观众的审美冲突
    3. SHAP 非线性解释 -> 捕捉年龄与行业的交互效应 (含异常捕获)
    4. 报告生成 -> 输出 JSON 摘要与 LaTeX 表格
    """
    logger = logging.getLogger("CAUSALITY_PIPELINE")
    logger.info("=" * 60)
    logger.info(">>> STAGE 4: 启动因果归因与偏好分歧流水线 (Task 3) <<<")
    logger.info("=" * 60)

    # 初始化报告容器
    report = {
        "meta": {"sample_size": len(df_platinum)},
        "metrics": {}
    }

    # 确保输出目录存在
    os.makedirs(fig_dir, exist_ok=True)
    report_dir = os.path.join(fig_dir, "..", "mechanism_audit")
    os.makedirs(report_dir, exist_ok=True)

    try:
        # ======================================================================
        # Step 1: 线性分层归因 (Linear Mixed-Effects Model)
        # 目标：回答 "Do they impact in the same way?" 的线性部分
        # ======================================================================
        lmm_engine = LMMAttributionEngine(df_platinum, fig_dir=fig_dir)
        model_j, model_f = lmm_engine.run_dual_path_lmm()

        if model_j and model_f:
            # 绘制蝴蝶图 (Beta 系数对比)
            lmm_engine.plot_coefficient_butterfly(model_j, model_f)

            # 计算 ICC (舞伴效应)
            icc_f = lmm_engine.calculate_icc(model_f)
            icc_j = lmm_engine.calculate_icc(model_j)

            report["metrics"]["icc_fan"] = float(icc_f)
            report["metrics"]["icc_judge"] = float(icc_j)

            logger.info(f"LMM 阶段完成。舞伴解释力度 (ICC): Fan={icc_f:.4f} vs Judge={icc_j:.4f}")
            logger.info("发现：舞伴对评委分的影响通常大于对观众票的影响（技术壁垒）。")
        else:
            logger.warning("LMM 拟合中断，跳过线性归因部分。")

        # ======================================================================
        # Step 2: 审美背离度量 (Cognitive Dissonance)
        # 目标：量化 "Professionalism vs. Populism" 的冲突程度
        # ======================================================================
        if model_j and model_f:
            div_analyzer = DivergenceAnalyzer(fig_dir=fig_dir)
            d_metrics = div_analyzer.calculate_dissonance_index(model_j, model_f)

            if d_metrics:
                div_analyzer.plot_preference_radar(d_metrics)

                report["metrics"]["cosine_similarity"] = float(d_metrics['cosine_sim'])
                report["metrics"]["dissonance_index"] = float(d_metrics['dissonance_idx'])
                report["metrics"]["top_conflict_feature"] = d_metrics.get('conflict_feature', 'N/A')

                logger.info(f"Dissonance 阶段完成。认知失调指数: {d_metrics['dissonance_idx']:.4f}")
        else:
            logger.warning("模型缺失，无法计算审美背离指数。")

        # ======================================================================
        # Step 3: 非线性交互归因 (SHAP / XAI)
        # 目标：捕捉 "年龄歧视" 或 "特定行业红利" 的非线性特征
        # ======================================================================
        try:
            shap_engine = ShapInterpreter(df_platinum, fig_dir=fig_dir)
            # 执行双路 SHAP 分析
            X, shap_j, shap_f = shap_engine.run_dual_shap_analysis()

            if shap_f is not None:
                # 绘制全局重要性蜂群图
                shap_engine.plot_global_importance(shap_f)

                # 绘制年龄依赖对比图 (U型 vs 线性)
                if shap_j is not None:
                    shap_engine.plot_age_dependence_contrast(shap_j, shap_f)

                logger.info("SHAP 阶段完成。非线性特征云图已生成。")
            else:
                logger.warning("SHAP 分析返回空结果，跳过绘图。")

        except Exception as e:
            # 事务保护：即使 SHAP 失败，也不要回滚 LMM 的结果
            logger.error(f"SHAP 非线性归因模块发生非致命错误: {e}")
            logger.info("系统将继续执行报告生成，仅跳过 SHAP 部分。")

        # ======================================================================
        # Step 4: 汇总导出 (LaTeX & JSON)
        # ======================================================================
        # 1. 保存 JSON (供 AbstractHelper 读取)
        _save_json_summary(report, report_dir)

        # 2. 生成 LaTeX 表格 (供论文直接使用)
        latex_table = _generate_latex_table(report)

        # 将 LaTeX 保存到文件
        tex_path = os.path.join(fig_dir, "attribution_table.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_table)

        logger.info(f"STAGE 4 任务圆满结束。LaTeX 表格已生成: {tex_path}")
        return report

    except Exception as e:
        logger.critical(f"FATAL: 因果归因流水线主进程崩溃: {str(e)}", exc_info=True)
        # 返回已有的部分结果，防止全盘皆输
        return report


def _save_json_summary(report: dict, output_dir: str):
    """保存结构化数据供后续分析引用"""
    path = os.path.join(output_dir, "causality_summary.json")
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4)
        print(f"   [Data] 归因摘要已落盘: {path}")
    except Exception as e:
        print(f"   [Error] JSON 保存失败: {e}")


def _generate_latex_table(report: dict) -> str:
    """
    自动生成符合数模竞赛标准的 LaTeX 三线表。
    """
    metrics = report.get("metrics", {})

    # 提取指标，若不存在则显示 N/A
    def fmt(key):
        val = metrics.get(key, 0.0)
        return f"{val:.4f}" if isinstance(val, (int, float)) else "N/A"

    icc_f = fmt('icc_fan')
    icc_j = fmt('icc_judge')
    cos_sim = fmt('cosine_similarity')
    dis_idx = fmt('dissonance_index')
    conflict = str(metrics.get('top_conflict_feature', 'N/A')).replace('_', '\\_')

    # 计算残差方差 (1 - ICC)
    resid_j = f"{1 - metrics.get('icc_judge', 0):.4f}"
    resid_f = f"{1 - metrics.get('icc_fan', 0):.4f}"

    table = r"""
\begin{table}[htbp]
    \centering
    \caption{Dual-Path Attribution Analysis: Decomposing the Merit-Popularity Gap}
    \label{tab:causality_metrics}
    \begin{tabular}{llcc}
        \toprule
        \textbf{Dimension} & \textbf{Metric} & \textbf{Judge (Merit)} & \textbf{Fan (Popularity)} \\
        \midrule
        \multirow{2}{*}{Structural Dependency} 
            & Partner ICC (Pro-Dancer Effect) & """ + icc_j + r""" & """ + icc_f + r""" \\
            & Residual Variance (Idiosyncratic) & """ + resid_j + r""" & """ + resid_f + r""" \\
        \midrule
        \multirow{3}{*}{Preference Alignment} 
            & Cosine Similarity ($\cos \theta$) & \multicolumn{2}{c}{""" + cos_sim + r"""} \\
            & \textbf{Dissonance Index} ($1 - \cos \theta$) & \multicolumn{2}{c}{\textbf{""" + dis_idx + r"""}} \\
            & Most Conflicting Feature & \multicolumn{2}{c}{\texttt{""" + conflict + r"""}} \\
        \bottomrule
    \end{tabular}
    \vspace{0.2cm}
    \begin{minipage}{0.9\textwidth}
    \small \textit{Note: ICC (Intraclass Correlation) quantifies the variance explained by the professional partner. A high Dissonance Index indicates a systemic divergence between expert criteria and public sentiment.}
    \end{minipage}
\end{table}
"""
    return table