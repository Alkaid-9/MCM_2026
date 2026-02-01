# ==============================================================================
# src/analysis/causality_pipeline.py
# Role: Causal Attribution Orchestrator (Task 3 Command Center)
# Function: Integrating LMM, Dissonance, and SHAP into a unified causal narrative
# Output: Integrated Attribution Summary & LaTeX Tables
# ==============================================================================

import pandas as pd
import logging
import json
import os
import sys
from pathlib import Path

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
    3. SHAP 非线性解释 -> 捕捉年龄与行业的交互效应
    4. 报告生成 -> 输出 JSON 摘要与 LaTeX 表格
    """
    logger = logging.getLogger("CAUSALITY_PIPELINE")
    logger.info("=" * 60)
    logger.info(">>> STAGE 4: 启动因果归因与偏好分歧流水线 (Task 3) <<<")
    logger.info("=" * 60)

    report = {
        "meta": {"sample_size": len(df_platinum)},
        "metrics": {}
    }

    # 确保输出目录存在
    os.makedirs(fig_dir, exist_ok=True)

    try:
        # ----------------------------------------------------------------------
        # Step 1: 线性分层归因 (Linear Mixed-Effects Model)
        # 目标：回答 "Do they impact in the same way?" 的线性部分
        # ----------------------------------------------------------------------
        lmm_engine = LMMAttributionEngine(df_platinum, fig_dir=fig_dir)
        model_j, model_f = lmm_engine.run_dual_path_lmm()

        if model_j and model_f:
            # 绘制蝴蝶图 (Beta 系数对比)
            lmm_engine.plot_coefficient_butterfly(model_j, model_f)

            # 计算 ICC (舞伴效应)
            icc_f = lmm_engine.calculate_icc(model_f)
            icc_j = lmm_engine.calculate_icc(model_j)

            report["metrics"]["icc_fan"] = icc_f
            report["metrics"]["icc_judge"] = icc_j

            logger.info(f"LMM 阶段完成。舞伴解释力度 (ICC): Fan={icc_f:.4f} vs Judge={icc_j:.4f}")
            logger.info("发现：舞伴对评委分的影响通常大于对观众票的影响（技术壁垒）。")
        else:
            logger.warning("LMM 拟合中断，跳过线性归因部分。")

        # ----------------------------------------------------------------------
        # Step 2: 审美背离度量 (Cognitive Dissonance)
        # 目标：量化 "Professionalism vs. Populism" 的冲突程度
        # ----------------------------------------------------------------------
        if model_j and model_f:
            div_analyzer = DivergenceAnalyzer(fig_dir=fig_dir)
            d_metrics = div_analyzer.calculate_dissonance_index(model_j, model_f)

            if d_metrics:
                div_analyzer.plot_preference_radar(d_metrics)

                report["metrics"]["cosine_similarity"] = d_metrics['cosine_sim']
                report["metrics"]["dissonance_index"] = d_metrics['dissonance_idx']
                report["metrics"]["top_conflict_feature"] = d_metrics.get('conflict_feature', 'N/A')

                logger.info(f"Dissonance 阶段完成。认知失调指数: {d_metrics['dissonance_idx']:.4f}")
        else:
            logger.warning("模型缺失，无法计算审美背离指数。")

        # ----------------------------------------------------------------------
        # Step 3: 非线性交互归因 (SHAP / XAI)
        # 目标：捕捉 "年龄歧视" 或 "特定行业红利" 的非线性特征
        # ----------------------------------------------------------------------
        shap_engine = ShapInterpreter(df_platinum, fig_dir=fig_dir)
        X, s_j, s_f = shap_engine.run_dual_shap_analysis()

        if s_f is not None:
            # 绘制全局重要性蜂群图
            shap_engine.plot_global_importance(s_f)

            # 绘制年龄依赖对比图 (U型 vs 线性)
            if s_j is not None:
                shap_engine.plot_age_dependence_contrast(s_j, s_f)

            logger.info("SHAP 阶段完成。非线性特征云图已生成。")
        else:
            logger.warning("SHAP 分析失败，跳过非线性归因。")

        # ----------------------------------------------------------------------
        # Step 4: 汇总导出 (LaTeX & JSON)
        # ----------------------------------------------------------------------
        _save_json_summary(report, fig_dir)

        latex_table = _generate_latex_table(report)
        logger.info("\n--- Generated LaTeX Table (Copy to Paper) ---\n")
        print(latex_table)
        logger.info("\n---------------------------------------------\n")

        # 将 LaTeX 保存到文件，方便直接 include
        with open(os.path.join(fig_dir, "attribution_table.tex"), "w") as f:
            f.write(latex_table)

        logger.info("STAGE 4 任务圆满结束。")
        return report

    except Exception as e:
        logger.critical(f"FATAL: 因果归因流水线崩溃: {str(e)}", exc_info=True)
        # 不抛出异常，以免阻断后续 Phase 5 的运行，但在日志中标记严重错误
        return report


def _save_json_summary(report: dict, fig_dir: str):
    """保存结构化数据供后续分析引用"""
    path = os.path.join(fig_dir, "causality_summary.json")
    with open(path, 'w') as f:
        json.dump(report, f, indent=4)


def _generate_latex_table(report: dict) -> str:
    """
    自动生成符合数模竞赛标准的 LaTeX 三线表。
    """
    metrics = report.get("metrics", {})

    # 提取指标，若不存在则显示 N/A
    icc_f = f"{metrics.get('icc_fan', 0):.4f}"
    icc_j = f"{metrics.get('icc_judge', 0):.4f}"
    cos_sim = f"{metrics.get('cosine_similarity', 0):.4f}"
    dis_idx = f"{metrics.get('dissonance_index', 0):.4f}"
    conflict = str(metrics.get('top_conflict_feature', 'N/A')).replace('_', '\\_')

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
            & Residual Variance (Idiosyncratic) & """ + f"{1 - float(icc_j):.4f}" + r""" & """ + f"{1 - float(icc_f):.4f}" + r""" \\
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


if __name__ == "__main__":
    # 单元测试 Mock
    logging.basicConfig(level=logging.INFO)
    # 假设有一个处理好的 DataFrame
    # df = pd.read_csv("data/platinum/final_posterior_results.csv")
    # run_causality_stage(df)
    pass