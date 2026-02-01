# ==============================================================================
# MCM 2026 Problem C: "The Invisible Hand of the Audience"
# MAIN INTEGRATION PIPELINE (Final Production Version)
# Role: Supreme Command Center for Stage 1-6 Execution
# Architecture: Hybrid HPC (C++/Numba) + Bayesian Inference + Causal XAI
# Standard: O-Prize Academic Excellence / Quant Engineering
# ==============================================================================

import os
import sys
import time
import logging
import pandas as pd
import numpy as np

# --- 0. 环境霸权配置 (Environment Locking) ---
# 必须在导入 numpy/pandas/sklearn 之前设置，否则无效
# 物理意义：将所有数学库的算力锁定在 23 个物理核心上，避免超线程上下文切换的开销
os.environ["OMP_NUM_THREADS"] = "23"
os.environ["MKL_NUM_THREADS"] = "23"
os.environ["NUMBA_NUM_THREADS"] = "23"
os.environ["OPENBLAS_NUM_THREADS"] = "23"

# 导入自研组件
from src.etl.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.etl.pipeline import run_etl_stage
from src.bridge.mcmc_wrapper import MCMCInferenceWrapper
from src.analysis.mechanism_pipeline import MechanismAnalysisPipeline
from src.analysis.causality_pipeline import run_causality_stage
from src.solvers.design_pipeline import MechanismDesignPipeline
from src.utils.abstract_helper import AbstractHelper
from src.utils.exporter import MCMProjectExporter


def main():
    # --- 1. 系统初始化 ---
    # 加载全局配置与日志系统
    config_provider = ConfigLoader()
    log_path = config_provider.get_path('logs')
    logger = setup_logger("MCM_COMMAND_CENTER", log_path)

    start_wall_clock = time.time()

    logger.info("=" * 80)
    logger.info("   MCM 2026 PROBLEM C: FULL SYSTEM ACTIVATED   ")
    logger.info("   Kernel: C++17 Accelerated Bayesian Inference Engine   ")
    logger.info("=" * 80)

    try:
        # ======================================================================
        # STAGE 1: 数据取证与信号精炼 (Data Forensics & Signal Processing)
        # 目标：从脏数据中提取 33 维黄金因子库，建立因果防火墙
        # ======================================================================
        logger.info(">>> STAGE 1: 启动数据地基构建 (ETL Pipeline)...")
        df_gold = run_etl_stage()

        # 审计检查
        logger.info(f"      [OK] 数据地基已夯实: {df_gold.shape[0]} 观测点, {df_gold.shape[1]} 因子")

        # ======================================================================
        # STAGE 2: 贝叶斯潜变量反演 (Bayesian Inference Engine)
        # 目标：利用 C++ Kernel 逆向推导 34 季隐藏的观众投票分布 (Task 1)
        # ======================================================================
        logger.info(">>> STAGE 2: 启动 C++ 高性能反演引擎 (Bayesian Task 1)...")
        inference_engine = MCMCInferenceWrapper()

        # 执行批处理推理：23 核并行咆哮
        # 这里会调用 bindings.cpp 里的 run_parallel_inference
        df_platinum = inference_engine.run_batch_inference(df_gold)

        # 持久化 Platinum Layer (这是所有后续分析的“浓缩铀”)
        platinum_path = config_provider.get_path('platinum_results')
        # 确保目录存在
        os.makedirs(os.path.dirname(platinum_path), exist_ok=True)
        df_platinum.to_csv(platinum_path, index=False)
        logger.info(f"      [OK] 潜变量后验分布已固化至: {platinum_path}")

        # ======================================================================
        # STAGE 3: 机制审计与反事实推演 (Forensics & Multiverse)
        # 目标：量化 Rank/Percent 优劣，审计 Bobby Bones 案例 (Task 2)
        # ======================================================================
        logger.info(">>> STAGE 3: 启动‘平行宇宙’机制审计 (Forensics Task 2)...")
        forensics_pipeline = MechanismAnalysisPipeline(df_platinum)

        # 执行生存分析、敏感性测试、Bobby Bones 专项审计
        audit_report = forensics_pipeline.run_full_audit()

        # 提取关键学术指标用于日志展示
        robustness_gain = audit_report['sensitivity_metrics'].get('robustness_advantage', 0)
        logger.info(f"      [OK] 机制审计完成。Rank 赛制稳定性相对增益: {robustness_gain:.2%}")

        # ======================================================================
        # STAGE 4: 因果归因与审美分歧 (Causal Attribution & XAI)
        # 目标：剥离舞伴红利，量化认知失调，绘制蝴蝶图 (Task 3)
        # ======================================================================
        logger.info(">>> STAGE 4: 启动归因分析与因果推断 (Causality Task 3)...")
        fig_dir = config_provider.get_path('figures_dir')

        # 执行 LMM、SHAP、Dissonance 分析
        causality_report = run_causality_stage(df_platinum, fig_dir=fig_dir)

        dissonance = causality_report['metrics'].get('dissonance_index', 0)
        logger.info(f"      [OK] 归因完成。识别系统性审美分歧指数: {dissonance:.4f}")

        # ======================================================================
        # STAGE 5: 帕累托最优机制设计 (Mechanism Design)
        # 目标：寻找 Equity-Efficiency 平衡点，提出 DAW 系统 (Task 4)
        # ======================================================================
        logger.info(">>> STAGE 5: 启动多目标寻优与博弈论审计 (Design Task 4)...")
        design_pipeline = MechanismDesignPipeline(df_platinum)

        # 针对 Bobby Bones 所在的 S27 进行高压参数寻优
        design_metrics = design_pipeline.run_design_suite(target_season=27)

        equity_lift = design_metrics['DAW']['equity'] - design_metrics['PERCENT']['equity']
        logger.info(f"      [OK] 机制进化完成。DAW 系统公平性帕累托提升: {equity_lift:.4f}")

        # ======================================================================
        # STAGE 6: 成果收割与论文自动化 (Final Deliverables)
        # 目标：生成 Abstract 论据、LaTeX 代码附录
        # ======================================================================
        logger.info(">>> STAGE 6: 执行最终科研成果收割 (Final Harvesting)...")

        # 1. 自动生成摘要核心论据 (Punchlines)
        harvest_helper = AbstractHelper()
        all_metrics = harvest_helper.harvest_all_metrics()
        harvest_helper.generate_punchlines(all_metrics)

        # 2. 自动化代码附录生成 (Exporter)
        project_root = os.path.dirname(os.path.abspath(__file__))
        exporter = MCMProjectExporter(project_root)
        exporter.run()

        # --- 完美落幕 ---
        total_runtime = time.time() - start_wall_clock
        logger.info("=" * 80)
        logger.info(f"   MISSION ACCOMPLISHED | Total Runtime: {total_runtime / 60:.2f} min   ")
        logger.info(f"   Output Artifacts Ready in: reports/   ")
        logger.info("=" * 80)

    except Exception as e:
        logger.critical(f" [CRITICAL ERROR] 全生命周期流水线崩溃: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()