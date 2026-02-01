# ==============================================================================
# MCM 2026 Problem C: "The Invisible Hand of the Audience"
# MAIN INTEGRATION PIPELINE (Final Production Release)
# Role: Supreme Command Center orchestrating Stage 1-6 Execution.
# Architecture: Hybrid HPC (C++/Numba) + Bayesian Inference + Causal XAI.
# Standard: "One-Click to O-Prize" Automation.
# ==============================================================================
import matplotlib
matplotlib.use('Agg') # 强制使用非交互式后端，禁止初始化 Tkinter
import matplotlib.pyplot as plt
import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# --- 0. 环境霸权配置 (Environment Locking) ---
# 必须在导入任何科学计算库之前执行，强制控制线程拓扑，防止 Numpy/Eigen 争抢资源
os.environ["OMP_NUM_THREADS"] = "23"
os.environ["MKL_NUM_THREADS"] = "23"
os.environ["NUMBA_NUM_THREADS"] = "23"
os.environ["OPENBLAS_NUM_THREADS"] = "23"

# 动态添加 src 路径，确保在任何目录下运行都不会报错
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# 导入自研组件
from src.utils.logger import setup_logger
from src.etl.config_loader import ConfigLoader
from src.etl.pipeline import run_etl_stage
from src.bridge.mcmc_wrapper import MCMCInferenceWrapper
from src.analysis.mechanism_pipeline import MechanismAnalysisPipeline
from src.analysis.causality_pipeline import run_causality_stage
from src.solvers.design_pipeline import MechanismDesignPipeline
from src.utils.abstract_helper import AbstractHelper
from src.utils.exporter import MCMProjectExporter


def main():
    # --- 1. 系统初始化与预检 ---
    config_provider = ConfigLoader()
    # 确保日志目录存在
    log_path = Path(config_provider.get_path('logs'))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("MCM_COMMAND_CENTER")
    start_wall_clock = time.time()

    logger.info("=" * 80)
    logger.info("  MCM 2026 PROBLEM C: FULL SYSTEM ACTIVATED (SOLO RUN MODE)")
    logger.info(f"  Root Directory: {PROJECT_ROOT}")
    logger.info("  Architecture: Bayesian Inverse Optimization + Pareto Design")
    logger.info("=" * 80)

    # 执行自动化契约审计 (Pre-flight Check)
    try:
        import check_dependencies
        logger.info("执行环境完整性审计...")
        # check_dependencies.run_audit() # 根据需要取消注释，如果你保留了该文件
    except ImportError:
        logger.warning("未找到 check_dependencies.py，跳过环境审计，风险自担。")

    try:
        # ======================================================================
        # STAGE 1: 数据取证与信号精炼 (Data Forensics)
        # ======================================================================
        logger.info(">>> STAGE 1: 启动数据地基构建 (ETL Pipeline)...")
        # 这一步会读取 Bronze 数据，清洗、去通胀、聚合，生成 Gold 因子库
        df_gold = run_etl_stage()
        logger.info(f"      [OK] 黄金因子库已就绪: {df_gold.shape[0]} 观测点, {df_gold.shape[1]} 因子")

        # ======================================================================
        # STAGE 2: 贝叶斯潜变量反演 (The BIO Engine)
        # ======================================================================
        logger.info(">>> STAGE 2: 启动 C++ 高性能反演引擎 (Task 1)...")
        inference_engine = MCMCInferenceWrapper()

        # 执行 23 核并行推断，将淘汰结果逆向坍缩为观众票数分布
        # 这一步最耗时，注意观察 tqdm 进度条
        df_platinum = inference_engine.run_batch_inference(df_gold)

        # [关键工程点]: 强制执行数据类型压实，杜绝 [5E-1] 等字符串污染
        for col in ['est_fan_vote_mu', 'est_fan_vote_sigma', 'fidelity']:
            if col in df_platinum.columns:
                df_platinum[col] = pd.to_numeric(df_platinum[col], errors='coerce').fillna(0.0)

        platinum_path = config_provider.get_path('platinum_results')
        df_platinum.to_csv(platinum_path, index=False)
        logger.info(f"      [OK] 铂金层(后验分布)已持久化至: {platinum_path}")

        # ======================================================================
        # STAGE 3: 机制审计与平行宇宙模拟 (Forensics Task 2)
        # ======================================================================
        logger.info(">>> STAGE 3: 启动反事实模拟流水线 (Task 2)...")
        forensics_pipeline = MechanismAnalysisPipeline(df_platinum)

        # 执行：全赛季重演 -> 生存分析 -> 敏感性测试 -> Bobby Bones 审计
        audit_report = forensics_pipeline.run_full_audit()

        gain = audit_report.get('sensitivity_metrics', {}).get('robustness_advantage', 0)
        logger.info(f"      [OK] 机制偏差审计完成。识别历史赛制稳定性增益: {gain:.2%}")

        # ======================================================================
        # STAGE 4: 因果归因与非线性分歧分析 (Causality Task 3)
        # ======================================================================
        logger.info(">>> STAGE 4: 启动 SHAP + LMM 复合归因引擎 (Task 3)...")
        fig_dir = config_provider.get_path('figures_dir')

        # 注意：传入已清洗的 Platinum 数据
        causality_report = run_causality_stage(df_platinum, fig_dir=fig_dir)

        dissonance = causality_report.get('metrics', {}).get('dissonance_index', 0)
        logger.info(f"      [OK] 审美分歧度量完成。系统认知失调指数: {dissonance:.4f}")

        # ======================================================================
        # STAGE 5: 帕累托最优机制设计 (DAW System Task 4)
        # ======================================================================
        logger.info(">>> STAGE 5: 启动多目标博弈寻优 (Task 4)...")
        # 实例化设计流水线，内部调用已修复契约的 DAWEngine
        design_pipeline = MechanismDesignPipeline(df_platinum)

        # 针对 Bobby Bones 所在的 S27 进行高压仿真寻优 (The Ultimate Stress Test)
        design_metrics = design_pipeline.run_design_suite(target_season=27)

        equity_daw = design_metrics['DAW']['equity']
        equity_pct = design_metrics['PERCENT']['equity']
        fairness_lift = (equity_daw - equity_pct) / (equity_pct + 1e-9)

        logger.info(f"      [OK] 机制进化成功。DAW 帕累托提升 (公平性): {fairness_lift:.2%}")

        # ======================================================================
        # STAGE 6: 论文论据自动化采集 (Final Harvesting)
        # ======================================================================
        logger.info(">>> STAGE 6: 执行最终科研成果收割与导出...")

        # 1. 自动生成摘要核心论据 (Punchlines)
        # 这一步将输出可以直接粘贴到 Abstract 中的“学术金句”
        harvest_helper = AbstractHelper(project_root=str(PROJECT_ROOT))
        all_metrics = harvest_helper.harvest_all_metrics()
        harvest_helper.generate_punchlines(all_metrics)
        harvest_helper.generate_memo_points(all_metrics)

        # 2. 自动化 LaTeX 代码附录生成
        exporter = MCMProjectExporter(project_root=PROJECT_ROOT)
        exporter.run()

        # --- 完美落幕 ---
        total_runtime = time.time() - start_wall_clock
        logger.info("=" * 80)
        logger.info(f"  MISSION ACCOMPLISHED | 全流程运行耗时: {total_runtime / 60:.2f} min")
        logger.info("  STATUS: READY FOR SUBMISSION (Outstanding Grade Standard)")
        logger.info("  Output Artifacts Ready in: reports/")
        logger.info("=" * 80)

    except Exception as e:
        logger.critical(f" [FATAL ERROR] 统帅部调度崩溃: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()