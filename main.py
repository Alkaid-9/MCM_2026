# ==============================================================================
# main.py
# Role: Supreme Command Center (MCM 2026 Problem C)
# Function: End-to-End Orchestration (ETL -> BIO-Inference -> Strategic Audit)
# Author: "The Invisible Hand" Team
# Standard: Industrial O-Prize Architecture
# ==============================================================================

import os
import sys
import time
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

# --- 1. 环境自适应与路径注入 ---
# 确保无论在哪个目录下运行，都能找到 src 和 conf
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# --- 2. 模块导入 (Lazy Import 防止初始化开销) ---
from src.utils.logger import setup_logger
from src.etl.config_loader import ConfigLoader
from src.etl.pipeline import run_etl_stage
from src.bridge.mcmc_wrapper import MCMCInferenceWrapper


def print_banner(logger):
    """打印系统启动横幅"""
    banner = """
    ========================================================================
       MCM 2026 PROBLEM C: THE INVISIBLE HAND (BIO-ENGINE V4.0)
       --------------------------------------------------------
       Target: Bayesian Inverse Optimization of Latent Fan Preferences
       Kernel: C++17 / OpenMP (23-Core Parallel) / Dual-Averaging NUTS
    ========================================================================
    """
    for line in banner.split('\n'):
        logger.info(line.strip())


def main():
    # --- A. 系统引导 (Bootstrap) ---
    # 初始化配置加载器 (单例)
    config_loader = ConfigLoader()

    # 初始化日志系统
    log_path = config_loader.get_path('logs')
    # 动态生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_path).parent / f"mcm_run_{timestamp}.log"

    logger = setup_logger("COMMAND_CENTER", str(log_file))
    print_banner(logger)

    start_time = time.time()

    try:
        # ======================================================================
        # STAGE 1: 数据精炼与因子工程 (ETL & Factor Engineering)
        # ======================================================================
        logger.info(">>> [STAGE 1/3] 启动 ETL 流水线 (Data Refinery)...")
        t1 = time.time()

        # 调用 ETL 编排器
        df_gold = run_etl_stage()

        # 阶段性审计
        if df_gold is None or df_gold.empty:
            logger.critical("ETL 阶段产出为空，系统熔断！")
            sys.exit(1)

        elapsed_t1 = time.time() - t1
        logger.info(f"Stage 1 完成。耗时: {elapsed_t1:.2f}s | 因子库维度: {df_gold.shape}")
        logger.info(f"包含因子: {list(df_gold.columns)}")

        # ======================================================================
        # STAGE 2: 贝叶斯逆向演化 (Bayesian Inverse Optimization)
        # ======================================================================
        logger.info(">>> [STAGE 2/3] 启动 C++ 高性能反演引擎 (BIO-Kernel)...")
        t2 = time.time()

        # 实例化 Python-C++ 桥接器
        inference_engine = MCMCInferenceWrapper()

        # 执行批量反演 (Batch Inference)
        # 这里会调用 C++ 的 run_parallel_inference
        df_platinum = inference_engine.run_batch_inference(df_gold)

        # 结果持久化 (Platinum Layer)
        output_path = PROJECT_ROOT / "data" / "platinum" / "final_posterior_results.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_platinum.to_csv(output_path, index=False)

        elapsed_t2 = time.time() - t2
        logger.info(f"Stage 2 完成。耗时: {elapsed_t2:.2f}s")
        logger.info(f"核心产物已保存至: {output_path}")

        # ======================================================================
        # STAGE 3: 战略审计与关键结论 (Strategic Audit)
        # ======================================================================
        logger.info(">>> [STAGE 3/3] 执行系统级审计与关键案例复盘...")

        # 3.1 全局收敛性审计
        convergence_rate = df_platinum['mcmc_converged'].mean()
        avg_fidelity = df_platinum['fidelity_score'].mean()
        avg_r_hat = df_platinum['r_hat_max'].mean()

        logger.info("--- Global Diagnostics ---")
        logger.info(f"优化器收敛率 (R-hat < 1.1): {convergence_rate:.2%}")
        logger.info(f"历史还原保真度 (Fidelity):   {avg_fidelity:.2%}")
        logger.info(f"平均 R-hat (Max per week):    {avg_r_hat:.4f}")

        if convergence_rate < 0.95:
            logger.warning("警告: 全局收敛率低于 95%，建议增加采样步数 (n_samples) 或调整先验。")

        # 3.2 关键案例分析 (针对题目要求的 Controversy)
        # Case A: Bobby Bones (Season 27) - 低分夺冠
        bobby = df_platinum[
            (df_platinum['season'] == 27) &
            (df_platinum['celebrity_name'].str.contains('Bones', na=False))
            ]
        if not bobby.empty:
            avg_vote_share = bobby['est_fan_vote_mu'].mean()
            logger.info("--- Case Study: Bobby Bones (S27) ---")
            logger.info(f"平均估计粉丝得票率: {avg_vote_share:.2%}")
            logger.info(f"该案例 Fidelity: {bobby['fidelity_score'].mean():.2%}")
            if avg_vote_share > 0.40:
                logger.info("结论: 模型确认 Bobby Bones 拥有压倒性的粉丝基数 (Superstar Effect)。")

        # Case B: Jerry Rice (Season 2) - 争议亚军
        jerry = df_platinum[
            (df_platinum['season'] == 2) &
            (df_platinum['celebrity_name'].str.contains('Rice', na=False))
            ]
        if not jerry.empty:
            logger.info("--- Case Study: Jerry Rice (S2) ---")
            logger.info(f"平均估计粉丝得票率: {jerry['est_fan_vote_mu'].mean():.2%}")

        # ======================================================================
        # END: 系统关闭
        # ======================================================================
        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"全系统运行成功。总耗时: {total_time:.2f}s")
        logger.info("O-Prize 论文所需数据已准备就绪。")
        logger.info("=" * 80)

    except Exception as e:
        logger.critical("系统在运行过程中发生致命错误！", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()