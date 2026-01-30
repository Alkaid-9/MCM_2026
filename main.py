# ==============================================================================
# main.py
# Role: Supreme Command Center (Stage 2 Integration)
# Function: Executing ETL -> BIO-Inference -> Quality Summary
# ==============================================================================

import os
import sys
import logging
import pandas as pd
from datetime import datetime

# 环境变量与路径自适应
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from src.etl.pipeline import run_etl_stage
from src.core.bayes_sampler import BayesianVoteInference
from src.etl.config_loader import ConfigLoader


def setup_global_logger():
    """配置工业级双路日志系统"""
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"mcm_full_run_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file


def main():
    log_path = setup_global_logger()
    logger = logging.getLogger("MASTER")

    logger.info("=" * 80)
    logger.info("MCM 2026 PROBLEM C: THE INVISIBLE HAND - 全系统合龙启动")
    logger.info(f"项目根目录: {PROJECT_ROOT}")
    logger.info("=" * 80)

    try:
        # ----------------------------------------------------------------------
        # STAGE 1: 数据地基 (ETL & Factor Engineering)
        # ----------------------------------------------------------------------
        logger.info(">>> [STEP 1] 启动 Stage 1: ETL 与 黄金因子库构建...")
        df_gold = run_etl_stage()

        # 提取 Stage 1 的关键统计量（用于论文展示）
        # 注意：这里假设 run_etl_stage 内部已经打印了 T-test 结果
        logger.info(f"Stage 1 成功。产出因子维度: {df_gold.shape}")

        # ----------------------------------------------------------------------
        # STAGE 2: 逆向反演 (Bayesian Latent Variable Inference)
        # ----------------------------------------------------------------------
        logger.info(">>> [STEP 2] 启动 Stage 2: 贝叶斯逆向优化反演 (Task 1)...")

        sampler = BayesianVoteInference(df_gold)
        df_platinum = sampler.run_inference()

        # 保存最终研究成果：铂金层数据 (Platinum Layer)
        # 铂金层 = 原始数据 + 衍生因子 + 反演票数 + 不确定性度量
        output_path = os.path.join(PROJECT_ROOT, "data", "gold", "final_platinum_results.csv")
        df_platinum.to_csv(output_path, index=False)

        # ----------------------------------------------------------------------
        # 系统审计与统计摘要
        # ----------------------------------------------------------------------
        logger.info("=" * 80)
        logger.info("FINAL SYSTEM SUMMARY")
        logger.info("=" * 80)

        # 1. 优化器表现
        conv_rate = df_platinum['solver_converged'].mean()
        logger.info(f"优化器收敛率 (Solver Convergence): {conv_rate:.2%}")

        # 2. 不确定性摘要
        avg_certainty = df_platinum['est_certainty_score'].mean()
        logger.info(f"平均估计确定性 (Mean Certainty): {avg_certainty:.2%}")

        # 3. 结果抽样（以第27季争议选手 Bobby Bones 为例）
        bobby_bones = df_platinum[
            (df_platinum['celebrity_name'].str.contains("Bones", na=False)) &
            (df_platinum['season'] == 27)
            ]
        if not bobby_bones.empty:
            avg_vote = bobby_bones['est_fan_vote_pct'].mean()
            logger.info(f"重点观测 [Bobby Bones]: 估算平均得票权重 = {avg_vote:.2%}")

        logger.info(f"全流程运行结束。结果已持久化至: {output_path}")
        logger.info(f"运行日志详见: {log_path}")
        logger.info("=" * 80)

    except Exception as e:
        logger.critical(f"系统在运行过程中崩溃: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()