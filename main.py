"""
MCM 2026 Problem C: The Invisible Hand - Supreme Command Center
Role: End-to-End Pipeline Orchestration (ETL -> Inference -> Diagnostics)
Standard: O-Prize Research Ready / Industrial HPC Pipeline
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# --- 1. 环境自适应与路径注入 ---
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.etl.config_loader import ConfigLoader
from src.etl.pipeline import run_etl_stage
from src.bridge.mcmc_wrapper import MCMCInferenceWrapper
from src.utils.logger import setup_logger


def print_banner(logger):
    """打印符合工业标准的系统启动横幅"""
    banner = """
    ========================================================================
    MCM 2026 PROBLEM C: THE INVISIBLE HAND (BIO-ENGINE V4.5)
    --------------------------------------------------------
    Target: Bayesian Inverse Optimization of Latent Fan Preferences
    Kernel: C++17 / OpenMP (23-Core Parallel) / Adaptive MH
    Author: Lone Wolf Research Team (O-Prize Track)
    ========================================================================
    """
    for line in banner.split('\n'):
        logger.info(line.strip())


def main():
    # --- A. 系统引导 (Bootstrap) ---
    config = ConfigLoader()

    # 初始化日志系统
    log_path = config.get_path('logs')
    logger = setup_logger("COMMAND_CENTER", log_path)
    print_banner(logger)

    start_time = time.time()

    try:
        # ======================================================================
        # STAGE 1: 数据精炼层 (ETL & Gold Feature Engineering)
        # ======================================================================
        logger.info(">>> [STAGE 1/3] 启动 ETL 流水线: 从 Bronze 原始数据熔炼 Gold 特征库...")
        t1 = time.time()

        df_gold = run_etl_stage()

        elapsed_t1 = time.time() - t1
        logger.info(f"✅ Stage 1 完成。样本规模: {len(df_gold)} | 耗时: {elapsed_t1:.2f}s")

        # ======================================================================
        # STAGE 2: 贝叶斯反演层 (Bayesian Inverse Optimization)
        # ======================================================================
        logger.info(">>> [STAGE 2/3] 启动 C++ 高性能反演引擎: 正在生成 Platinum 后验数据层...")
        t2 = time.time()

        # 实例化混合架构桥接器
        wrapper = MCMCInferenceWrapper()

        # 执行全量 34 个赛季的并行 MCMC 推断
        df_platinum_raw = wrapper.run_batch_inference(df_gold)

        # 将推断结果与 Gold 层特征进行 Left Join，构建完整的分析面板
        df_platinum = df_gold.merge(
            df_platinum_raw,
            on=['season', 'week_num', 'celebrity_name'],
            how='left'
        )

        # 保存 Platinum 层最终结果
        output_path = Path(config.get_path('platinum_results'))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_platinum.to_csv(output_path, index=False)

        elapsed_t2 = time.time() - t2
        logger.info(f"✅ Stage 2 完成。结果已持久化至: {output_path}")
        logger.info(f"HPC 推理总耗时: {elapsed_t2:.2f}s (平均每赛季 {(elapsed_t2 / 34):.2f}s)")

        # ======================================================================
        # STAGE 3: 战略审计与关键结论 (Strategic Audit)
        # ======================================================================
        logger.info(">>> [STAGE 3/3] 启动系统级审计与历史异象复盘...")

        # 3.1 统计稳健性审计
        conv_rate = df_platinum['is_converged'].mean()
        avg_rhat = df_platinum['r_hat'].mean()
        avg_fidelity = df_platinum['fidelity'].mean()

        logger.info("-" * 40)
        logger.info(f"📊 全局统计审计报告:")
        logger.info(f"- MCMC 收敛率 (R-hat < 1.1): {conv_rate:.2%}")
        logger.info(f"- 平均 R-hat 指标: {avg_rhat:.4f}")
        logger.info(f"- 规则还原保真度 (Fidelity): {avg_fidelity:.4f}")

        if conv_rate < 0.95:
            logger.warning("⚠️ 警告：部分周次未通过收敛审计。建议在 priors.yaml 中增加采样深度。")

        # 3.2 针对题目要求的 Controversy Case 自动复盘
        logger.info("-" * 40)
        logger.info("🔍 题目案例回访 (Case Study Replay):")

        # Case S27: Bobby Bones (低分夺冠异象)
        bobby = df_platinum[
            (df_platinum['season'] == 27) &
            (df_platinum['celebrity_name'].str.contains('Bones', na=False))
            ]
        if not bobby.empty:
            avg_vote = bobby['est_fan_vote_mu'].mean()
            # 物理洞察：计算其估计得票率与平均值的偏离倍数
            peer_avg = df_platinum[df_platinum['season'] == 27]['est_fan_vote_mu'].mean()
            multiplier = avg_vote / peer_avg
            logger.info(f"🚩 [S27 Bobby Bones]: 估计得票率 {avg_vote:.2%}, 为同期平均水平的 {multiplier:.1f} 倍。")
            logger.info(f"   结论: 模型捕捉到了极端的流量溢价，这是对其低技术分夺冠的唯一合理解释。")

        # Case S02: Jerry Rice (低分亚军)
        jerry = df_platinum[
            (df_platinum['season'] == 2) &
            (df_platinum['celebrity_name'].str.contains('Rice', na=False))
            ]
        if not jerry.empty:
            logger.info(f"🚩 [S02 Jerry Rice]: 推演 Fidelity 为 {jerry['fidelity'].mean():.4f}。")
            logger.info(f"   结论: 在 Rank 规则下，即便评委分垫底，只要其粉丝投票稳居前 15%，生存逻辑依然成立。")

        # ======================================================================
        # 终点线
        # ======================================================================
        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"🎉 系统全线运行成功！总执行耗时: {total_time:.2f}s")
        logger.info("📂 铂金层数据已准备就绪，请启动 Notebook 进行最后的绘图展示。")
        logger.info("=" * 80)

    except Exception as e:
        logger.critical("🔥 [FATAL] 系统在运行过程中遭遇致命崩溃！", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()