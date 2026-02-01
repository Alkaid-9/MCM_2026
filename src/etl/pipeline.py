# ==============================================================================
# src/etl/pipeline.py
# Role: ETL Stage Orchestrator (The Industrial Assembly Line v5.5)
# Function: Sequential execution of Data Archeology, Signal Refining & Factor Building.
# Physics: Transforming noisy human-scored observations into a high-SNR feature manifold.
# Standard: O-Prize Quality / Top-tier Journal Reproducibility / SNR Analytics.
# ==============================================================================

import logging
import pandas as pd
import sys
import time
import numpy as np
from pathlib import Path

# 路径自适应：确保在任何目录下都能找到 src
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- 核心组件导入 ---
from src.utils.logger import setup_logger
from src.etl.config_loader import ConfigLoader
from src.etl.loader import DataLoader
from src.etl.parsers import TextParser
from src.etl.transformers import run_transform_pipeline
from src.etl.validators import DataValidator
from src.etl.feature_factory import FeatureFactory


def run_etl_stage() -> pd.DataFrame:
    """
    Stage 1 核心调度器：将 34 个赛季的非结构化原始数据转化为贝叶斯反演所需的“黄金因子库”。

    [学术逻辑流程]:
    1. Data Ingestion: 强类型读取，建立内存防御工事。
    2. Data Archeology: 解析文本描述，构建生存截断(Censorship)标签。
    3. Signal Refining: Wide-to-Long 变换，执行 Recursive Robust Z-Score 标准化。
    4. Forensics: 执行制度断裂检验 (Structural Break Test)，实证规则变更的显著性。
    5. Aggregation: 将评委级数据压缩为选手-周级数据 (解决 KeyError 的关键)。
    6. Alpha Generation: 构建因果因子（舞伴红利、动量），实施因果防火墙。
    7. System Audit: 最终红线审计，确保数据集满足贝叶斯推断的数学前置条件。
    """

    # 1. 初始化单例与日志
    config = ConfigLoader()
    logger = setup_logger("ETL_PIPELINE")
    start_time = time.time()

    logger.info("=" * 80)
    logger.info(">>> STAGE 1 启动: 工业级数据基建与信号工程流水线 <<<")
    logger.info("=" * 80)

    try:
        # --- A. 基础设施实例化 ---
        loader = DataLoader()
        parser = TextParser()
        factory = FeatureFactory()

        # --- STEP 1: 原始数据提取 (Bronze Layer) ---
        logger.info("[Step 1/6] 正在提取原始观测记录 (Bronze Data Ingestion)...")
        df_raw = loader.load_bronze_data()

        # --- STEP 2: 文本精炼与生存分析标签 (Data Archeology) ---
        # 物理意义：识别右删失(Right-Censored)点，这是量化推断的统计学底座
        logger.info("[Step 2/6] 执行实体标准化与 Censorship 屏障构建...")
        df_standardized = parser.standardize_entities(df_raw)
        df_parsed = parser.parse_survival_results(df_standardized)

        # --- STEP 3: 信号变换与周度聚合 (Silver Layer) ---
        # 【关键逻辑】：在此处完成从“评委级”到“选手-周级”的跨越
        # run_transform_pipeline 内部包含了：宽转长 -> 生存过滤 -> 去通胀 -> 制度审计 -> 聚合
        logger.info("[Step 3/6] 执行信号变换：Wide-to-Long & Robust Normalization...")

        # 返回的是已聚合的 DataFrame，包含 'week_avg_score', 'tech_rank' 等核心列
        df_silver = run_transform_pipeline(df_parsed)

        # 持久化中间产物：Silver 层供机制对比分析(Task 2)使用
        loader.save_processed_data(df_silver, layer='silver')

        # --- STEP 4: 高阶因子工程 (Gold Layer Generation) ---
        # 【物理意义】：将统计信号炼制为因果特征。实施“因果防火墙”防止前瞻偏差。
        logger.info("[Step 4/6] 正在构建黄金因子库 (Partner Alpha / Score Momentum)...")

        # factory 现在接收的是干净的聚合表，生成 Alpha, Acceleration 等关键因子
        df_gold = factory.generate_gold_library(df_silver)

        # --- STEP 5: 逻辑一致性红线审计 (System Audit) ---
        # 【学术底线】：在进入 MCMC 引擎前，必须排除所有逻辑悖论（如排名单调性违规）
        logger.info("[Step 5/6] 启动数学前置条件审计与一致性体检...")
        validator = DataValidator(df_gold)

        if not validator.run_all():
            logger.critical(" [FATAL] 数据审计未通过：检测到核心逻辑冲突，Pipeline 强制熔断！")
            raise ValueError("Data Integrity Violation: The dataset is mathematically inconsistent for MCMC.")

        # --- STEP 6: 特征正交性审计 (Task 3 Preliminary) ---
        # 物理意义：量化“技术分”与“舞伴加成”的独立性，预警共线性风险。
        if 'partner_alpha' in df_gold.columns and 'week_avg_score' in df_gold.columns:
            correlation = df_gold[['partner_alpha', 'week_avg_score']].corr().iloc[0, 1]
            logger.info(f" [Forensics] 因子正交性检测：Partner_Alpha 与 Tech_Score 相关性 = {correlation:.4f}")

            if abs(correlation) > 0.80:
                logger.warning(" [ANALYSIS WARN] 因子共线性过高，后期回归需采用混合效应模型 (LMM) 剥离。")

        # --- STEP 7: 结果持久化 (Gold Layer) ---
        logger.info("[Step 6/6] 黄金因子库落盘存档 (Gold Data Lake)...")
        loader.save_processed_data(df_gold, layer='gold')

        # --- 打印运行摘要与战果 ---
        elapsed = time.time() - start_time
        logger.info("=" * 80)
        logger.info("✅ STAGE 1 全流程合龙成功！")
        logger.info(f" 耗时: {elapsed:.2f}s")
        logger.info(f" 最终观测样本数: {len(df_gold)} (Unique Athlete-Week pairs)")
        logger.info(f" 黄金因子总维度: {df_gold.shape[1]} (Contains Latent Prior Anchors)")
        logger.info(f" 交付路径: {config.get_path('gold_factors')}")
        logger.info("=" * 80)

        return df_gold

    except Exception as e:
        logger.critical(f"🔥 ETL 阶段发生致命崩溃，指挥部已拦截: {str(e)}", exc_info=True)
        # 在工业级系统中，我们选择主动抛出异常以阻止错误数据进入下一阶段
        raise RuntimeError("ETL Pipeline Failure")


# --- 独立运行入口 ---
if __name__ == "__main__":
    # 执行流水线点火
    try:
        run_etl_stage()
    except Exception:
        sys.exit(1)