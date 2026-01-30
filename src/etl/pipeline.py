"""
MCM 2026 Problem C: ETL Stage Orchestrator (The Industrial Assembly Line)
Role: Sequential execution of data loading, parsing, transformation, factor building, and auditing.
Standard: O-Prize Quality / Production-Grade Robustness.
"""

import logging
import pandas as pd
from src.etl.config_loader import ConfigLoader
from src.etl.loader import DataLoader
from src.etl.parsers import TextParser
from src.etl.transformers import run_transform_pipeline
from src.etl.validators import DataValidator
from src.etl.feature_factory import FeatureFactory


def run_etl_stage() -> pd.DataFrame:
    """
    Stage 1 全流程调度：从 Bronze (原始数据) 生产出 Gold (因子库)。

    流水线工序：
    1. Extraction: 加载强类型原始数据。
    2. Parsing: 执行文本解析与异常校正（处理 Week 110 等脏数据）。
    3. Transformation: 维度变换、评委映射、Robust Z-Score 去通胀。
    4. Auditing: 逻辑一致性红线审计（失败则熔断）。
    5. Factor Engineering: 生成 Alpha 因子与归因特征。
    6. Persistence: 保存 Silver (清洗后) 与 Gold (因子库) 数据。
    """
    logger = logging.getLogger("ETL_PIPELINE")
    logger.info("=" * 80)
    logger.info(">>> STAGE 1 启动: 数据基建与黄金因子库构建 <<<")
    logger.info("=" * 80)

    try:
        # 实例化组件
        loader = DataLoader()
        parser = TextParser()
        factory = FeatureFactory()

        # --- STEP 1: 数据提取 (Bronze Layer) ---
        logger.info("[Step 1/6] 正在提取 Bronze 原始数据...")
        df_bronze = loader.load_bronze_data()

        # --- STEP 2: 文本解析与清洗 (Refining) ---
        logger.info("[Step 2/6] 执行实体标准化与生存标签解析...")
        df_parsed = parser.standardize_entities(df_bronze)
        df_parsed = parser.parse_survival_labels(df_parsed)

        # --- STEP 3: 核心统计变换 (Transformation) ---
        logger.info("[Step 3/6] 执行 Wide-to-Long 变换与单集标准化 (去通胀)...")
        # 这里调用了我们在 transformers.py 中重构的高级流水线
        df_silver = run_transform_pipeline(df_parsed)

        # --- STEP 4: 数据质量红线审计 (Validation) ---
        logger.info("[Step 4/6] 启动逻辑一致性红线审计 (Silver Layer Audit)...")
        validator = DataValidator(df_silver)
        if not validator.run_all():
            logger.critical("❌ 数据审计未通过！检测到逻辑冲突，Pipeline 强制熔断。")
            raise ValueError("Data Integrity Violation in Silver Layer.")

        # 保存 Silver 层数据（清洗完成，待因子化）
        loader.save_processed_data(df_silver, layer='silver')

        # --- STEP 5: 高阶因子构建 (Alpha Generation) ---
        logger.info("[Step 5/6] 正在构建黄金因子库 (Partner Alpha, Momentum, SHAP Features)...")
        # 物理意义：将统计信号转化为因果特征
        df_gold = factory.generate_gold_library(df_silver)

        # --- STEP 6: 数据持久化 (Gold Layer) ---
        logger.info("[Step 6/6] 正在将黄金因子库持久化至磁盘...")
        loader.save_processed_data(df_gold, layer='gold')

        logger.info("=" * 80)
        logger.info(f"✅ STAGE 1 成功结束！")
        logger.info(f"   最终观测点数量: {len(df_gold)}")
        logger.info(f"   特征总维度: {df_gold.shape[1]}")
        logger.info(f"   输出路径: {ConfigLoader().get_path('gold_factors')}")
        logger.info("=" * 80)

        return df_gold

    except Exception as e:
        logger.critical(f"💥 ETL 阶段发生致命崩溃: {str(e)}", exc_info=True)
        raise

# 逻辑：检查 Partner Alpha 和原始分数的相关性
# 如果相关性 > 0.9，说明因子是冗余的。
# 但在这里，Alpha 是历史值，Score 是当前值，理论上是低相关的，这证明Alpha具有独立预测能力。
correlation = gold_df[['partner_alpha', 'week_avg_score']].corr().iloc[0, 1]
logger.info(f"因子正交性审计：Alpha 与 Score 相关性 = {correlation:.4f}")

if __name__ == "__main__":
    # 配置根日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    run_etl_stage()