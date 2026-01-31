# ==============================================================================
# src/etl/pipeline.py
# Role: ETL Stage Orchestrator (The Industrial Assembly Line)
# Function: Sequential execution of data loading, parsing, transformation,
#           forensics, and factor building.
# Standard: O-Prize Quality / Production-Grade Robustness.
# ==============================================================================

import logging
import pandas as pd
import sys
from pathlib import Path

# 路径自适应：确保模块导入路径正确
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.etl.config_loader import ConfigLoader
from src.etl.loader import DataLoader
from src.etl.parsers import TextParser
from src.etl.transformers import run_transform_pipeline
from src.etl.validators import DataValidator
from src.etl.feature_factory import FeatureFactory


def run_etl_stage() -> pd.DataFrame:
    """
    Stage 1 全流程调度：从 Bronze (原始数据) 生产出 Gold (因子库)。

    【核心改进】：
    1. 颗粒度对齐：确保从 Transformers 出来的 df_silver 是选手-周级的聚合表。
    2. 解决 KeyError：通过先聚合、再因子化的顺序，确保 'week_avg_score' 存在。
    3. 逻辑分层：清晰定义了从原始清洗到统计取证，再到特征工程的流水线逻辑。
    """
    logger = logging.getLogger("ETL_PIPELINE")
    logger.info("=" * 80)
    logger.info(">>> STAGE 1 启动: 数据基建与黄金因子库构建 <<<")
    logger.info("=" * 80)

    try:
        # --- A. 实例化工程组件 ---
        config = ConfigLoader()
        loader = DataLoader()
        parser = TextParser()
        factory = FeatureFactory()

        # --- STEP 1: 原始数据提取 (Bronze Layer) ---
        logger.info("[Step 1/6] 加载强类型原始数据...")
        df_raw = loader.load_bronze_data()

        # --- STEP 2: 文本精炼与生存标签解析 ---
        logger.info("[Step 2/6] 执行实体标准化与 Censorship 标记提取...")
        df_parsed = parser.standardize_entities(df_raw)
        df_parsed = parser.parse_survival_labels(df_parsed)

        # --- STEP 3: 核心转换与周度信号聚合 (Crucial Transformation) ---
        # 【物理意义】：这里完成了从“评委级长表”向“选手级聚合表”的跨越
        # 返回的 df_agg 已经包含了 'week_avg_score'，这是解决 KeyError 的关键
        logger.info("[Step 3/6] 执行维度展平、标准化与信号聚合...")
        df_agg_candidate = run_transform_pipeline(df_parsed)

        # 将聚合后的初步数据持久化为 Silver 层（供审计使用）
        loader.save_processed_data(df_agg_candidate, layer='silver')

        # --- STEP 4: 因子工程 (Gold Layer Generation) ---
        # 【物理意义】：在纯净的聚合信号上构建动量因子、红利因子、背景因子
        logger.info("[Step 4/6] 正在基于聚合信号构建黄金因子库 (Alpha/Momentum)...")
        # factory 现在接收到的是包含 week_avg_score 的聚合表，逻辑完美闭环
        df_gold = factory.generate_gold_library(df_agg_candidate)

        # --- STEP 5: 数据质量红线审计 ---
        logger.info("[Step 5/6] 启动数学前置条件审计与一致性检查...")
        validator = DataValidator(df_gold)
        if not validator.run_all():
            logger.critical("❌ 数据审计失败：检测到核心逻辑冲突，Pipeline 强制熔断！")
            raise ValueError("Data Integrity Violation in ETL Stage.")

        # 因子正交性审计 (回答 Task 3 关注点)
        # 物理意义：验证“技术分”与“舞伴加成”是否具有独立的解释空间
        if 'partner_alpha' in df_gold.columns and 'week_avg_score' in df_gold.columns:
            correlation = df_gold[['partner_alpha', 'week_avg_score']].corr().iloc[0, 1]
            logger.info(f"📊 因子正交性审计：Partner_Alpha 与 Week_Score 相关性 = {correlation:.4f}")
            if abs(correlation) > 0.85:
                logger.warning("⚠️ 因子共线性过高，建议在回归模型中引入正则化。")

        # --- STEP 6: 结果持久化 (Gold Layer) ---
        logger.info("[Step 6/6] 黄金因子库落盘...")
        loader.save_processed_data(df_gold, layer='gold')

        # --- 打印运行摘要 ---
        logger.info("=" * 80)
        logger.info("✅ STAGE 1 成功结束！")
        logger.info(f"观测样本总数 (选手-周): {len(df_gold)}")
        logger.info(f"因子特征总维度:      {df_gold.shape[1]}")
        logger.info(f"输出路径:           {config.get_path('gold_factors')}")
        logger.info("=" * 80)

        return df_gold

    except Exception as e:
        logger.critical(f"🔥 ETL 阶段发生致命崩溃: {str(e)}", exc_info=True)
        raise


# --- 独立运行入口 ---
if __name__ == "__main__":
    # 配置控制台日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    # 点火运行
    df_result = run_etl_stage()