# ==============================================================================
# src/etl/pipeline.py
# Role: Stage 1 Pipeline Orchestrator (The Industrial Assembly Line)
# Function: Sequential execution of parsing, transformation, factor building, and auditing
# ==============================================================================

import logging
import sys
import os
import pandas as pd

# 导入所有自研组件
from src.etl.config_loader import ConfigLoader
from src.etl.loader import DataLoader
from src.etl.parsers import TextParser
from src.etl.transformers import DataTransformer, run_transformations
from src.etl.feature_factory import FeatureFactory, calculate_historical_partner_alpha
from src.etl.validators import DataValidator


def run_etl_stage():
    """
    Stage 1 全流程总控：
    1. Bronze -> 清洗与解析
    2. 物理变换 (Wide to Long)
    3. 统计对齐 (Robust Normalization)
    4. 因子生成 (Partner Alpha, Momentum)
    5. 质量审计 (Logical Validation)
    6. 持久化存储 (Silver & Gold)
    """
    logging.info("=" * 60)
    logging.info("STAGE 1: 启动全流程数据处理流水线 (Bronze -> Silver -> Gold)")
    logging.info("=" * 60)

    try:
        # --- 1. 数据提取 (Extraction) ---
        logging.info("[Step 1/8] 正在提取 Bronze 层原始数据...")
        df_raw = DataLoader.load_bronze_data()

        # --- 2. 文本逻辑解析 (Parsing) ---
        logging.info("[Step 2/8] 执行文本解析与字符串归一化...")
        df_parsed = TextParser.parse_results_column(df_raw)
        df_parsed = TextParser.standardize_strings(df_parsed)

        # --- 3. 核心维度变换 (Transformation) ---
        logging.info("[Step 3/8] 执行长短表变换及统计标准化 (Robust Z-Score)...")
        # 调用 transformers.py 中的集成函数
        df_silver = run_transformations(df_parsed)
        # 增加周度聚合分 (反演引擎必需)
        df_silver = DataTransformer.generate_aggregates(df_silver)

        # --- 4. 自动化审计 (Silver Audit) ---
        logging.info("[Step 4/8] 启动 Silver 层逻辑一致性审计...")
        validator = DataValidator(df_silver)
        if not validator.run_all():
            logging.warning("!!! Silver 层审计未通过，请检查日志 !!!")
        else:
            logging.info("Silver 层审计通过。")

        # --- 5. 持久化 Silver 数据 (Persistence) ---
        DataLoader.save_to_silver(df_silver)

        # --- 6. 因子计算 (Alpha Factory) ---
        logging.info("[Step 6/8] 启动特征工厂：计算 Partner Alpha 与 行业溢价...")
        # 计算舞伴历史胜率因子
        df_with_alpha = calculate_historical_partner_alpha(df_silver)

        # --- 7. 构建黄金因子库 (Gold Layer) ---
        logging.info("[Step 7/8] 正在构建黄金因子库 (Gold Layer)...")
        # 生成动量因子、行业哑变量、竞争环境因子
        df_gold = FeatureFactory.generate_gold_library(df_with_alpha)

        # --- 8. 持久化 Gold 数据 ---
        gold_path = ConfigLoader.get_path('gold_data')
        os.makedirs(os.path.dirname(gold_path), exist_ok=True)
        df_gold.to_csv(gold_path, index=False)
        logging.info(f"[Step 8/8] 黄金因子库已保存至: {gold_path}")

        logging.info("=" * 60)
        logging.info("STAGE 1 任务成功结束！数据地基已固若金汤。")
        logging.info(f"Silver 观测点: {len(df_silver)} | Gold 因子数: {len(df_gold.columns)}")
        logging.info("=" * 60)

        return df_gold

    except Exception as e:
        logging.error(f"FATAL: ETL Pipeline 在 Stage 1 发生不可逆溃败!")
        logging.error(f"错误报告: {str(e)}")
        # 抛出异常供 main.py 捕获，或直接熔断
        raise


# ------------------------------------------------------------------------------
# 脚本入口 (用于单元测试)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # 配置基础控制台日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    run_etl_stage()