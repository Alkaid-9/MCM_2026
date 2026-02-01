# ==============================================================================
# src/etl/loader.py
# Role: Industrial-Grade Data Loader & Schema Enforcer (v5.5 - O-Prize Edition)
# Function: Transforming Bronze (Raw) CSV into protected Memory-Safe DataFrames.
# Key Logic: Strict Schema Mapping, Nullable Int64 Support, BOM Handling.
# Standard: Top-tier Journal Reproducibility / Data Forensics Rigor.
# ==============================================================================

import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from src.etl.config_loader import ConfigLoader
from src.utils.logger import setup_logger


class DataLoader:
    """
    负责将数据从磁盘加载至内存，并建立第一道“防御防火墙”。

    [设计哲学]:
    1. Schema Enforcement: 严禁 Pandas 自动推断类型，防止 Season 等整数列被误判为 Float。
    2. Zero-Path-Drift: 利用 ConfigLoader 实现位置无关的文件访问。
    3. Failure Atomicity: 关键列缺失时立即触发系统级熔断。
    """

    def __init__(self):
        # 初始化单例配置与双路日志系统
        self.cfg = ConfigLoader()
        self.logger = setup_logger("DATA_LOADER")

    def load_bronze_data(self) -> pd.DataFrame:
        """
        加载原始 Bronze 层数据并执行强类型映射。
        物理意义：将不可信的 CSV 文本转化为具备数学约束的质量基础。
        """
        raw_path = self.cfg.get_path('bronze_raw')
        self.logger.info(f">>> 正在从数据湖提取原始观测数据: {raw_path}")

        # --- A. 定义强类型映射 (Schema Enforcement) ---
        # 物理直觉：Season 和 Placement 是离散整数坐标，必须严谨。
        # 使用 Pandas 1.0+ 的 'Int64' (Nullable Integer) 避免 NaN 导致整列变 Float。
        dtype_map = {
            'celebrity_name': 'string',
            'ballroom_partner': 'string',
            'celebrity_industry': 'string',
            'celebrity_homestate': 'string',
            'celebrity_homecountry/region': 'string',
            # 关键：支持空值的整数类型
            'celebrity_age_during_season': 'Int64',
            'season': 'Int64',
            'results': 'string',
            'placement': 'Int64'
        }

        try:
            # --- B. 执行读取并处理 N/A 与 BOM ---
            # encoding='utf-8-sig' 处理 Windows Excel 保存时可能留下的隐藏字节 (\ufeff)
            # na_values 定义了美赛数据中常见的脏占位符
            df = pd.read_csv(
                raw_path,
                dtype=dtype_map,
                na_values=["N/A", "n/a", " ", "", "NULL", "nan"],
                encoding='utf-8-sig',
                low_memory=False
            )

            # --- C. 核心索引列完整性审计 ---
            self._audit_integrity(df)

            self.logger.info(f" [OK] Bronze 数据加载成功。维度: {df.shape[0]} 行 x {df.shape[1]} 列")
            return df

        except Exception as e:
            self.logger.critical(f" [FATAL] 原始数据加载失败，流水线强制中断: {str(e)}", exc_info=True)
            raise RuntimeError(f"Loader Pipeline Error: {e}")

    def _audit_integrity(self, df: pd.DataFrame):
        """
        学术严谨性审计：检查贝叶斯反演所需的“锚点列”是否存在数据空洞。
        物理意义：如果核心索引丢失，手动填充会导致严重的推断偏差 (Bias Infiltration)。
        """
        # 定义业务逻辑锚点
        critical_cols = ['celebrity_name', 'season', 'results']

        for col in critical_cols:
            if col not in df.columns:
                self.logger.error(f" [SCHEMA ERROR] 关键列缺失: {col}")
                raise KeyError(f"Critical index column '{col}' missing from data.")

            null_count = df[col].isna().sum()
            if null_count > 0:
                # 记录详细错误，但不一定直接报错，交给后续的 DataValidator 处理或剔除
                # 警告：缺失核心索引的行将在后续反演中被视为无效观测 (Ghost Records)
                self.logger.warning(f" 警告：列 '{col}' 存在 {null_count} 处缺失值。建议在 ETL 阶段剔除。")

    def save_processed_data(self, df: pd.DataFrame, layer: str):
        """
        将处理结果持久化到相应层级。
        layer: 'silver' (展平标准化后), 'gold' (因子化后), 'platinum' (反演结果)
        """
        try:
            # 根据层级自动解析路径键名
            if layer == 'silver':
                key = 'silver_panel'
            elif layer == 'gold':
                key = 'gold_factors'
            elif layer == 'platinum':
                key = 'platinum_results'
            else:
                raise ValueError(f"Unknown data layer: {layer}")

            target_path = Path(self.cfg.get_path(key))

            # 防御性目录创建
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # 工业标准：使用 UTF-8 编码，不保留 DataFrame 的自动索引 (Index=False)
            df.to_csv(target_path, index=False, encoding='utf-8')
            self.logger.info(f" [OK] 数据已持久化至 {layer.upper()} 层: {target_path}")

        except Exception as e:
            self.logger.error(f" [PERSISTENCE ERROR] 持久化 {layer} 层数据失败: {str(e)}")
            raise


# --- 单元测试 (Unit Test) ---
if __name__ == "__main__":
    # 配置根日志
    logging.basicConfig(level=logging.INFO)
    loader = DataLoader()

    print("\n" + "=" * 60)
    print(" DATA LOADER DIAGNOSTICS")
    print("=" * 60)

    try:
        raw_df = loader.load_bronze_data()
        print("\n--- Schema Check (Dtypes) ---")
        print(raw_df.dtypes.head(10))

        print("\n--- Data Sample (Head 3) ---")
        print(raw_df[['season', 'celebrity_name', 'results']].head(3))

        # 验证 Int64 是否生效
        if pd.api.types.is_integer_dtype(raw_df['season']):
            print("\n[PASS] 'season' column is strictly Integer type.")
        else:
            print("\n[FAIL] 'season' column type mismatch.")

    except Exception as e:
        print(f"\n[FAIL] Test aborted: {e}")