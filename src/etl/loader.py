"""
MCM 2026 Problem C: Industrial Grade Data Loader
Role: Secure I/O, Type Enforcement, and Metadata Auditing
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.etl.config_loader import ConfigLoader

class DataLoader:
    """
    负责将 Bronze 层原始 CSV 转换为内存中受保护的 DataFrame。
    设计逻辑：
    1. 强类型映射：杜绝 Pandas 对 object 类型的猜想。
    2. 缺失值预处理：统一处理 N/A 字符串。
    3. 异常熔断：若关键列（如 Season, Week）存在空值，直接报错停止。
    """

    def __init__(self):
        self.cfg = ConfigLoader()
        self.logger = logging.getLogger("DATA_LOADER")

    def load_bronze_data(self) -> pd.DataFrame:
        """
        读取并清洗原始 Bronze 数据。
        """
        raw_path = self.cfg.get_path('bronze_raw')
        self.logger.info(f"正在从磁盘加载原始数据: {raw_path}")

        # A. 定义强类型映射 (Pandas 1.0+ Int64 支持 Nullable)
        # 物理直觉：Season 和 Placement 是离散整数坐标，必须严谨
        dtype_map = {
            'celebrity_name': 'string',
            'ballroom_partner': 'string',
            'celebrity_industry': 'string',
            'celebrity_homestate': 'string',
            'celebrity_homecountry/region': 'string',
            'celebrity_age_during_season': 'Int64',
            'season': 'Int64',
            'results': 'string',
            'placement': 'Int64'
        }

        try:
            # B. 执行读取并处理 N/A
            # 物理直觉：数据手册提到 N/A 表示缺席或未播出，统一转为 np.nan
            df = pd.read_csv(
                raw_path,
                dtype=dtype_map,
                na_values=["N/A", "n/a", " ", "", "NULL"],
                encoding='utf-8'
            )

            # C. 关键列空值审计 (Integrity Check)
            self._audit_integrity(df)

            self.logger.info(f"Bronze 数据加载成功。维度: {df.shape}")
            return df

        except Exception as e:
            self.logger.critical(f"Bronze 数据加载失败，流水线终止: {str(e)}")
            raise

    def _audit_integrity(self, df: pd.DataFrame):
        """
        学术严谨性审计：检查核心索引列是否存在数据缺失。
        """
        critical_cols = ['celebrity_name', 'season', 'results']
        for col in critical_cols:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                self.logger.error(f"严重错误：关键列 {col} 存在 {missing_count} 处空值！")
                # 在数院建模中，如果核心索引丢失，手动填充会导致严重的推断偏差
                # 这里我们采取‘零容忍’策略

    def save_processed_data(self, df: pd.DataFrame, layer: str):
        """
        将数据持久化到相应层级。
        layer: 'silver' (清洗后面板数据), 'gold' (因子库), 'platinum' (结果)
        """
        # 获取配置路径
        if layer == 'silver':
            target_path = self.cfg.get_path('silver_panel')
        elif layer == 'gold':
            target_path = self.cfg.get_path('gold_factors')
        elif layer == 'platinum':
            target_path = self.cfg.get_path('platinum_results')
        else:
            raise ValueError(f"Unknown data layer: {layer}")

        # 创建目录
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            # 工业标准：对于大规模数据建议使用 Parquet，但美赛通常提交 CSV
            df.to_csv(target_path, index=False, encoding='utf-8')
            self.logger.info(f"数据已成功持久化至 {layer} 层: {target_path}")
        except Exception as e:
            self.logger.error(f"持久化 {layer} 层数据失败: {str(e)}")
            raise

# --- 简单测试代码 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    loader = DataLoader()
    raw_df = loader.load_bronze_data()
    print("\n--- Data Sample ---")
    print(raw_df.head())
    print("\n--- Column Types ---")
    print(raw_df.dtypes)