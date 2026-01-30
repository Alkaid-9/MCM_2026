# ==============================================================================
# src/etl/loader.py
# Role: I/O & Type Enforcement Layer (Industrial Standard)
# Function: Securely reading bronze data and persisting silver/gold datasets
# ==============================================================================

import pandas as pd
import logging
from src.etl.config_loader import ConfigLoader


class DataLoader:
    """
    负责数据的物理读取与存储。确保进入内存的数据类型符合模型预期。
    """

    @staticmethod
    def load_bronze_data() -> pd.DataFrame:
        """
        加载原始 Bronze 数据。
        核心点：处理 N/A 字符并强制指定数据类型。
        """
        path = ConfigLoader.get_path('bronze_data')
        logging.info(f"正在从 {path} 读取原始数据...")

        # 定义强制类型转换映射，防止 Pandas 乱推断
        # 尤其是 season 和 placement，必须是整型或可转整型的
        dtype_map = {
            'season': 'Int64',  # 使用 Pandas 可空整型
            'placement': 'Int64',
            'celebrity_age_during_season': 'Int64'
        }

        try:
            # 原始数据中包含 "N/A" 字符串，必须显式指定为 na_values
            df = pd.read_csv(
                path,
                na_values=["N/A", "n/a", " ", ""],
                dtype=dtype_map,
                encoding='utf-8'  # 显式指定编码防止 Windows 环境报错
            )
            logging.info(f"Bronze 数据加载成功，Shape: {df.shape}")
            return df
        except FileNotFoundError:
            logging.error(f"找不到 Bronze 文件: {path}")
            raise
        except Exception as e:
            logging.error(f"读取 CSV 过程发生未知错误: {str(e)}")
            raise

    @staticmethod
    def save_to_silver(df: pd.DataFrame):
        """
        将清洗后的数据持久化到 Silver 层。
        """
        path = ConfigLoader.get_path('silver_data')
        try:
            df.to_csv(path, index=False, encoding='utf-8')
            logging.info(f"Silver 数据已持久化至: {path}")
        except Exception as e:
            logging.error(f"写入 Silver 数据失败: {str(e)}")
            raise


# ------------------------------------------------------------------------------
# 单元测试逻辑
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # 配置简单的日志输出以便测试
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    loader = DataLoader()
    raw_df = loader.load_bronze_data()
    print(raw_df.head())
    print("\nColumn Dtypes:\n", raw_df.dtypes)